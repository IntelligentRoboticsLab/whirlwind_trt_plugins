#pragma once

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <string>
#include <vector>

namespace liteany {

// Fused cost_stem_3d plugin. Replaces three (Conv3D 3x3x3 + BN + Clip[0,6])
// stages running over a [1, 1, 48, 32, 40] cost-volume tensor with a single
// fused CUDA kernel that holds intermediates in shared memory.
//
// The TRT-generated graph runs the three layers as 18 sm50_xmma_conv3d_c1_k1
// kernel calls (sm50 fallback because the shape is too small/odd for the
// sm80 tile configurations), totalling ~67 µs at s025. A custom fused
// kernel cuts this to launch-overhead-floor (~2-3 µs).
//
// The plugin holds the BN-folded weights/biases as serialised plugin
// attributes; the kernel reads them from constant/global memory.

void launchCostStem3D(
    const void* input,         // fp16 [B, 1, D, H, W]
    void* output,              // fp16 [B, 1, D, H, W]
    const void* w0,            // fp16 [4, 1, 3, 3, 3]
    const void* b0,            // fp16 [4]
    const void* w1,            // fp16 [4, 4, 3, 3, 3]
    const void* b1,            // fp16 [4]
    const void* w2,            // fp16 [1, 4, 3, 3, 3]
    const void* b2,            // fp16 [1]
    int B, int D, int H, int W,
    cudaStream_t stream);

class CostStem3DPlugin final : public nvinfer1::IPluginV3,
                               public nvinfer1::IPluginV3OneCore,
                               public nvinfer1::IPluginV3OneBuild,
                               public nvinfer1::IPluginV3OneRuntime {
public:
    // Receives folded weights+biases from the ONNX symbolic. Sizes are
    // fixed by the model (108 + 4 + 432 + 4 + 108 + 1 = 657 fp16 elements).
    CostStem3DPlugin(const std::vector<uint16_t>& w0,
                     const std::vector<uint16_t>& b0,
                     const std::vector<uint16_t>& w1,
                     const std::vector<uint16_t>& b1,
                     const std::vector<uint16_t>& w2,
                     const std::vector<uint16_t>& b2);
    ~CostStem3DPlugin() override;

    // ---- IPluginV3 ----
    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    // ---- IPluginV3OneCore ----
    nvinfer1::AsciiChar const* getPluginName() const noexcept override;
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override;
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(const char* ns) noexcept;

    // ---- IPluginV3OneBuild ----
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    int32_t getOutputDataTypes(
        nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(
        nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override;

    // ---- IPluginV3OneRuntime ----
    int32_t onShapeChange(
        nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept override { return 0; }

    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;

    nvinfer1::IPluginV3* attachToContext(
        nvinfer1::IPluginResourceContext* context) noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    // CPU copies of the weights — kept for serialisation and clone.
    std::vector<uint16_t> mW0, mB0, mW1, mB1, mW2, mB2;

    // Device copies — allocated lazily on first enqueue, freed on destroy.
    void* mDevW0 = nullptr;
    void* mDevB0 = nullptr;
    void* mDevW1 = nullptr;
    void* mDevB1 = nullptr;
    void* mDevW2 = nullptr;
    void* mDevB2 = nullptr;

    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
    std::vector<nvinfer1::PluginField> mDataToSerialize;

    void uploadWeightsIfNeeded();
};

class CostStem3DPluginCreator final : public nvinfer1::IPluginCreatorV3One {
public:
    CostStem3DPluginCreator();
    nvinfer1::AsciiChar const* getPluginName() const noexcept override;
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override;
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV3* createPlugin(
        nvinfer1::AsciiChar const* name,
        nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase phase) noexcept override;
    void setPluginNamespace(const char* ns) noexcept { mNamespace = ns ? ns : ""; }

private:
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFC{};
    std::vector<nvinfer1::PluginField> mFields;
};

}  // namespace liteany
