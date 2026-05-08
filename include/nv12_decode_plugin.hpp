#pragma once

#include <NvInferRuntimePlugin.h>
#include <NvInferRuntime.h>

#include <string>
#include <vector>

namespace liteany {

// Fused NV12 -> normalised RGB plugin (IPluginV3).
//
// Input  : int8 [B, H_NV12, W_IN] — bytes are NV12 *uint8* values, but
//          TRT's IPluginV3 I/O layer does not expose UINT8 as a datatype, so
//          the network declares them as INT8 and we reinterpret to uint8 in
//          the kernel. (signed-int8 == uint8 at the byte level.)
//          H_NV12 = H_IN * 3 / 2.
// Output : fp16 or fp32 [B, 3, H_OUT, W_OUT] in [-1, 1].
//
// Normalisation: full-range BT.601 YCbCr -> RGB, fused with `2*(rgb/255) - 1`.

void launchNv12Decode(
    const void* nv12,
    void* output,
    int H_IN,
    int W_IN,
    int H_OUT,
    int W_OUT,
    nvinfer1::DataType outDtype,
    cudaStream_t stream);

class Nv12DecodePlugin final : public nvinfer1::IPluginV3,
                               public nvinfer1::IPluginV3OneCore,
                               public nvinfer1::IPluginV3OneBuild,
                               public nvinfer1::IPluginV3OneRuntime {
public:
    Nv12DecodePlugin(int outH, int outW);

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
        nvinfer1::DynamicPluginTensorDesc const* in,
        int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    int32_t getOutputDataTypes(
        nvinfer1::DataType* outputTypes,
        int32_t nbOutputs,
        const nvinfer1::DataType* inputTypes,
        int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(
        nvinfer1::DimsExprs const* inputs,
        int32_t nbInputs,
        nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs,
        int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int32_t pos,
        nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs,
        int32_t nbOutputs) noexcept override;

    // ---- IPluginV3OneRuntime ----
    int32_t onShapeChange(
        nvinfer1::PluginTensorDesc const* in,
        int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override { return 0; }

    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override;

    nvinfer1::IPluginV3* attachToContext(
        nvinfer1::IPluginResourceContext* context) noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    int mOutH;
    int mOutW;
    std::string mNamespace;

    nvinfer1::PluginFieldCollection mFCToSerialize{};
    std::vector<nvinfer1::PluginField> mDataToSerialize;
};

class Nv12DecodePluginCreator final : public nvinfer1::IPluginCreatorV3One {
public:
    Nv12DecodePluginCreator();

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

} // namespace liteany
