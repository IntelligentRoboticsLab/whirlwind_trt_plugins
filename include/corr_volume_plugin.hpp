#pragma once

#include <NvInferRuntimePlugin.h>
#include <NvInferRuntime.h>

#include <string>
#include <vector>

namespace liteany {

void launchCorrVolume(
    const void* left,
    const void* right,
    void* output,
    int B,
    int C,
    int H,
    int W,
    int D,
    nvinfer1::DataType dtype,
    cudaStream_t stream);

class CorrVolumePlugin final : public nvinfer1::IPluginV3,
                               public nvinfer1::IPluginV3OneCore,
                               public nvinfer1::IPluginV3OneBuild,
                               public nvinfer1::IPluginV3OneRuntime {
public:
    explicit CorrVolumePlugin(int maxDisp);

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
    int mMaxDisp;
    std::string mNamespace;

    // Buffers backing getFieldsToSerialize().
    nvinfer1::PluginFieldCollection mFCToSerialize{};
    std::vector<nvinfer1::PluginField> mDataToSerialize;
};

class CorrVolumePluginCreator final : public nvinfer1::IPluginCreatorV3One {
public:
    CorrVolumePluginCreator();

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
