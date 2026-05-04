#pragma once

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <string>
#include <vector>

namespace liteany {

void launchMSDeformAttn(
    const void* value0,
    const void* value1,
    const void* value2,
    const void* grid0,
    const void* grid1,
    const void* grid2,
    const void* weights,
    void* output,
    int BH,
    int C,
    int Q,
    int H0,
    int W0,
    int H1,
    int W1,
    int H2,
    int W2,
    int numHeads,
    nvinfer1::DataType valueDtype,
    nvinfer1::DataType gridDtype,
    nvinfer1::DataType weightDtype,
    cudaStream_t stream);

void launchMSDeformAttnFromLocations(
    const void* value0,
    const void* value1,
    const void* value2,
    const void* locations,
    const void* weights,
    void* output,
    int B,
    int C,
    int Q,
    int H0,
    int W0,
    int H1,
    int W1,
    int H2,
    int W2,
    int numHeads,
    nvinfer1::DataType dtype,
    cudaStream_t stream);

void launchMSDeformAttnFlat(
    const void* value,
    const void* locations,
    const void* weights,
    void* output,
    int B,
    int C,
    int Q,
    int numHeads,
    nvinfer1::DataType dtype,
    cudaStream_t stream);

class MSDeformAttnPlugin final : public nvinfer1::IPluginV3,
                                 public nvinfer1::IPluginV3OneCore,
                                 public nvinfer1::IPluginV3OneBuild,
                                 public nvinfer1::IPluginV3OneRuntime {
public:
    MSDeformAttnPlugin(int numHeads, int p0, int p1, int p2);

    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    nvinfer1::AsciiChar const* getPluginName() const noexcept override;
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override;
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(const char* ns) noexcept;

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

    int32_t onShapeChange(
        nvinfer1::PluginTensorDesc const* in,
        int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

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
    int mNumHeads;
    int mP0;
    int mP1;
    int mP2;
    std::string mNamespace;

    nvinfer1::PluginFieldCollection mFCToSerialize{};
    std::vector<nvinfer1::PluginField> mDataToSerialize;
};

class MSDeformAttnPluginCreator final : public nvinfer1::IPluginCreatorV3One {
public:
    MSDeformAttnPluginCreator();

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
