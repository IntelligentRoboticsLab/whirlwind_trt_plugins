#pragma once

#include "common/plugin_base.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

namespace whirlwind {

// Multi-scale deformable attention. Accepts three input layouts — 3, 5 or 7
// tensors — see ms_deform_attn_kernel.hpp for what each one carries.
class MSDeformAttnPlugin final : public PluginV3Base<MSDeformAttnPlugin> {
public:
    static constexpr char const* kPluginName = "MSDeformAttn";
    static constexpr char const* kPluginVersion = "1";

    MSDeformAttnPlugin(int numHeads, int p0, int p1, int p2);

    MSDeformAttnPlugin* cloneImpl() const;

    // ---- IPluginV3OneBuild ----
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
        int32_t nbOutputs) noexcept override;

    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    int mNumHeads;
    int mP0;
    int mP1;
    int mP2;
};

class MSDeformAttnPluginCreator final : public PluginCreatorBase<MSDeformAttnPluginCreator> {
public:
    static constexpr char const* kPluginName = MSDeformAttnPlugin::kPluginName;
    static constexpr char const* kPluginVersion = MSDeformAttnPlugin::kPluginVersion;

    MSDeformAttnPluginCreator();

    MSDeformAttnPlugin* createPluginImpl(nvinfer1::PluginFieldCollection const* fc) noexcept;
};

} // namespace whirlwind
