#pragma once

#include "common/plugin_base.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

namespace whirlwind {

class CorrVolumePlugin final : public PluginV3Base<CorrVolumePlugin> {
public:
    static constexpr char const* kPluginName = "CorrVolume";
    static constexpr char const* kPluginVersion = "1";

    explicit CorrVolumePlugin(int maxDisp);

    CorrVolumePlugin* cloneImpl() const;

    // ---- IPluginV3OneBuild ----
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
    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    int mMaxDisp;
};

class CorrVolumePluginCreator final : public PluginCreatorBase<CorrVolumePluginCreator> {
public:
    static constexpr char const* kPluginName = CorrVolumePlugin::kPluginName;
    static constexpr char const* kPluginVersion = CorrVolumePlugin::kPluginVersion;

    CorrVolumePluginCreator();

    CorrVolumePlugin* createPluginImpl(nvinfer1::PluginFieldCollection const* fc) noexcept;
};

} // namespace whirlwind
