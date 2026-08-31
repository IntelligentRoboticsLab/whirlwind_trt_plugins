#include "plugins/corr_volume/corr_volume.hpp"

#include "common/dtype_dispatch.hpp"
#include "common/plugin_fields.hpp"
#include "plugins/corr_volume/corr_volume_kernel.hpp"

#include <cassert>

namespace whirlwind {

// ---------------------------------------------------------------------------
// CorrVolumePlugin
//
// 2 inputs: left, right [B, C, H, W]
// 1 output: cost volume  [B, max_disp, H, W]
// ---------------------------------------------------------------------------

CorrVolumePlugin::CorrVolumePlugin(int maxDisp)
    : mMaxDisp(maxDisp)
{
}

CorrVolumePlugin* CorrVolumePlugin::cloneImpl() const
{
    return new CorrVolumePlugin(mMaxDisp);
}

int32_t CorrVolumePlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    const nvinfer1::DataType* inputTypes,
    int32_t nbInputs) const noexcept
{
    if (nbInputs < 1 || nbOutputs < 1) return 1;
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t CorrVolumePlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::DimsExprs const* /*shapeInputs*/,
    int32_t /*nbShapeInputs*/,
    nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (nbInputs != 2 || nbOutputs != 1) return 1;

    // input  : [B, C, H, W]
    // output : [B, D, H, W]
    outputs[0].nbDims = 4;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = exprBuilder.constant(mMaxDisp);
    outputs[0].d[2] = inputs[0].d[2];
    outputs[0].d[3] = inputs[0].d[3];
    return 0;
}

bool CorrVolumePlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept
{
    assert(nbInputs == 2);
    assert(nbOutputs == 1);
    assert(pos >= 0 && pos < nbInputs + nbOutputs);

    const auto& desc = inOut[pos].desc;
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) return false;

    if (pos == 0) {
        // FP16 only: the fast-path kernel is FP16, and advertising FP32
        // here lets TRT insert a Half->Float reformat upstream, bypassing
        // it entirely.
        return desc.type == nvinfer1::DataType::kHALF;
    }

    // Other inputs and the output must match input 0.
    return desc.type == inOut[0].desc.type;
}

int32_t CorrVolumePlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* /*outputDesc*/,
    const void* const* inputs,
    void* const* outputs,
    void* /*workspace*/,
    cudaStream_t stream) noexcept
{
    const auto& dims = inputDesc[0].dims;
    if (dims.nbDims != 4) return 1;

    const int B = dims.d[0];
    const int C = dims.d[1];
    const int H = dims.d[2];
    const int W = dims.d[3];

    cudaError_t err = cudaSuccess;
    const bool dispatched = dispatchFloatType(inputDesc[0].type, [&](auto tag) {
        using T = typename decltype(tag)::type;
        err = launchCorrVolume<T>(
            static_cast<const T*>(inputs[0]),
            static_cast<const T*>(inputs[1]),
            static_cast<T*>(outputs[0]),
            B, C, H, W, mMaxDisp, stream);
    });
    if (!dispatched) return 1;
    return err == cudaSuccess ? 0 : 1;
}

nvinfer1::PluginFieldCollection const* CorrVolumePlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(
        nvinfer1::PluginField{
            "max_disp",
            &mMaxDisp,
            nvinfer1::PluginFieldType::kINT32,
            1});
    return publishSerializedFields();
}

// ---------------------------------------------------------------------------
// CorrVolumePluginCreator
// ---------------------------------------------------------------------------

CorrVolumePluginCreator::CorrVolumePluginCreator()
{
    mFields.emplace_back(intField("max_disp"));
    publishFields();
}

CorrVolumePlugin* CorrVolumePluginCreator::createPluginImpl(
    nvinfer1::PluginFieldCollection const* fc) noexcept
{
    return new CorrVolumePlugin(getIntField(fc, "max_disp", 48));
}

} // namespace whirlwind
