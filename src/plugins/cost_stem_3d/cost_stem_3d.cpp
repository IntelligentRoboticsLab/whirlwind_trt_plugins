#include "plugins/cost_stem_3d/cost_stem_3d.hpp"

#include "plugins/cost_stem_3d/cost_stem_3d_kernel.hpp"

#include <cassert>
#include <cuda_fp16.h>

namespace whirlwind {

// ---------------------------------------------------------------------------
// CostStem3DPlugin
// ---------------------------------------------------------------------------

CostStem3DPlugin* CostStem3DPlugin::cloneImpl() const
{
    return new CostStem3DPlugin();
}

int32_t CostStem3DPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes, int32_t nbOutputs,
    const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept
{
    if (nbInputs < 1 || nbOutputs < 1) return 1;
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t CostStem3DPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::DimsExprs const*, int32_t,
    nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
    nvinfer1::IExprBuilder&) noexcept
{
    if (nbInputs != 7 || nbOutputs != 1) return 1;
    outputs[0] = inputs[0];
    return 0;
}

bool CostStem3DPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs, int32_t nbOutputs) noexcept
{
    assert(nbInputs == 7 && nbOutputs == 1);
    assert(pos >= 0 && pos < nbInputs + nbOutputs);
    const auto& d = inOut[pos].desc;
    if (d.format != nvinfer1::TensorFormat::kLINEAR) return false;
    return d.type == nvinfer1::DataType::kHALF;
}

int32_t CostStem3DPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const*,
    const void* const* inputs, void* const* outputs,
    void*, cudaStream_t stream) noexcept
{
    const auto& dims = inputDesc[0].dims;
    if (dims.nbDims != 5) return 1;
    const int B = dims.d[0];
    const int D = dims.d[2];
    const int H = dims.d[3];
    const int W = dims.d[4];

    // fp16-only, enforced by supportsFormatCombination.
    const auto half = [](const void* p) { return static_cast<const __half*>(p); };

    const cudaError_t err = launchCostStem3D(
        half(inputs[0]), static_cast<__half*>(outputs[0]),
        half(inputs[1]), half(inputs[2]),
        half(inputs[3]), half(inputs[4]),
        half(inputs[5]), half(inputs[6]),
        B, D, H, W, stream);
    return err == cudaSuccess ? 0 : 1;
}

nvinfer1::PluginFieldCollection const* CostStem3DPlugin::getFieldsToSerialize() noexcept
{
    // No persistent state to serialize — weights are graph initializers,
    // not plugin attributes. Return an empty collection.
    mDataToSerialize.clear();
    return publishSerializedFields();
}

// ---------------------------------------------------------------------------
// CostStem3DPluginCreator
// ---------------------------------------------------------------------------

CostStem3DPluginCreator::CostStem3DPluginCreator()
{
    // No fields — weights flow as inputs.
    publishFields();
}

CostStem3DPlugin* CostStem3DPluginCreator::createPluginImpl(
    nvinfer1::PluginFieldCollection const* /*fc*/) noexcept
{
    return new CostStem3DPlugin();
}

} // namespace whirlwind
