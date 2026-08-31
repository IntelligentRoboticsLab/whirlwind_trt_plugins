#include "plugins/ms_deform_attn/ms_deform_attn.hpp"

#include "common/dtype_dispatch.hpp"
#include "common/plugin_fields.hpp"
#include "plugins/ms_deform_attn/ms_deform_attn_kernel.hpp"

#include <cassert>

namespace whirlwind {

namespace {

bool validInputCount(int32_t nbInputs) noexcept
{
    return nbInputs == 3 || nbInputs == 5 || nbInputs == 7;
}

} // namespace

MSDeformAttnPlugin::MSDeformAttnPlugin(int numHeads, int p0, int p1, int p2)
    : mNumHeads(numHeads)
    , mP0(p0)
    , mP1(p1)
    , mP2(p2)
{
}

MSDeformAttnPlugin* MSDeformAttnPlugin::cloneImpl() const
{
    return new MSDeformAttnPlugin(mNumHeads, mP0, mP1, mP2);
}

int32_t MSDeformAttnPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* /*in*/,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* /*out*/,
    int32_t nbOutputs) noexcept
{
    return (validInputCount(nbInputs) && nbOutputs == 1) ? 0 : 1;
}

int32_t MSDeformAttnPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    const nvinfer1::DataType* inputTypes,
    int32_t nbInputs) const noexcept
{
    if (!validInputCount(nbInputs) || nbOutputs != 1) return 1;
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t MSDeformAttnPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::DimsExprs const* /*shapeInputs*/,
    int32_t /*nbShapeInputs*/,
    nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (!validInputCount(nbInputs) || nbOutputs != 1) return 1;

    // 3-input flat mode:
    // value    : [B, 8400, num_heads, C]
    // locations: [B, Q, num_heads, P0 + P1 + P2, 2] in [0, 1]
    // weights  : [B, Q, num_heads, P0 + P1 + P2]
    //
    // 5-input mode:
    // locations: [B, Q, num_heads, P0 + P1 + P2, 2] in [0, 1]
    // weights  : [B, Q, num_heads, P0 + P1 + P2]
    //
    // 7-input compatibility mode:
    // grid0/1/2: [B * num_heads, Q, P, 2] in GridSample [-1, 1] coordinates
    // weights  : [B, Q, num_heads, P0 + P1 + P2]
    //
    // value0 : [B * num_heads, C, H0, W0]
    // output : [B, Q, num_heads * C]
    auto const* channels = nbInputs == 3
        ? exprBuilder.operation(
            nvinfer1::DimensionOperation::kPROD,
            *inputs[0].d[3],
            *exprBuilder.constant(mNumHeads))
        : exprBuilder.operation(
            nvinfer1::DimensionOperation::kPROD,
            *inputs[0].d[1],
            *exprBuilder.constant(mNumHeads));
    if (channels == nullptr) return 1;

    outputs[0].nbDims = 3;
    if (nbInputs == 3) {
        outputs[0].d[0] = inputs[1].d[0];
        outputs[0].d[1] = inputs[1].d[1];
    } else if (nbInputs == 5) {
        outputs[0].d[0] = inputs[3].d[0];
        outputs[0].d[1] = inputs[3].d[1];
    } else {
        outputs[0].d[0] = inputs[6].d[0];
        outputs[0].d[1] = inputs[3].d[1];
    }
    outputs[0].d[2] = channels;
    return 0;
}

bool MSDeformAttnPlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept
{
    assert(validInputCount(nbInputs));
    assert(nbOutputs == 1);
    assert(pos >= 0 && pos < nbInputs + nbOutputs);

    const auto& desc = inOut[pos].desc;
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) return false;

    if (pos == 0) {
        return desc.type == nvinfer1::DataType::kHALF
            || desc.type == nvinfer1::DataType::kFLOAT;
    }

    return desc.type == inOut[0].desc.type;
}

int32_t MSDeformAttnPlugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* /*out*/,
    int32_t nbOutputs) noexcept
{
    if (!validInputCount(nbInputs) || nbOutputs != 1) return 1;
    if (nbInputs == 3) {
        if (in[0].dims.nbDims != 4 || in[1].dims.nbDims != 5 || in[2].dims.nbDims != 4) return 1;
    } else if (nbInputs == 5) {
        if (in[0].dims.nbDims != 4 || in[1].dims.nbDims != 4 || in[2].dims.nbDims != 4) return 1;
        if (in[3].dims.nbDims != 5 || in[4].dims.nbDims != 4) return 1;
    } else {
        if (in[0].dims.nbDims != 4 || in[1].dims.nbDims != 4 || in[2].dims.nbDims != 4) return 1;
        if (in[3].dims.nbDims != 4 || in[4].dims.nbDims != 4 || in[5].dims.nbDims != 4) return 1;
        if (in[6].dims.nbDims != 4) return 1;
    }
    return 0;
}

int32_t MSDeformAttnPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* /*workspace*/,
    cudaStream_t stream) noexcept
{
    const auto& v0 = inputDesc[0].dims;
    const auto& v1 = inputDesc[1].dims;
    const auto& v2 = inputDesc[2].dims;
    const auto& out = outputDesc[0].dims;

    const int B = out.d[0];
    const int Q = out.d[1];
    const int C = out.d[2] / mNumHeads;
    const int BH = B * mNumHeads;

    cudaError_t err = cudaSuccess;
    bool dispatched = false;

    if (v0.nbDims == 4 && v1.nbDims == 5) {
        dispatched = dispatchFloatType(inputDesc[0].type, [&](auto tag) {
            using T = typename decltype(tag)::type;
            err = launchMSDeformAttnFlat<T>(
                static_cast<const T*>(inputs[0]),
                static_cast<const T*>(inputs[1]),
                static_cast<const T*>(inputs[2]),
                static_cast<T*>(outputs[0]),
                B, C, Q, mNumHeads, stream);
        });
    } else if (inputDesc[3].dims.nbDims == 5) {
        dispatched = dispatchFloatType(inputDesc[0].type, [&](auto tag) {
            using T = typename decltype(tag)::type;
            err = launchMSDeformAttnFromLocations<T>(
                static_cast<const T*>(inputs[0]),
                static_cast<const T*>(inputs[1]),
                static_cast<const T*>(inputs[2]),
                static_cast<const T*>(inputs[3]),
                static_cast<const T*>(inputs[4]),
                static_cast<T*>(outputs[0]),
                B, C, Q,
                v0.d[2], v0.d[3], v1.d[2], v1.d[3], v2.d[2], v2.d[3],
                mNumHeads, stream);
        });
    } else {
        // Values, grids and weights are independently typed — TRT may hand
        // us fp32 grids alongside fp16 values.
        dispatchFloatType(inputDesc[0].type, [&](auto valueTag) {
            dispatchFloatType(inputDesc[3].type, [&](auto gridTag) {
                dispatched = dispatchFloatType(inputDesc[6].type, [&](auto weightTag) {
                    using TValue = typename decltype(valueTag)::type;
                    using TGrid = typename decltype(gridTag)::type;
                    using TWeight = typename decltype(weightTag)::type;
                    err = launchMSDeformAttn<TValue, TGrid, TWeight>(
                        static_cast<const TValue*>(inputs[0]),
                        static_cast<const TValue*>(inputs[1]),
                        static_cast<const TValue*>(inputs[2]),
                        static_cast<const TGrid*>(inputs[3]),
                        static_cast<const TGrid*>(inputs[4]),
                        static_cast<const TGrid*>(inputs[5]),
                        static_cast<const TWeight*>(inputs[6]),
                        static_cast<TValue*>(outputs[0]),
                        BH, C, Q,
                        v0.d[2], v0.d[3], v1.d[2], v1.d[3], v2.d[2], v2.d[3],
                        mNumHeads, stream);
                });
            });
        });
    }

    if (!dispatched) return 1;
    return err == cudaSuccess ? 0 : 1;
}

nvinfer1::PluginFieldCollection const* MSDeformAttnPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(
        nvinfer1::PluginField{"num_heads", &mNumHeads, nvinfer1::PluginFieldType::kINT32, 1});
    mDataToSerialize.emplace_back(
        nvinfer1::PluginField{"p0", &mP0, nvinfer1::PluginFieldType::kINT32, 1});
    mDataToSerialize.emplace_back(
        nvinfer1::PluginField{"p1", &mP1, nvinfer1::PluginFieldType::kINT32, 1});
    mDataToSerialize.emplace_back(
        nvinfer1::PluginField{"p2", &mP2, nvinfer1::PluginFieldType::kINT32, 1});
    return publishSerializedFields();
}

// ---------------------------------------------------------------------------
// MSDeformAttnPluginCreator
// ---------------------------------------------------------------------------

MSDeformAttnPluginCreator::MSDeformAttnPluginCreator()
{
    mFields.emplace_back(intField("num_heads"));
    mFields.emplace_back(intField("p0"));
    mFields.emplace_back(intField("p1"));
    mFields.emplace_back(intField("p2"));
    publishFields();
}

MSDeformAttnPlugin* MSDeformAttnPluginCreator::createPluginImpl(
    nvinfer1::PluginFieldCollection const* fc) noexcept
{
    return new MSDeformAttnPlugin(
        getIntField(fc, "num_heads", 8),
        getIntField(fc, "p0", 3),
        getIntField(fc, "p1", 6),
        getIntField(fc, "p2", 3));
}

} // namespace whirlwind
