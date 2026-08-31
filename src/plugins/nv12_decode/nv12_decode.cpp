#include "plugins/nv12_decode/nv12_decode.hpp"

#include "common/dtype_dispatch.hpp"
#include "common/plugin_fields.hpp"
#include "plugins/nv12_decode/nv12_decode_kernel.hpp"

#include <cassert>
#include <cstdint>

namespace whirlwind {

// ---------------------------------------------------------------------------
// Nv12DecodePlugin
// ---------------------------------------------------------------------------

Nv12DecodePlugin::Nv12DecodePlugin(int outH, int outW)
    : mOutH(outH), mOutW(outW)
{
}

Nv12DecodePlugin* Nv12DecodePlugin::cloneImpl() const
{
    return new Nv12DecodePlugin(mOutH, mOutW);
}

int32_t Nv12DecodePlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    const nvinfer1::DataType* /*inputTypes*/,
    int32_t /*nbInputs*/) const noexcept
{
    if (nbOutputs < 1) return 1;
    // Declare fp32 so the ONNX parser keeps the downstream graph type-
    // consistent (InstanceNorm scale/bias are fp32 initialisers). TRT will
    // still pick fp16 at build time via supportsFormatCombination — the
    // declared type is only used for graph-validity checks.
    outputTypes[0] = nvinfer1::DataType::kFLOAT;
    return 0;
}

int32_t Nv12DecodePlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::DimsExprs const* /*shapeInputs*/,
    int32_t /*nbShapeInputs*/,
    nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (nbInputs != 1 || nbOutputs != 1) return 1;

    // input  : [B, H_NV12, W_IN]
    // output : [B, 3, H_OUT, W_OUT]
    outputs[0].nbDims = 4;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = exprBuilder.constant(3);
    outputs[0].d[2] = exprBuilder.constant(mOutH);
    outputs[0].d[3] = exprBuilder.constant(mOutW);
    return 0;
}

bool Nv12DecodePlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept
{
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
    assert(pos >= 0 && pos < nbInputs + nbOutputs);

    const auto& desc = inOut[pos].desc;
    if (desc.format != nvinfer1::TensorFormat::kLINEAR) return false;

    if (pos == 0) {
        // INT8-only: the network input bytes are raw uint8 NV12 values; we
        // accept INT8 here and reinterpret to uint8_t in the kernel.
        // Refusing fp32 prevents TRT from silently inserting a
        // copyVectorizedKernel<signed char, float> cast that would
        // (a) cost ~1.8 µs/eye and (b) sign-extend values >127.
        // (kUINT8 is not a supported PluginV3 I/O datatype.)
        return desc.type == nvinfer1::DataType::kINT8;
    }
    return desc.type == nvinfer1::DataType::kHALF
        || desc.type == nvinfer1::DataType::kFLOAT;
}

int32_t Nv12DecodePlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* /*workspace*/,
    cudaStream_t stream) noexcept
{
    const auto& dims = inputDesc[0].dims;
    if (dims.nbDims != 3) return 1;

    const int H_NV12 = dims.d[1];
    const int W_IN   = dims.d[2];
    const int H_IN   = (H_NV12 * 2) / 3;

    // Declared INT8, actually uint8 — see supportsFormatCombination.
    const auto* nv12 = static_cast<const uint8_t*>(inputs[0]);

    cudaError_t err = cudaSuccess;
    const bool dispatched = dispatchFloatType(outputDesc[0].type, [&](auto tag) {
        using TOut = typename decltype(tag)::type;
        err = launchNv12Decode<TOut>(
            nv12, static_cast<TOut*>(outputs[0]),
            H_IN, W_IN, mOutH, mOutW, stream);
    });
    if (!dispatched) return 1;
    return err == cudaSuccess ? 0 : 1;
}

nvinfer1::PluginFieldCollection const* Nv12DecodePlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(
        nvinfer1::PluginField{
            "out_h", &mOutH, nvinfer1::PluginFieldType::kINT32, 1});
    mDataToSerialize.emplace_back(
        nvinfer1::PluginField{
            "out_w", &mOutW, nvinfer1::PluginFieldType::kINT32, 1});
    return publishSerializedFields();
}

// ---------------------------------------------------------------------------
// Nv12DecodePluginCreator
// ---------------------------------------------------------------------------

Nv12DecodePluginCreator::Nv12DecodePluginCreator()
{
    mFields.emplace_back(intField("out_h"));
    mFields.emplace_back(intField("out_w"));
    publishFields();
}

Nv12DecodePlugin* Nv12DecodePluginCreator::createPluginImpl(
    nvinfer1::PluginFieldCollection const* fc) noexcept
{
    return new Nv12DecodePlugin(
        getIntField(fc, "out_h", 224),
        getIntField(fc, "out_w", 288));
}

} // namespace whirlwind
