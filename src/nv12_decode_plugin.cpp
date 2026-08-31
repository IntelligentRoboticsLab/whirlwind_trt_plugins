#include "nv12_decode_plugin.hpp"

#include <cassert>
#include <cstring>

namespace whirlwind {

static constexpr char PLUGIN_NAME[] = "Nv12Decode";
static constexpr char PLUGIN_VERSION[] = "1";

// ---------------------------------------------------------------------------
// Nv12DecodePlugin
// ---------------------------------------------------------------------------

Nv12DecodePlugin::Nv12DecodePlugin(int outH, int outW)
    : mOutH(outH), mOutW(outW)
{
}

nvinfer1::IPluginCapability* Nv12DecodePlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept
{
    switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

nvinfer1::IPluginV3* Nv12DecodePlugin::clone() noexcept
{
    auto* p = new Nv12DecodePlugin(mOutH, mOutW);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

nvinfer1::AsciiChar const* Nv12DecodePlugin::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

nvinfer1::AsciiChar const* Nv12DecodePlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

nvinfer1::AsciiChar const* Nv12DecodePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void Nv12DecodePlugin::setPluginNamespace(const char* ns) noexcept
{
    mNamespace = ns ? ns : "";
}

int32_t Nv12DecodePlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* /*in*/,
    int32_t /*nbInputs*/,
    nvinfer1::DynamicPluginTensorDesc const* /*out*/,
    int32_t /*nbOutputs*/) noexcept
{
    return 0;
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

    launchNv12Decode(
        inputs[0], outputs[0],
        H_IN, W_IN, mOutH, mOutW,
        outputDesc[0].type, stream);
    return 0;
}

nvinfer1::IPluginV3* Nv12DecodePlugin::attachToContext(
    nvinfer1::IPluginResourceContext* /*context*/) noexcept
{
    return clone();
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
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ---------------------------------------------------------------------------
// Nv12DecodePluginCreator
// ---------------------------------------------------------------------------

Nv12DecodePluginCreator::Nv12DecodePluginCreator()
{
    mFields.emplace_back(
        nvinfer1::PluginField{
            "out_h", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    mFields.emplace_back(
        nvinfer1::PluginField{
            "out_w", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    mFC.nbFields = static_cast<int32_t>(mFields.size());
    mFC.fields = mFields.data();
}

nvinfer1::AsciiChar const* Nv12DecodePluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

nvinfer1::AsciiChar const* Nv12DecodePluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

nvinfer1::AsciiChar const* Nv12DecodePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

nvinfer1::PluginFieldCollection const* Nv12DecodePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV3* Nv12DecodePluginCreator::createPlugin(
    nvinfer1::AsciiChar const* /*name*/,
    nvinfer1::PluginFieldCollection const* fc,
    nvinfer1::TensorRTPhase /*phase*/) noexcept
{
    int outH = 224;
    int outW = 288;
    if (fc != nullptr) {
        for (int32_t i = 0; i < fc->nbFields; ++i) {
            const auto& f = fc->fields[i];
            if (f.data == nullptr) continue;
            if (std::strcmp(f.name, "out_h") == 0) {
                outH = *static_cast<const int*>(f.data);
            } else if (std::strcmp(f.name, "out_w") == 0) {
                outW = *static_cast<const int*>(f.data);
            }
        }
    }
    auto* p = new Nv12DecodePlugin(outH, outW);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

} // namespace whirlwind
