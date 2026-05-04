#include "ms_deform_attn_plugin.hpp"

#include <cassert>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

namespace liteany {

static constexpr char PLUGIN_NAME[] = "MSDeformAttn";
static constexpr char PLUGIN_VERSION[] = "1";

MSDeformAttnPlugin::MSDeformAttnPlugin(int numHeads, int p0, int p1, int p2)
    : mNumHeads(numHeads)
    , mP0(p0)
    , mP1(p1)
    , mP2(p2)
{
}

nvinfer1::IPluginCapability* MSDeformAttnPlugin::getCapabilityInterface(
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

nvinfer1::IPluginV3* MSDeformAttnPlugin::clone() noexcept
{
    auto* p = new MSDeformAttnPlugin(mNumHeads, mP0, mP1, mP2);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

nvinfer1::AsciiChar const* MSDeformAttnPlugin::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

nvinfer1::AsciiChar const* MSDeformAttnPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

nvinfer1::AsciiChar const* MSDeformAttnPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void MSDeformAttnPlugin::setPluginNamespace(const char* ns) noexcept
{
    mNamespace = ns ? ns : "";
}

int32_t MSDeformAttnPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* /*in*/,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* /*out*/,
    int32_t nbOutputs) noexcept
{
    return ((nbInputs == 3 || nbInputs == 5 || nbInputs == 7) && nbOutputs == 1) ? 0 : 1;
}

int32_t MSDeformAttnPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    const nvinfer1::DataType* inputTypes,
    int32_t nbInputs) const noexcept
{
    if ((nbInputs != 3 && nbInputs != 5 && nbInputs != 7) || nbOutputs != 1) return 1;
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
    if ((nbInputs != 3 && nbInputs != 5 && nbInputs != 7) || nbOutputs != 1) return 1;

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
    assert(nbInputs == 3 || nbInputs == 5 || nbInputs == 7);
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
    if ((nbInputs != 3 && nbInputs != 5 && nbInputs != 7) || nbOutputs != 1) return 1;
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

    if (inputDesc[0].dims.nbDims == 4 && inputDesc[1].dims.nbDims == 5) {
        launchMSDeformAttnFlat(
            inputs[0],
            inputs[1],
            inputs[2],
            outputs[0],
            B,
            C,
            Q,
            mNumHeads,
            inputDesc[0].type,
            stream);
    } else if (inputDesc[3].dims.nbDims == 5) {
        launchMSDeformAttnFromLocations(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            outputs[0],
            B,
            C,
            Q,
            v0.d[2],
            v0.d[3],
            v1.d[2],
            v1.d[3],
            v2.d[2],
            v2.d[3],
            mNumHeads,
            inputDesc[0].type,
            stream);
    } else {
        launchMSDeformAttn(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[6],
            outputs[0],
            BH,
            C,
            Q,
            v0.d[2],
            v0.d[3],
            v1.d[2],
            v1.d[3],
            v2.d[2],
            v2.d[3],
            mNumHeads,
            inputDesc[0].type,
            inputDesc[3].type,
            inputDesc[6].type,
            stream);
    }
    auto const err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "MSDeformAttn launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

nvinfer1::IPluginV3* MSDeformAttnPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* /*context*/) noexcept
{
    return clone();
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
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

MSDeformAttnPluginCreator::MSDeformAttnPluginCreator()
{
    mFields.emplace_back(
        nvinfer1::PluginField{"num_heads", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    mFields.emplace_back(
        nvinfer1::PluginField{"p0", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    mFields.emplace_back(
        nvinfer1::PluginField{"p1", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    mFields.emplace_back(
        nvinfer1::PluginField{"p2", nullptr, nvinfer1::PluginFieldType::kINT32, 1});
    mFC.nbFields = static_cast<int32_t>(mFields.size());
    mFC.fields = mFields.data();
}

nvinfer1::AsciiChar const* MSDeformAttnPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

nvinfer1::AsciiChar const* MSDeformAttnPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

nvinfer1::AsciiChar const* MSDeformAttnPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

nvinfer1::PluginFieldCollection const* MSDeformAttnPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV3* MSDeformAttnPluginCreator::createPlugin(
    nvinfer1::AsciiChar const* /*name*/,
    nvinfer1::PluginFieldCollection const* fc,
    nvinfer1::TensorRTPhase /*phase*/) noexcept
{
    int numHeads = 8;
    int p0 = 3;
    int p1 = 6;
    int p2 = 3;

    if (fc != nullptr) {
        for (int32_t i = 0; i < fc->nbFields; ++i) {
            const auto& f = fc->fields[i];
            if (f.data == nullptr) continue;
            if (std::strcmp(f.name, "num_heads") == 0) {
                numHeads = *static_cast<const int*>(f.data);
            } else if (std::strcmp(f.name, "p0") == 0) {
                p0 = *static_cast<const int*>(f.data);
            } else if (std::strcmp(f.name, "p1") == 0) {
                p1 = *static_cast<const int*>(f.data);
            } else if (std::strcmp(f.name, "p2") == 0) {
                p2 = *static_cast<const int*>(f.data);
            }
        }
    }

    auto* p = new MSDeformAttnPlugin(numHeads, p0, p1, p2);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

} // namespace liteany
