#include "corr_volume_plugin.hpp"

#include <cassert>
#include <cstring>

namespace liteany {

static constexpr char PLUGIN_NAME[] = "CorrVolume";
static constexpr char PLUGIN_VERSION[] = "1";

// ---------------------------------------------------------------------------
// CorrVolumePlugin
// ---------------------------------------------------------------------------

CorrVolumePlugin::CorrVolumePlugin(int maxDisp)
    : mMaxDisp(maxDisp)
{
}

nvinfer1::IPluginCapability* CorrVolumePlugin::getCapabilityInterface(
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

nvinfer1::IPluginV3* CorrVolumePlugin::clone() noexcept
{
    auto* p = new CorrVolumePlugin(mMaxDisp);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

nvinfer1::AsciiChar const* CorrVolumePlugin::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

nvinfer1::AsciiChar const* CorrVolumePlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

nvinfer1::AsciiChar const* CorrVolumePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void CorrVolumePlugin::setPluginNamespace(const char* ns) noexcept
{
    mNamespace = ns ? ns : "";
}

int32_t CorrVolumePlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* /*in*/,
    int32_t /*nbInputs*/,
    nvinfer1::DynamicPluginTensorDesc const* /*out*/,
    int32_t /*nbOutputs*/) noexcept
{
    return 0;
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
        return desc.type == nvinfer1::DataType::kHALF
            || desc.type == nvinfer1::DataType::kFLOAT;
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

    int B = dims.d[0];
    int C = dims.d[1];
    int H = dims.d[2];
    int W = dims.d[3];

    launchCorrVolume(
        inputs[0], inputs[1], outputs[0],
        B, C, H, W, mMaxDisp,
        inputDesc[0].type, stream);
    return 0;
}

nvinfer1::IPluginV3* CorrVolumePlugin::attachToContext(
    nvinfer1::IPluginResourceContext* /*context*/) noexcept
{
    return clone();
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
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

// ---------------------------------------------------------------------------
// CorrVolumePluginCreator
// ---------------------------------------------------------------------------

CorrVolumePluginCreator::CorrVolumePluginCreator()
{
    mFields.emplace_back(
        nvinfer1::PluginField{
            "max_disp",
            nullptr,
            nvinfer1::PluginFieldType::kINT32,
            1});
    mFC.nbFields = static_cast<int32_t>(mFields.size());
    mFC.fields = mFields.data();
}

nvinfer1::AsciiChar const* CorrVolumePluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

nvinfer1::AsciiChar const* CorrVolumePluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

nvinfer1::AsciiChar const* CorrVolumePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

nvinfer1::PluginFieldCollection const* CorrVolumePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV3* CorrVolumePluginCreator::createPlugin(
    nvinfer1::AsciiChar const* /*name*/,
    nvinfer1::PluginFieldCollection const* fc,
    nvinfer1::TensorRTPhase /*phase*/) noexcept
{
    int maxDisp = 48;
    if (fc != nullptr) {
        for (int32_t i = 0; i < fc->nbFields; ++i) {
            const auto& f = fc->fields[i];
            if (std::strcmp(f.name, "max_disp") == 0 && f.data != nullptr) {
                maxDisp = *static_cast<const int*>(f.data);
            }
        }
    }
    auto* p = new CorrVolumePlugin(maxDisp);
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

} // namespace liteany
