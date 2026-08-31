#include "cost_stem_3d_plugin.hpp"

#include <cassert>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace whirlwind {

static constexpr char PLUGIN_NAME[] = "CostStem3D";
static constexpr char PLUGIN_VERSION[] = "1";

// ---------------------------------------------------------------------------
// CostStem3DPlugin
//
// 7 inputs: cost-volume [B, 1, D, H, W] fp16,
//           w0 [4*1*27], b0 [4],
//           w1 [4*4*27], b1 [4],
//           w2 [1*4*27], b2 [1]   (all fp16, all device-resident)
// 1 output: same shape as input 0.
// ---------------------------------------------------------------------------

CostStem3DPlugin::CostStem3DPlugin(
    const std::vector<uint16_t>&, const std::vector<uint16_t>&,
    const std::vector<uint16_t>&, const std::vector<uint16_t>&,
    const std::vector<uint16_t>&, const std::vector<uint16_t>&)
{
    // Constructor signature kept for the creator's contract; weights
    // arrive at runtime as plugin inputs, not creation-time attributes.
}

CostStem3DPlugin::~CostStem3DPlugin() = default;

void CostStem3DPlugin::uploadWeightsIfNeeded() {
    // No-op: weights flow as inputs at enqueue time.
}

nvinfer1::IPluginCapability* CostStem3DPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept
{
    switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:    return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:   return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME: return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

nvinfer1::IPluginV3* CostStem3DPlugin::clone() noexcept
{
    auto* p = new CostStem3DPlugin({}, {}, {}, {}, {}, {});
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

nvinfer1::AsciiChar const* CostStem3DPlugin::getPluginName() const noexcept { return PLUGIN_NAME; }
nvinfer1::AsciiChar const* CostStem3DPlugin::getPluginVersion() const noexcept { return PLUGIN_VERSION; }
nvinfer1::AsciiChar const* CostStem3DPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void CostStem3DPlugin::setPluginNamespace(const char* ns) noexcept { mNamespace = ns ? ns : ""; }

int32_t CostStem3DPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const*, int32_t,
    nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept
{ return 0; }

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
    int B = dims.d[0];
    int D = dims.d[2];
    int H = dims.d[3];
    int W = dims.d[4];

    launchCostStem3D(
        inputs[0], outputs[0],
        inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6],
        B, D, H, W, stream);
    return 0;
}

nvinfer1::IPluginV3* CostStem3DPlugin::attachToContext(
    nvinfer1::IPluginResourceContext*) noexcept
{ return clone(); }

nvinfer1::PluginFieldCollection const* CostStem3DPlugin::getFieldsToSerialize() noexcept
{
    // No persistent state to serialize — weights are graph initializers,
    // not plugin attributes. Return an empty collection.
    mDataToSerialize.clear();
    mFCToSerialize.nbFields = 0;
    mFCToSerialize.fields = nullptr;
    return &mFCToSerialize;
}

// ---------------------------------------------------------------------------
// CostStem3DPluginCreator
// ---------------------------------------------------------------------------

CostStem3DPluginCreator::CostStem3DPluginCreator()
{
    // No fields — weights flow as inputs.
    mFC.nbFields = 0;
    mFC.fields = nullptr;
}

nvinfer1::AsciiChar const* CostStem3DPluginCreator::getPluginName() const noexcept { return PLUGIN_NAME; }
nvinfer1::AsciiChar const* CostStem3DPluginCreator::getPluginVersion() const noexcept { return PLUGIN_VERSION; }
nvinfer1::AsciiChar const* CostStem3DPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
nvinfer1::PluginFieldCollection const* CostStem3DPluginCreator::getFieldNames() noexcept { return &mFC; }

nvinfer1::IPluginV3* CostStem3DPluginCreator::createPlugin(
    nvinfer1::AsciiChar const*,
    nvinfer1::PluginFieldCollection const*,
    nvinfer1::TensorRTPhase) noexcept
{
    auto* p = new CostStem3DPlugin({}, {}, {}, {}, {}, {});
    p->setPluginNamespace(mNamespace.c_str());
    return p;
}

}  // namespace whirlwind
