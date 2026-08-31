#pragma once

#include "common/plugin_base.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

namespace whirlwind {

// Fused cost_stem_3d plugin. Replaces three (Conv3D 3x3x3 + BN + Clip[0,6])
// stages running over a [1, 1, 48, 32, 40] cost-volume tensor with a single
// fused CUDA kernel that holds intermediates in shared memory.
//
// The TRT-generated graph runs the three layers as 18 sm50_xmma_conv3d_c1_k1
// kernel calls (sm50 fallback because the shape is too small/odd for the
// sm80 tile configurations), totalling ~67 µs at s025. A custom fused
// kernel cuts this to launch-overhead-floor (~2-3 µs).
//
// 7 inputs: cost-volume [B, 1, D, H, W] fp16,
//           w0 [4*1*27], b0 [4],
//           w1 [4*4*27], b1 [4],
//           w2 [1*4*27], b2 [1]   (all fp16, all device-resident)
// 1 output: same shape as input 0.
//
// The BN-folded weights arrive as graph initialisers wired to input slots
// 1..6, so the plugin itself is stateless — nothing to serialise, nothing
// to upload.
class CostStem3DPlugin final : public PluginV3Base<CostStem3DPlugin> {
public:
    static constexpr char const* kPluginName = "CostStem3D";
    static constexpr char const* kPluginVersion = "1";

    CostStem3DPlugin() = default;

    CostStem3DPlugin* cloneImpl() const;

    // ---- IPluginV3OneBuild ----
    int32_t getOutputDataTypes(
        nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(
        nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override;

    // ---- IPluginV3OneRuntime ----
    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;
};

class CostStem3DPluginCreator final : public PluginCreatorBase<CostStem3DPluginCreator> {
public:
    static constexpr char const* kPluginName = CostStem3DPlugin::kPluginName;
    static constexpr char const* kPluginVersion = CostStem3DPlugin::kPluginVersion;

    CostStem3DPluginCreator();

    CostStem3DPlugin* createPluginImpl(nvinfer1::PluginFieldCollection const* fc) noexcept;
};

} // namespace whirlwind
