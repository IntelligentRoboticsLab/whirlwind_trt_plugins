#pragma once

#include "common/plugin_base.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

namespace whirlwind {

// Fused NV12 -> normalised RGB plugin (IPluginV3).
//
// Input  : int8 [B, H_NV12, W_IN] — bytes are NV12 *uint8* values, but
//          TRT's IPluginV3 I/O layer does not expose UINT8 as a datatype, so
//          the network declares them as INT8 and we reinterpret to uint8 in
//          the kernel. (signed-int8 == uint8 at the byte level.)
//          H_NV12 = H_IN * 3 / 2.
// Output : fp16 or fp32 [B, 3, H_OUT, W_OUT] in [-1, 1].
//
// Normalisation: full-range BT.601 YCbCr -> RGB, fused with `2*(rgb/255) - 1`.
class Nv12DecodePlugin final : public PluginV3Base<Nv12DecodePlugin> {
public:
    static constexpr char const* kPluginName = "Nv12Decode";
    static constexpr char const* kPluginVersion = "1";

    Nv12DecodePlugin(int outH, int outW);

    Nv12DecodePlugin* cloneImpl() const;

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
    int mOutH;
    int mOutW;
};

class Nv12DecodePluginCreator final : public PluginCreatorBase<Nv12DecodePluginCreator> {
public:
    static constexpr char const* kPluginName = Nv12DecodePlugin::kPluginName;
    static constexpr char const* kPluginVersion = Nv12DecodePlugin::kPluginVersion;

    Nv12DecodePluginCreator();

    Nv12DecodePlugin* createPluginImpl(nvinfer1::PluginFieldCollection const* fc) noexcept;
};

} // namespace whirlwind
