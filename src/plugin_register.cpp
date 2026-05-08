#include "corr_volume_plugin.hpp"
#include "cost_stem_3d_plugin.hpp"
#include "ms_deform_attn_plugin.hpp"
#include "nv12_decode_plugin.hpp"

#include <NvInferPlugin.h>

using liteany::CorrVolumePluginCreator;
using liteany::CostStem3DPluginCreator;
using liteany::MSDeformAttnPluginCreator;
using liteany::Nv12DecodePluginCreator;

// Static registration.
// This is what lets TensorRT discover the plugins when the .so is loaded.
REGISTER_TENSORRT_PLUGIN(CorrVolumePluginCreator);
REGISTER_TENSORRT_PLUGIN(CostStem3DPluginCreator);
REGISTER_TENSORRT_PLUGIN(MSDeformAttnPluginCreator);
REGISTER_TENSORRT_PLUGIN(Nv12DecodePluginCreator);
