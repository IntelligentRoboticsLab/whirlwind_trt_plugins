#include "plugins/corr_volume/corr_volume.hpp"
#include "plugins/cost_stem_3d/cost_stem_3d.hpp"
#include "plugins/ms_deform_attn/ms_deform_attn.hpp"
#include "plugins/nv12_decode/nv12_decode.hpp"

#include <NvInferPlugin.h>

using whirlwind::CorrVolumePluginCreator;
using whirlwind::CostStem3DPluginCreator;
using whirlwind::MSDeformAttnPluginCreator;
using whirlwind::Nv12DecodePluginCreator;

// Static registration.
// This is what lets TensorRT discover the plugins when the .so is loaded.
REGISTER_TENSORRT_PLUGIN(CorrVolumePluginCreator);
REGISTER_TENSORRT_PLUGIN(CostStem3DPluginCreator);
REGISTER_TENSORRT_PLUGIN(MSDeformAttnPluginCreator);
REGISTER_TENSORRT_PLUGIN(Nv12DecodePluginCreator);
