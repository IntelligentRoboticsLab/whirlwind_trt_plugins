#include "corr_volume_plugin.hpp"
#include "nv12_decode_plugin.hpp"

#include <NvInferPlugin.h>

using liteany::CorrVolumePluginCreator;
using liteany::Nv12DecodePluginCreator;

// Static registration.
// This is what lets TensorRT discover the plugins when the .so is loaded.
REGISTER_TENSORRT_PLUGIN(CorrVolumePluginCreator);
REGISTER_TENSORRT_PLUGIN(Nv12DecodePluginCreator);