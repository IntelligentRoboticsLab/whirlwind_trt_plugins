#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace whirlwind {

// Fused NV12 -> normalised planar RGB.
//
// nv12   : uint8 [H_NV12, W_IN] with H_NV12 = H_IN * 3 / 2. The first H_IN
//          rows are the Y plane; the remaining H_IN/2 rows are interleaved
//          U/V chroma at half resolution.
// output : [3, H_OUT, W_OUT] in [-1, 1], full-range BT.601 YCbCr -> RGB
//          fused with `2*(rgb/255) - 1`, bilinearly resized to H_OUT/W_OUT.
//
// Templated on the output element type; instantiated for __half and float
// in the .cu.
template <typename TOut>
cudaError_t launchNv12Decode(
    const uint8_t* nv12,
    TOut* output,
    int H_IN,
    int W_IN,
    int H_OUT,
    int W_OUT,
    cudaStream_t stream);

} // namespace whirlwind
