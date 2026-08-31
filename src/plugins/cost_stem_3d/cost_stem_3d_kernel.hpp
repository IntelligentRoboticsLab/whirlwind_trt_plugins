#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace whirlwind {

// Fused (Conv3D 3x3x3 + BN + Clip[0,6]) x3 over a cost-volume tensor.
//
// input/output : fp16 [B, 1, D, H, W]
// w0/b0        : fp16 [4, 1, 3, 3, 3] / [4]
// w1/b1        : fp16 [4, 4, 3, 3, 3] / [4]
// w2/b2        : fp16 [1, 4, 3, 3, 3] / [1]
//
// fp16-only — the fused kernel holds its intermediates in shared memory and
// is sized for half precision, so there is nothing to dispatch on here.
cudaError_t launchCostStem3D(
    const __half* input,
    __half* output,
    const __half* w0,
    const __half* b0,
    const __half* w1,
    const __half* b1,
    const __half* w2,
    const __half* b2,
    int B,
    int D,
    int H,
    int W,
    cudaStream_t stream);

} // namespace whirlwind
