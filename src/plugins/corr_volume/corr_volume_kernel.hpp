#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace whirlwind {

// Correlation volume over a stereo pair:
//   out[b, d, y, x] = mean_c( left[b, c, y, x] * right[b, c, y, x - d] )
//
// left/right : [B, C, H, W]
// output     : [B, D, H, W]
//
// Templated on the element type rather than switched on a TRT DataType so
// the kernels stay callable — and testable — without TensorRT. Instantiated
// for __half and float in the .cu.
template <typename T>
cudaError_t launchCorrVolume(
    const T* left,
    const T* right,
    T* output,
    int B,
    int C,
    int H,
    int W,
    int D,
    cudaStream_t stream);

} // namespace whirlwind
