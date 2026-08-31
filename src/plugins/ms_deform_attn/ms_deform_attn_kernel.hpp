#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace whirlwind {

// Multi-scale deformable attention, in the three input layouts the plugin
// accepts. All three are templated on their element types and instantiated
// for __half/float in the .cu, so nothing here depends on TensorRT.

// 7-input compatibility mode. value0/1/2 are per-level feature maps
// [B*numHeads, C, H, W]; grid0/1/2 are GridSample-style [-1, 1] sampling
// grids [B*numHeads, Q, P, 2]; weights is [B, Q, numHeads, P0+P1+P2].
// Output: [B, Q, numHeads*C].
//
// The three tensors are independently typed because TRT may hand us fp32
// grids alongside fp16 values.
template <typename TValue, typename TGrid, typename TWeight>
cudaError_t launchMSDeformAttn(
    const TValue* value0,
    const TValue* value1,
    const TValue* value2,
    const TGrid* grid0,
    const TGrid* grid1,
    const TGrid* grid2,
    const TWeight* weights,
    TValue* output,
    int BH,
    int C,
    int Q,
    int H0,
    int W0,
    int H1,
    int W1,
    int H2,
    int W2,
    int numHeads,
    cudaStream_t stream);

// 5-input mode: one packed `locations` tensor [B, Q, numHeads, P, 2] in
// [0, 1] coordinates replaces the three grids.
template <typename T>
cudaError_t launchMSDeformAttnFromLocations(
    const T* value0,
    const T* value1,
    const T* value2,
    const T* locations,
    const T* weights,
    T* output,
    int B,
    int C,
    int Q,
    int H0,
    int W0,
    int H1,
    int W1,
    int H2,
    int W2,
    int numHeads,
    cudaStream_t stream);

// 3-input flat mode: the levels are pre-concatenated into a single
// [B, 8400, numHeads, C] value tensor.
template <typename T>
cudaError_t launchMSDeformAttnFlat(
    const T* value,
    const T* locations,
    const T* weights,
    T* output,
    int B,
    int C,
    int Q,
    int numHeads,
    cudaStream_t stream);

} // namespace whirlwind
