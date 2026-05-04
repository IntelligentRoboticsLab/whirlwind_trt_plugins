#include "corr_volume_plugin.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

namespace liteany {

// ---------------------------------------------------------------------------
// Naive fallback kernel.
//
// One thread per output element (b, d, y, x); inner loop over C.
// Used for FP32 and for any FP16 shape that does not match the fast path.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void corrVolumeKernel(
    const T* __restrict__ left,
    const T* __restrict__ right,
    T* __restrict__ out,
    int B,
    int C,
    int H,
    int W,
    int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D * H * W;
    if (idx >= total) return;

    int x = idx % W;
    int y = (idx / W) % H;
    int d = (idx / (W * H)) % D;
    int b = idx / (W * H * D);

    int xr = x - d;

    float acc = 0.0f;

    if (xr >= 0) {
        for (int c = 0; c < C; ++c) {
            int lidx = ((b * C + c) * H + y) * W + x;
            int ridx = ((b * C + c) * H + y) * W + xr;

            acc += static_cast<float>(left[lidx]) *
                   static_cast<float>(right[ridx]);
        }

        acc /= static_cast<float>(C);
    }

    out[idx] = static_cast<T>(acc);
}

template <>
__global__ void corrVolumeKernel<half>(
    const half* __restrict__ left,
    const half* __restrict__ right,
    half* __restrict__ out,
    int B,
    int C,
    int H,
    int W,
    int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D * H * W;
    if (idx >= total) return;

    int x = idx % W;
    int y = (idx / W) % H;
    int d = (idx / (W * H)) % D;
    int b = idx / (W * H * D);

    int xr = x - d;

    float acc = 0.0f;

    if (xr >= 0) {
        for (int c = 0; c < C; ++c) {
            int lidx = ((b * C + c) * H + y) * W + x;
            int ridx = ((b * C + c) * H + y) * W + xr;

            acc += __half2float(left[lidx]) * __half2float(right[ridx]);
        }

        acc /= static_cast<float>(C);
    }

    out[idx] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// Fast row-tiled FP16 kernel.
//
// One block per (y, b). The block has W threads; each thread owns one x and
// computes all D disparities for that x.
//
// Per channel:
//   * each thread loads one left[c, y, x] into a register
//   * threads cooperatively load right[c, y, *] into shared memory once
//   * the shared row is prefixed with (D-1) zeros so the inner unrolled loop
//     over d can index sR[x + D - 1 - d] without a bounds check
//
// Disparities are processed two at a time using HFMA2 (`__hmul2` on a
// broadcast left value × a packed (right[x-d], right[x-d-1])), giving 2x the
// fp16 throughput of the scalar form.
//
// Bandwidth: each left/right element is loaded from global memory exactly
// once per (b, c, y) — a D-fold reduction vs the naive kernel.
// ---------------------------------------------------------------------------
template <int D_TILE, int W_TILE>
__global__ void corrVolumeRowKernel(
    const __half* __restrict__ left,
    const __half* __restrict__ right,
    __half* __restrict__ out,
    int C,
    int H)
{
    static_assert((D_TILE & 1) == 0, "D_TILE must be even for HFMA2 path");

    extern __shared__ __half sR[];   // size: W_TILE + D_TILE - 1

    const int x = threadIdx.x;
    const int y = blockIdx.x;
    const int b = blockIdx.y;

    // Zero-pad the (D_TILE-1) entries that correspond to xr = x - d < 0.
    // Done once at the start; the loop below only ever writes to indices
    // [D_TILE-1, W_TILE + D_TILE - 2].
    if (x < D_TILE - 1) {
        sR[x] = __float2half_rn(0.0f);
    }
    __syncthreads();

    float acc[D_TILE];
    #pragma unroll
    for (int d = 0; d < D_TILE; ++d) acc[d] = 0.0f;

    const int channelStride = H * W_TILE;
    const int rowOffset = (b * C * H + y) * W_TILE + x;   // c == 0

    for (int c = 0; c < C; ++c) {
        const int idx = rowOffset + c * channelStride;
        const __half lv = left[idx];
        sR[x + D_TILE - 1] = right[idx];
        __syncthreads();

        const __half2 lv2 = __half2half2(lv);

        #pragma unroll
        for (int d = 0; d < D_TILE; d += 2) {
            const int ix = x + D_TILE - 1 - d;
            const __half2 rh2 = __halves2half2(sR[ix], sR[ix - 1]);
            const __half2 ph2 = __hmul2(lv2, rh2);
            acc[d    ] += __half2float(__low2half(ph2));
            acc[d + 1] += __half2float(__high2half(ph2));
        }
        __syncthreads();
    }

    const float invC = 1.0f / static_cast<float>(C);
    const int outBase = ((b * D_TILE) * H + y) * W_TILE + x;
    const int outStride = H * W_TILE;

    #pragma unroll
    for (int d = 0; d < D_TILE; ++d) {
        out[outBase + d * outStride] = __float2half_rn(acc[d] * invC);
    }
}

// ---------------------------------------------------------------------------
// Dispatcher.
// ---------------------------------------------------------------------------
void launchCorrVolume(
    const void* left,
    const void* right,
    void* output,
    int B,
    int C,
    int H,
    int W,
    int D,
    nvinfer1::DataType dtype,
    cudaStream_t stream)
{
    // Fast path: FP16, the LiteAnyStereo configuration (D=48, W=72).
    // Any other shape falls through to the naive kernel.
    if (dtype == nvinfer1::DataType::kHALF && D == 48 && W == 72) {
        constexpr int D_TILE = 48;
        constexpr int W_TILE = 72;

        dim3 grid(H, B, 1);
        dim3 block(W_TILE, 1, 1);
        size_t shmemBytes = (W_TILE + D_TILE - 1) * sizeof(__half);

        corrVolumeRowKernel<D_TILE, W_TILE><<<grid, block, shmemBytes, stream>>>(
            static_cast<const __half*>(left),
            static_cast<const __half*>(right),
            static_cast<__half*>(output),
            C, H);
        return;
    }

    int total = B * D * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (dtype == nvinfer1::DataType::kHALF) {
        corrVolumeKernel<half><<<blocks, threads, 0, stream>>>(
            static_cast<const half*>(left),
            static_cast<const half*>(right),
            static_cast<half*>(output),
            B, C, H, W, D);
    } else {
        corrVolumeKernel<float><<<blocks, threads, 0, stream>>>(
            static_cast<const float*>(left),
            static_cast<const float*>(right),
            static_cast<float*>(output),
            B, C, H, W, D);
    }
}

} // namespace liteany
