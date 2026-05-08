#include "corr_volume_plugin.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

namespace liteany {

// ---------------------------------------------------------------------------
// Naive kernel — one thread per output element (b, d, y, x); inner loop over C.
// Used for FP32 only and as a fallback.
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

// ---------------------------------------------------------------------------
// Naive-vec FP16 kernel — same parallelism as the naive scalar kernel but
// loads 8 halves at a time per side via half4 (16 B per LDG.E).
//
// Requires C % 8 == 0 (LiteAnyStereo: C=24 always satisfies). Each thread
// still computes one (b, d, y, x), but the inner channel loop now does
// C/8 vector loads instead of C scalar loads, cutting global-memory
// transactions ~4x and freeing the ALU to issue HFMA2 pairs.
// ---------------------------------------------------------------------------
__global__ void corrVolumeKernelVec(
    const __half* __restrict__ left,
    const __half* __restrict__ right,
    __half* __restrict__ out,
    int B, int C, int H, int W, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * D * H * W;
    if (idx >= total) return;

    int x = idx % W;
    int y = (idx / W) % H;
    int d = (idx / (W * H)) % D;
    int b = idx / (W * H * D);

    int xr = x - d;
    if (xr < 0) {
        out[idx] = __float2half_rn(0.0f);
        return;
    }

    const int vecC = C >> 3;     // C/8 — guaranteed divisible
    const int chPerOff = H * W;  // bytes/elements per channel
    const int baseL = (b * C * H + y) * W + x;
    const int baseR = (b * C * H + y) * W + xr;

    float2 acc2 = make_float2(0.0f, 0.0f);

    #pragma unroll 1
    for (int v = 0; v < vecC; ++v) {
        // Each iteration covers 8 channels at once. Use 2 half4 loads per side.
        const __half2* lp = reinterpret_cast<const __half2*>(&left [baseL + (v * 8) * chPerOff]);
        const __half2* rp = reinterpret_cast<const __half2*>(&right[baseR + (v * 8) * chPerOff]);

        // Load c, c+1 (both via __half2 strided by chPerOff/2 in half2 indexing).
        // Easiest: load each pair manually — strided, can't be one LDG.
        // Layout is NCHW so channel stride is chPerOff halves -> stride in half2 is chPerOff/2.
        // chPerOff is even because H*W = 32*40 = 1280 (s025) / 56*72 (s05).
        const int hp = chPerOff >> 1;

        __half2 l0 = lp[0];
        __half2 l1 = lp[hp];
        __half2 l2 = lp[2*hp];
        __half2 l3 = lp[3*hp];

        __half2 r0 = rp[0];
        __half2 r1 = rp[hp];
        __half2 r2 = rp[2*hp];
        __half2 r3 = rp[3*hp];

        __half2 p0 = __hmul2(l0, r0);
        __half2 p1 = __hmul2(l1, r1);
        __half2 p2 = __hmul2(l2, r2);
        __half2 p3 = __hmul2(l3, r3);

        __half2 sum01 = __hadd2(p0, p1);
        __half2 sum23 = __hadd2(p2, p3);
        __half2 sum   = __hadd2(sum01, sum23);

        acc2.x += __half2float(__low2half(sum));
        acc2.y += __half2float(__high2half(sum));
    }

    float acc = (acc2.x + acc2.y) / static_cast<float>(C);
    out[idx] = __float2half_rn(acc);
}

// ---------------------------------------------------------------------------
// Fast row-tiled FP16 kernel — generalised over W.
//
// Layout:
//   * block.x = W threads — one thread per output column. (W can be < 32 or
//     non-multiple-of-32; partial-warp blocks are still cheap because each
//     thread does substantial work.)
//   * one block per (y, b)
//
// Per channel:
//   * each thread loads one left[c, y, x] into a register
//   * threads cooperatively load right[c, y, *] into shared memory once
//   * the shared row is prefixed with (D-1) zeros so the inner unrolled loop
//     over d can index sR[x + D - 1 - d] without a bounds check
//
// Disparities are processed two at a time using HFMA2 (`__hmul2` on a
// broadcast left value × a packed (right[x-d], right[x-d-1])), giving 2x the
// fp16 throughput of the scalar form. Accumulation is in fp32 (one register
// per disparity per thread) — that costs D registers per thread (48 for the
// canonical config), well below the per-thread limit on every supported
// architecture.
//
// Bandwidth: each left/right element is loaded from global memory exactly
// once per (b, c, y) — a D-fold reduction vs the naive kernel.
//
// Edge: the strided prologue zeros sR[0 .. D-2] and works for any W >= 1.
// ---------------------------------------------------------------------------
template <int D_TILE>
__global__ void corrVolumeRowKernel(
    const __half* __restrict__ left,
    const __half* __restrict__ right,
    __half* __restrict__ out,
    int C,
    int H,
    int W)
{
    static_assert((D_TILE & 1) == 0, "D_TILE must be even for HFMA2 path");

    extern __shared__ __half sR[];   // size: W + D_TILE - 1

    const int x = threadIdx.x;       // == [0, W)
    const int y = blockIdx.x;
    const int b = blockIdx.y;

    // Strided zero-fill of the (D_TILE-1) pad slots. Works for any W,
    // including W < D_TILE-1 (s025 case: W=40, D_TILE-1=47).
    for (int i = x; i < D_TILE - 1; i += blockDim.x) {
        sR[i] = __float2half_rn(0.0f);
    }
    __syncthreads();

    float acc[D_TILE];
    #pragma unroll
    for (int d = 0; d < D_TILE; ++d) acc[d] = 0.0f;

    const int channelStride = H * W;
    const int rowOffset = (b * C * H + y) * W + x;   // c == 0

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
    const int outBase = ((b * D_TILE) * H + y) * W + x;
    const int outStride = H * W;

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
    // Fast path: FP16, D == 48, W >= 64.
    //
    // The row-tiled kernel launches one block per (y, b) — only H*B blocks
    // total. On wide GPUs (4090: 128 SMs) the small block count starves the
    // SMs and the naive kernel (one thread per output, 60K+ threads) wins.
    // We gate on W >= 64 because at that point the per-block work amortises
    // launch overhead; below that the naive kernel is consistently faster
    // (microbench: at W=40 row-tiled is 2.4x slower than naive on sm_89).
    //
    // On Orin AGX (16 SMs, lower memory BW) the trade-off may flip — keep
    // the row-tiled kernel ready for the s05 case where W=72 already uses
    // it, and re-tune the threshold for sm_87 if needed.
    if (dtype == nvinfer1::DataType::kHALF && D == 48 && W >= 64) {
        constexpr int D_TILE = 48;

        dim3 grid(H, B, 1);
        dim3 block(W, 1, 1);
        size_t shmemBytes = (W + D_TILE - 1) * sizeof(__half);

        corrVolumeRowKernel<D_TILE><<<grid, block, shmemBytes, stream>>>(
            static_cast<const __half*>(left),
            static_cast<const __half*>(right),
            static_cast<__half*>(output),
            C, H, W);
        return;
    }

    int total = B * D * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (dtype == nvinfer1::DataType::kHALF) {
        // The __half2-vectorised variant (`corrVolumeKernelVec`) is faster
        // in microbench but mis-aligns global loads when `x` is odd because
        // the base address spans an odd half-stride. Re-enable it only if
        // the layout guarantees even base addresses (e.g., NHWC with C%8==0
        // and aligned tensors). For now use the scalar fallback.
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
