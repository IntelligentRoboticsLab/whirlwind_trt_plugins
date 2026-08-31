#include "common/cuda_check.hpp"
#include "plugins/cost_stem_3d/cost_stem_3d_kernel.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace whirlwind {

// ---------------------------------------------------------------------------
// Fused 3-stage 3D Conv (3x3x3, padding 1) kernel.
//
// Stages (matching cost_stem_3d):
//   0: 1 -> 4 channels  (Conv3D + BN-folded bias + ReLU6)
//   1: 4 -> 4 channels  (Conv3D + BN-folded bias + ReLU6)
//   2: 4 -> 1 channel   (Conv3D + BN-folded bias + ReLU6)
//
// Tile design:
//   * Each block produces an 8x8x8 output cube along (D, H, W).
//   * Halo of 3 each side → input cube of 14x14x14 (2.7 KB fp16).
//   * Stage-0 output (4 ch × 12³)             = 13.5 KB shmem
//   * Stage-1 output (4 ch × 10³)             =  8 KB shmem
//   * Stage-2 output is written direct to global mem (8x8x8 per block, 1 KB).
//   * Block size = 256 threads.
//
// Boundary handling:
//   The convolution is "same" padded; out-of-domain reads are clamped to
//   zero. We materialise this in shared memory: when loading the input
//   tile, threads whose global coord is outside [0,D)x[0,H)x[0,W) write 0.
//
// Weights live in __constant__ memory — 657 fp16 values, well under the
// 64 KB constant-memory budget.
// ---------------------------------------------------------------------------

// Tile design notes:
//
// 8x8x8 output tile -> 120 blocks (under-utilises 4090 SMs)
// 4x4x4 output tile -> 960 blocks (better, but per-block work too small;
//                                  also halo overhead dominates)
// 6x4x4 output tile -> 480 blocks; per-block volume is 6*4*4=96 outputs
//                                  with reasonable halo amortisation.
// On Orin (16 SMs) 120 is plenty; on 4090 we want >=512.
//
// Stage-2 has only OUT_TILE_VOL outputs to compute; if BLOCK_THREADS >
// OUT_TILE_VOL then those threads idle for stage 2. Pick BLOCK_THREADS
// equal to the largest stage's element count divided down. Stage 1
// dominates at 4 * S1_TILE_VOL elements.
constexpr int OUT_TILE_D = 6;
constexpr int OUT_TILE_H = 4;
constexpr int OUT_TILE_W = 4;
constexpr int HALO       = 3;  // 3-stage 3x3x3 fan-out
constexpr int IN_TILE_D  = OUT_TILE_D + 2 * HALO;  // 12
constexpr int IN_TILE_H  = OUT_TILE_H + 2 * HALO;  // 10
constexpr int IN_TILE_W  = OUT_TILE_W + 2 * HALO;  // 10
constexpr int S0_TILE_D  = OUT_TILE_D + 2 * 2;     // 10
constexpr int S0_TILE_H  = OUT_TILE_H + 2 * 2;     // 8
constexpr int S0_TILE_W  = OUT_TILE_W + 2 * 2;     // 8
constexpr int S1_TILE_D  = OUT_TILE_D + 2 * 1;     // 8
constexpr int S1_TILE_H  = OUT_TILE_H + 2 * 1;     // 6
constexpr int S1_TILE_W  = OUT_TILE_W + 2 * 1;     // 6

constexpr int IN_TILE_VOL = IN_TILE_D * IN_TILE_H * IN_TILE_W;        // 1200
constexpr int S0_TILE_VOL = S0_TILE_D * S0_TILE_H * S0_TILE_W;        // 640
constexpr int S1_TILE_VOL = S1_TILE_D * S1_TILE_H * S1_TILE_W;        // 288
constexpr int OUT_TILE_VOL = OUT_TILE_D * OUT_TILE_H * OUT_TILE_W;    // 96
constexpr int BLOCK_THREADS = 128;

__device__ __forceinline__ float relu6(float v) {
    return fminf(6.0f, fmaxf(0.0f, v));
}

__global__ void costStem3DFusedKernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    const __half* __restrict__ w0, const __half* __restrict__ b0,
    const __half* __restrict__ w1, const __half* __restrict__ b1,
    const __half* __restrict__ w2, const __half* __restrict__ b2,
    int B, int D, int H, int W)
{
    extern __shared__ __half smem[];
    __half* sIn = smem;                                          // 14*14*14
    __half* sS0 = smem + IN_TILE_VOL;                            // 4 * 12*12*12
    __half* sS1 = sS0  + 4 * S0_TILE_VOL;                        // 4 * 10*10*10

    const int tile_d = blockIdx.x;
    const int tile_h = blockIdx.y;
    const int tile_w = blockIdx.z;

    const int gd0 = tile_d * OUT_TILE_D - HALO;  // global d coord of sIn[0,0,0]
    const int gh0 = tile_h * OUT_TILE_H - HALO;
    const int gw0 = tile_w * OUT_TILE_W - HALO;

    const int b = 0;  // batch is fixed at 1

    // ---- Stage A: load 14x14x14 input tile (with zero-pad outside-domain).
    for (int i = threadIdx.x; i < IN_TILE_VOL; i += BLOCK_THREADS) {
        int sd = i / (IN_TILE_H * IN_TILE_W);
        int rem = i - sd * IN_TILE_H * IN_TILE_W;
        int sh = rem / IN_TILE_W;
        int sw = rem - sh * IN_TILE_W;
        int gd = gd0 + sd;
        int gh = gh0 + sh;
        int gw = gw0 + sw;
        __half v = __float2half_rn(0.0f);
        if (gd >= 0 && gd < D && gh >= 0 && gh < H && gw >= 0 && gw < W) {
            v = input[((b * D + gd) * H + gh) * W + gw];
        }
        sIn[i] = v;
    }
    __syncthreads();

    // ---- Stage 0: 1 -> 4 channels, output tile 12x12x12 (in 4 channels).
    // Each output element reads a 3x3x3 window of the 14x14x14 input tile.
    // Output position (d, h, w) in [0, 12) corresponds to sIn position
    // (d, h, w) .. (d+2, h+2, w+2).
    for (int i = threadIdx.x; i < 4 * S0_TILE_VOL; i += BLOCK_THREADS) {
        int co = i / S0_TILE_VOL;
        int rem = i - co * S0_TILE_VOL;
        int d = rem / (S0_TILE_H * S0_TILE_W);
        int rem2 = rem - d * S0_TILE_H * S0_TILE_W;
        int h = rem2 / S0_TILE_W;
        int w = rem2 - h * S0_TILE_W;

        float acc = 0.0f;
        // Cin = 1 for stage 0 (one input channel)
        #pragma unroll
        for (int kd = 0; kd < 3; ++kd) {
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    int sd = d + kd;
                    int sh = h + kh;
                    int sw = w + kw;
                    float v = __half2float(sIn[(sd * IN_TILE_H + sh) * IN_TILE_W + sw]);
                    float ww = __half2float(w0[((co * 1 + 0) * 3 + kd) * 9 + kh * 3 + kw]);
                    acc += v * ww;
                }
            }
        }
        acc += __half2float(b0[co]);
        // Zero out positions whose *global* coord is outside the domain,
        // so the next stage's conv treats them as PyTorch's padding=1 zeros.
        // Stage-0 output at (d, h, w) corresponds to global
        //   (gd0 + 1 + d, gh0 + 1 + h, gw0 + 1 + w).
        int g_d = gd0 + 1 + d;
        int g_h = gh0 + 1 + h;
        int g_w = gw0 + 1 + w;
        float v = (g_d < 0 || g_d >= D || g_h < 0 || g_h >= H ||
                   g_w < 0 || g_w >= W) ? 0.0f : relu6(acc);
        sS0[(co * S0_TILE_D + d) * S0_TILE_H * S0_TILE_W + h * S0_TILE_W + w]
            = __float2half_rn(v);
    }
    __syncthreads();

    // ---- Stage 1: 4 -> 4 channels, output tile 10x10x10.
    for (int i = threadIdx.x; i < 4 * S1_TILE_VOL; i += BLOCK_THREADS) {
        int co = i / S1_TILE_VOL;
        int rem = i - co * S1_TILE_VOL;
        int d = rem / (S1_TILE_H * S1_TILE_W);
        int rem2 = rem - d * S1_TILE_H * S1_TILE_W;
        int h = rem2 / S1_TILE_W;
        int w = rem2 - h * S1_TILE_W;

        float acc = 0.0f;
        #pragma unroll
        for (int ci = 0; ci < 4; ++ci) {
            const int chOff = ci * S0_TILE_D * S0_TILE_H * S0_TILE_W;
            #pragma unroll
            for (int kd = 0; kd < 3; ++kd) {
                #pragma unroll
                for (int kh = 0; kh < 3; ++kh) {
                    #pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        int sd = d + kd;
                        int sh = h + kh;
                        int sw = w + kw;
                        float v = __half2float(sS0[chOff + (sd * S0_TILE_H + sh) * S0_TILE_W + sw]);
                        float ww = __half2float(w1[((co * 4 + ci) * 3 + kd) * 9 + kh * 3 + kw]);
                        acc += v * ww;
                    }
                }
            }
        }
        acc += __half2float(b1[co]);
        // Same out-of-domain zero-pad trick as stage 0. Stage-1 output at
        // (d, h, w) is global (gd0 + 2 + d, gh0 + 2 + h, gw0 + 2 + w).
        int g_d = gd0 + 2 + d;
        int g_h = gh0 + 2 + h;
        int g_w = gw0 + 2 + w;
        float v = (g_d < 0 || g_d >= D || g_h < 0 || g_h >= H ||
                   g_w < 0 || g_w >= W) ? 0.0f : relu6(acc);
        sS1[(co * S1_TILE_D + d) * S1_TILE_H * S1_TILE_W + h * S1_TILE_W + w]
            = __float2half_rn(v);
    }
    __syncthreads();

    // ---- Stage 2: 4 -> 1 channel, output tile 8x8x8 → write to global out.
    for (int i = threadIdx.x; i < OUT_TILE_VOL; i += BLOCK_THREADS) {
        int d = i / (OUT_TILE_H * OUT_TILE_W);
        int rem = i - d * OUT_TILE_H * OUT_TILE_W;
        int h = rem / OUT_TILE_W;
        int w = rem - h * OUT_TILE_W;

        // Bounds check against the global output extent (some tiles spill).
        int gd = tile_d * OUT_TILE_D + d;
        int gh = tile_h * OUT_TILE_H + h;
        int gw = tile_w * OUT_TILE_W + w;
        if (gd >= D || gh >= H || gw >= W) continue;

        float acc = 0.0f;
        #pragma unroll
        for (int ci = 0; ci < 4; ++ci) {
            const int chOff = ci * S1_TILE_D * S1_TILE_H * S1_TILE_W;
            #pragma unroll
            for (int kd = 0; kd < 3; ++kd) {
                #pragma unroll
                for (int kh = 0; kh < 3; ++kh) {
                    #pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        int sd = d + kd;
                        int sh = h + kh;
                        int sw = w + kw;
                        float v = __half2float(sS1[chOff + (sd * S1_TILE_H + sh) * S1_TILE_W + sw]);
                        float ww = __half2float(w2[((0 * 4 + ci) * 3 + kd) * 9 + kh * 3 + kw]);
                        acc += v * ww;
                    }
                }
            }
        }
        acc += __half2float(b2[0]);
        output[((b * D + gd) * H + gh) * W + gw] = __float2half_rn(relu6(acc));
    }
}

// ---------------------------------------------------------------------------
// Launch wrapper. Weights live as ONNX initializers; TRT supplies device
// pointers via the plugin's input slots, so we just forward them.
// ---------------------------------------------------------------------------
cudaError_t launchCostStem3D(
    const __half* input,
    __half* output,
    const __half* w0, const __half* b0,
    const __half* w1, const __half* b1,
    const __half* w2, const __half* b2,
    int B, int D, int H, int W,
    cudaStream_t stream)
{
    dim3 grid((D + OUT_TILE_D - 1) / OUT_TILE_D,
              (H + OUT_TILE_H - 1) / OUT_TILE_H,
              (W + OUT_TILE_W - 1) / OUT_TILE_W);
    dim3 block(BLOCK_THREADS);

    size_t shmem = sizeof(__half) * (
        IN_TILE_VOL +
        4 * S0_TILE_VOL +
        4 * S1_TILE_VOL);

    costStem3DFusedKernel<<<grid, block, shmem, stream>>>(
        input, output, w0, b0, w1, b1, w2, b2, B, D, H, W);
    return lastLaunchError(__func__);
}

}  // namespace whirlwind
