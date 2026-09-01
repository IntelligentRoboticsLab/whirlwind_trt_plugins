#include "common/cuda_check.hpp"
#include "plugins/corr_volume/corr_volume_kernel.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <type_traits>

namespace whirlwind {
namespace {

constexpr int kSmemLimit = 48 * 1024;  // per-block dynamic shared memory budget

// Tiling parameters for the fast kernel. Each thread owns kVecW consecutive
// columns and kDispPerThread consecutive disparities.
constexpr int kVecW = 4;
constexpr int kDispPerThread = 6;

constexpr int kMinTileW = 32;   // narrower tiles lose more to per-block setup
                                // than they gain in occupancy (measured).

inline int roundUpTo(int v, int m) noexcept { return (v + m - 1) / m * m; }

// cudaGetDeviceProperties is expensive enough that calling it per enqueue()
// shows up in a 5 us kernel. Cache per device ordinal.
int smCount() noexcept
{
    static thread_local int cached = 0;
    if (cached == 0) {
        int dev = 0;
        cudaDeviceProp prop{};
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
            cached = 1;
        } else {
            cached = prop.multiProcessorCount;
        }
    }
    return cached;
}

} // namespace

// ---------------------------------------------------------------------------
// Naive kernel — one thread per output element (b, d, y, x); inner loop over C.
//
// Kept as the fallback: it handles any shape and dtype, and because it is one
// thread per output it still saturates a wide GPU on very small volumes, where
// the tiled kernel below is stuck at its launch floor. See launchCorrVolume.
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
// Fast FP16 kernel — vectorised staging + register tiling.
//
// One block covers TILE_W consecutive columns of one image row, for all D
// disparities. Two stages:
//
// 1. Staging. The left tile [C][TILE_W] and the right tile [C][TILE_W + PADA]
//    are copied to shared memory with float4 loads (8 halves per transaction).
//    The right tile is based at column (wt - PADA) with PADA = roundUp(D-1, 8)
//    rather than (D-1), so every staged segment starts on a 16-byte boundary;
//    the extra columns fall off the left edge of the image and stage as zeros,
//    which is exactly the padding the d > x case needs anyway.
//
// 2. Compute. Each thread owns VW consecutive columns AND ND *consecutive*
//    disparities. Consecutiveness is the point: the right-hand operands of all
//    ND*VW products land on only ND+VW-1 distinct shared slots, so one channel
//    iteration costs VW + (ND+VW-1) shared loads to produce ND*VW FMAs. At
//    VW=4, ND=6 that is 13 loads for 24 FMAs (1.8 FMA/load) against 0.5 for a
//    one-output-per-thread formulation. That ratio is where the speedup comes
//    from -- the kernel is otherwise memory bound, and this is what stops the
//    shared-memory pipe being the new bottleneck.
//
// Accumulation is fp32, so this is slightly *more* accurate than an fp16
// HFMA2 reduction over C.
//
// Requires: W % 8 == 0, TILE_W % 8 == 0, D % ND == 0. Enforced by the caller.
// ---------------------------------------------------------------------------
template <int VW, int ND>
__global__ void corrVolumeFastKernel(
    const __half* __restrict__ left,
    const __half* __restrict__ right,
    __half* __restrict__ out,
    int C,
    int H,
    int W,
    int D,
    int TILE_W,
    int PADA,
    float invC)
{
    extern __shared__ __align__(16) unsigned char smem[];
    const int RW = TILE_W + PADA;
    __half* ls = reinterpret_cast<__half*>(smem);
    __half* rs = ls + C * TILE_W;

    const int wt = blockIdx.x * TILE_W;
    const int y = blockIdx.y;
    const int b = blockIdx.z;

    const __half* leftRow  = left  + (static_cast<long long>(b) * C * H + y) * W;
    const __half* rightRow = right + (static_cast<long long>(b) * C * H + y) * W;
    const int channelStride = H * W;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    const int nvL = TILE_W >> 3;
    const int nvR = RW >> 3;

    // Walk the (channel, 8-column group) grid without a per-iteration integer
    // division: divide once, then advance by a precomputed stride.
    {
        int c = tid / nvL, wvi = tid - c * nvL;
        const int dc = nthreads / nvL, dw = nthreads - (nthreads / nvL) * nvL;
        for (int i = tid; i < C * nvL; i += nthreads,
             c += dc, wvi += dw, wvi >= nvL ? (wvi -= nvL, ++c) : 0) {
            const int wv = wvi << 3;
            const int gx = wt + wv;
            __half* dst = &ls[c * TILE_W + wv];
            if (gx + 8 <= W) {
                *reinterpret_cast<float4*>(dst) =
                    *reinterpret_cast<const float4*>(&leftRow[c * channelStride + gx]);
            } else {
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const int g = gx + k;
                    dst[k] = (g < W) ? leftRow[c * channelStride + g] : __half(0.0f);
                }
            }
        }
    }
    {
        int c = tid / nvR, wvi = tid - c * nvR;
        const int dc = nthreads / nvR, dw = nthreads - (nthreads / nvR) * nvR;
        for (int i = tid; i < C * nvR; i += nthreads,
             c += dc, wvi += dw, wvi >= nvR ? (wvi -= nvR, ++c) : 0) {
            const int wv = wvi << 3;
            const int gx = wt - PADA + wv;
            __half* dst = &rs[c * RW + wv];
            if (gx >= 0 && gx + 8 <= W) {
                *reinterpret_cast<float4*>(dst) =
                    *reinterpret_cast<const float4*>(&rightRow[c * channelStride + gx]);
            } else {
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const int g = gx + k;
                    dst[k] = (g >= 0 && g < W) ? rightRow[c * channelStride + g] : __half(0.0f);
                }
            }
        }
    }
    __syncthreads();

    constexpr int RVN = ND + VW - 1;
    const int xl = threadIdx.x * VW;
    const int d0 = threadIdx.y * ND;
    const int gx = wt + xl;
    if (gx >= W) return;

    float acc[ND][VW];
    #pragma unroll
    for (int dd = 0; dd < ND; ++dd)
        #pragma unroll
        for (int v = 0; v < VW; ++v) acc[dd][v] = 0.0f;

    const __half* lp = &ls[xl];
    const __half* rp = &rs[xl + PADA - d0 - (ND - 1)];
    for (int c = 0; c < C; ++c) {
        float lv[VW], rv[RVN];
        #pragma unroll
        for (int v = 0; v < VW; ++v) lv[v] = __half2float(lp[v]);
        #pragma unroll
        for (int k = 0; k < RVN; ++k) rv[k] = __half2float(rp[k]);
        #pragma unroll
        for (int dd = 0; dd < ND; ++dd)
            #pragma unroll
            for (int v = 0; v < VW; ++v) acc[dd][v] += lv[v] * rv[(ND - 1) - dd + v];
        lp += TILE_W;
        rp += RW;
    }

    __half* outB = out + static_cast<long long>(b) * D * H * W;
    const bool full = (gx + VW <= W);
    #pragma unroll
    for (int dd = 0; dd < ND; ++dd) {
        __half* op = &outB[((d0 + dd) * H + y) * W + gx];
        __half tmp[VW];
        #pragma unroll
        for (int v = 0; v < VW; ++v) tmp[v] = __float2half_rn(acc[dd][v] * invC);
        if (full) {
            if (VW == 8)      *reinterpret_cast<float4*>(op) = *reinterpret_cast<float4*>(tmp);
            else if (VW == 4) *reinterpret_cast<float2*>(op) = *reinterpret_cast<float2*>(tmp);
            else if (VW == 2) *reinterpret_cast<float*>(op)  = *reinterpret_cast<float*>(tmp);
            else              op[0] = tmp[0];
        } else {
            #pragma unroll
            for (int v = 0; v < VW; ++v)
                if (gx + v < W) op[v] = tmp[v];
        }
    }
}

namespace {

// Tile width for the fast kernel.
//
// The default is a full-row tile: the right-hand halo then falls off the image
// edge instead of overlapping a neighbouring tile, so every input element is
// read from DRAM exactly once. But one block per row means H*B blocks, and on
// a short image that starves a wide GPU -- at H=32, W=136 on a 4090 a full-row
// tile measured 7.45 us against 5.28 us for a split one. So when rows alone
// cannot supply ~half the SMs, split the row until they can.
int pickTileW(int C, int B, int H, int W, int D, int PADA, int dispY) noexcept
{
    int tile = roundUpTo(W, 8);

    // Only worth splitting a row that is wide enough to still leave a useful
    // tile behind; below 2*kMinTileW the split just trades DRAM re-reads for
    // blocks that are too small to pay for themselves.
    const int rowBlocks = H * B;
    const int target = smCount() / 2;
    if (rowBlocks > 0 && rowBlocks < target && tile > 2 * kMinTileW) {
        const int wantSplit = (target + rowBlocks - 1) / rowBlocks;
        const int split = roundUpTo((W + wantSplit - 1) / wantSplit, 8);
        if (split > 0 && split < tile) tile = split;
        if (tile < kMinTileW) tile = kMinTileW;
    }

    // Shared memory: C * (TILE_W + TILE_W + PADA) halves must fit.
    const int byBytes = (kSmemLimit / (static_cast<int>(sizeof(__half)) * C) - PADA) / 2;
    if (tile > byBytes) tile = byBytes;

    // Block size: (TILE_W / VW) * dispY threads must fit in 1024.
    const int byThreads = 1024 / dispY * kVecW;
    if (tile > byThreads) tile = byThreads;

    tile = tile / 8 * 8;

    // A tile of 0 means shared memory could not hold even one 8-column strip
    // (very large C). Bail out here rather than below: fastPathApplies() will
    // reject it and the caller falls back, and returning early keeps the
    // divisions that follow away from a zero divisor. Note x86 traps on
    // integer division by zero while AArch64 quietly yields 0, so getting this
    // wrong shows up as a crash on one host and silence on the other.
    if (tile < 8) return 0;

    // If the row ends up split -- either for occupancy above, or because the
    // full row did not fit shared memory -- balance the pieces. Taking the
    // largest tile that fits leaves a lopsided remainder: at W=312 the shared
    // memory cap gives tile=280, so the second block covers 32 real columns
    // while still staging a full right halo. Re-dividing into equal tiles keeps
    // the same block count and costs the same total halo, but gives every block
    // real work (measured: removes a ~3% regression at W=312 on sm_87).
    const int gridX = (W + tile - 1) / tile;
    if (gridX > 1) {
        const int balanced = roundUpTo((W + gridX - 1) / gridX, 8);
        if (balanced >= kMinTileW && balanced < tile) tile = balanced;
    }

    return tile;
}

// The fast kernel amortises the per-output channel loop, so its win grows with
// the volume -- but its fixed setup cost is a little above the naive kernel's,
// and on a volume small enough that both sit at their launch floor (~5 us on
// sm_89) the naive kernel's one-thread-per-output parallelism wins instead.
//
// The crossover is where the volume stops giving each SM a meaningful slice, so
// it is expressed per SM rather than as an absolute size: a 4090 stays launch
// bound far longer than an Orin AGX does, and an absolute threshold measured on
// the former would wrongly send the latter down the slow path. The constant is
// measured on sm_89, where it puts the boundary at ~102K outputs -- matching a
// sweep over H in {32..112} x W in {40..312} with one 2% miss.
constexpr long long kMinOutputsPerSM = 800;

bool fastPathApplies(int C, int B, int H, int W, int D, int tile, int dispY) noexcept
{
    if (W % 8 != 0 || D % kDispPerThread != 0) return false;
    const long long outputs = static_cast<long long>(B) * D * H * W;
    if (outputs < static_cast<long long>(smCount()) * kMinOutputsPerSM) return false;
    if (tile < 8 || tile % 8 != 0) return false;
    const int PADA = roundUpTo(D - 1, 8);
    const size_t smem = static_cast<size_t>(C) * (tile + tile + PADA) * sizeof(__half);
    if (smem > kSmemLimit) return false;
    const int threads = (tile / kVecW) * dispY;
    return threads > 0 && threads <= 1024;
}

} // namespace

// ---------------------------------------------------------------------------
// Launch wrapper.
// ---------------------------------------------------------------------------
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
    cudaStream_t stream)
{
    if constexpr (std::is_same_v<T, __half>) {
        const int PADA = roundUpTo(D - 1, 8);
        const int dispY = (D % kDispPerThread == 0) ? D / kDispPerThread : 0;
        const int tile = (dispY > 0) ? pickTileW(C, B, H, W, D, PADA, dispY) : 0;

        if (dispY > 0 && fastPathApplies(C, B, H, W, D, tile, dispY)) {
            dim3 block(tile / kVecW, dispY, 1);
            dim3 grid((W + tile - 1) / tile, H, B);
            size_t shmemBytes =
                static_cast<size_t>(C) * (tile + tile + PADA) * sizeof(__half);

            corrVolumeFastKernel<kVecW, kDispPerThread><<<grid, block, shmemBytes, stream>>>(
                left, right, output, C, H, W, D, tile, PADA, 1.0f / static_cast<float>(C));
            return lastLaunchError(__func__);
        }
    }

    int total = B * D * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    corrVolumeKernel<T><<<blocks, threads, 0, stream>>>(
        left, right, output, B, C, H, W, D);
    return lastLaunchError(__func__);
}

template cudaError_t launchCorrVolume<__half>(
    const __half*, const __half*, __half*, int, int, int, int, int, cudaStream_t);
template cudaError_t launchCorrVolume<float>(
    const float*, const float*, float*, int, int, int, int, int, cudaStream_t);

} // namespace whirlwind
