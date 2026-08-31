#include "common/cuda_check.hpp"
#include "plugins/nv12_decode/nv12_decode_kernel.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <type_traits>

namespace whirlwind {

// ---------------------------------------------------------------------------
// NV12 decode + bilinear resize + normalisation.
//
// Input  : uint8 NV12 buffer of size H_NV12 * W_IN, with H_NV12 = H_IN*3/2.
//          The first H_IN rows are the Y plane; the remaining H_IN/2 rows
//          are interleaved U/V chroma at half resolution.
// Output : planar [3, H_OUT, W_OUT] in [-1, 1] using BT.601 full-range
//          YCbCr -> RGB fused with `2*(rgb/255) - 1`.
//
// Two distinct kernels are provided — one for fp32 output and one for fp16.
// They share device-side helpers (bilinear sampling + colour conversion) but
// differ in their store paths. Keeping them separate lets us tune each
// independently (fp16 can convert in flight; fp32 stores can vectorise).
//
// Resize follows PyTorch align_corners=False:
//     src = (dst + 0.5) * src_dim/dst_dim - 0.5
// Out-of-domain coords are clamped (replicate) before the bilinear blend.
// ---------------------------------------------------------------------------

// ---- Shared device helpers ------------------------------------------------

// Bilinear sample of the Y plane (uint8).
__device__ __forceinline__ float bilinearSampleY(
    const uint8_t* __restrict__ y_plane,
    int H, int W, int rowStride,
    float yf, float xf)
{
    int y0 = static_cast<int>(floorf(yf));
    int x0 = static_cast<int>(floorf(xf));
    int y1 = y0 + 1;
    int x1 = x0 + 1;

    float wy1 = yf - static_cast<float>(y0);
    float wy0 = 1.0f - wy1;
    float wx1 = xf - static_cast<float>(x0);
    float wx0 = 1.0f - wx1;

    int y0c = max(0, min(H - 1, y0));
    int y1c = max(0, min(H - 1, y1));
    int x0c = max(0, min(W - 1, x0));
    int x1c = max(0, min(W - 1, x1));

    float v00 = static_cast<float>(y_plane[y0c * rowStride + x0c]);
    float v01 = static_cast<float>(y_plane[y0c * rowStride + x1c]);
    float v10 = static_cast<float>(y_plane[y1c * rowStride + x0c]);
    float v11 = static_cast<float>(y_plane[y1c * rowStride + x1c]);

    return wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
}

// Bilinear sample of the interleaved UV plane: returns (U, V) as one float2.
// Reading both components together cuts the 8 chroma loads to 4 uchar2 fetches.
__device__ __forceinline__ float2 bilinearSampleUV(
    const uint8_t* __restrict__ chroma,
    int H_UV, int W_UV, int W_IN,
    float yf, float xf)
{
    int y0 = static_cast<int>(floorf(yf));
    int x0 = static_cast<int>(floorf(xf));
    int y1 = y0 + 1;
    int x1 = x0 + 1;

    float wy1 = yf - static_cast<float>(y0);
    float wy0 = 1.0f - wy1;
    float wx1 = xf - static_cast<float>(x0);
    float wx0 = 1.0f - wx1;

    int y0c = max(0, min(H_UV - 1, y0));
    int y1c = max(0, min(H_UV - 1, y1));
    int x0c = max(0, min(W_UV - 1, x0));
    int x1c = max(0, min(W_UV - 1, x1));

    int o00 = y0c * W_IN + 2 * x0c;
    int o01 = y0c * W_IN + 2 * x1c;
    int o10 = y1c * W_IN + 2 * x0c;
    int o11 = y1c * W_IN + 2 * x1c;

    float u00 = static_cast<float>(chroma[o00    ]);
    float v00 = static_cast<float>(chroma[o00 + 1]);
    float u01 = static_cast<float>(chroma[o01    ]);
    float v01 = static_cast<float>(chroma[o01 + 1]);
    float u10 = static_cast<float>(chroma[o10    ]);
    float v10 = static_cast<float>(chroma[o10 + 1]);
    float u11 = static_cast<float>(chroma[o11    ]);
    float v11 = static_cast<float>(chroma[o11 + 1]);

    float U = wy0 * (wx0 * u00 + wx1 * u01) + wy1 * (wx0 * u10 + wx1 * u11);
    float V = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
    return make_float2(U, V);
}

// Decode (Y, Cb, Cr) -> RGB in [-1, 1] using BT.601 full-range fused with
// `2*(rgb/255) - 1`. Cb/Cr are *centred* (already had 128 subtracted).
__device__ __forceinline__ void ycbcrToRgbNorm(
    float Y, float Cb, float Cr,
    float& R, float& G, float& B)
{
    constexpr float s    = 2.0f / 255.0f;
    constexpr float bias = -1.0f;
    R = s * (Y + 1.402f    * Cr)                + bias;
    G = s * (Y - 0.344136f * Cb - 0.714136f * Cr) + bias;
    B = s * (Y + 1.772f    * Cb)                + bias;
}

// ---- FP32 output kernel ---------------------------------------------------
//
// One thread per output pixel; writes 3 floats to the planar [3, H, W] tensor.
__global__ void nv12DecodeKernelF32(
    const uint8_t* __restrict__ nv12,
    float* __restrict__ output,
    int H_IN, int W_IN,
    int H_OUT, int W_OUT,
    float scale_y_h, float scale_y_w,
    float scale_uv_h, float scale_uv_w)
{
    const int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_out >= W_OUT || y_out >= H_OUT) return;

    const float yf_y = (static_cast<float>(y_out) + 0.5f) * scale_y_h - 0.5f;
    const float xf_y = (static_cast<float>(x_out) + 0.5f) * scale_y_w - 0.5f;
    const float Y = bilinearSampleY(nv12, H_IN, W_IN, W_IN, yf_y, xf_y);

    const int H_UV = H_IN >> 1;
    const int W_UV = W_IN >> 1;
    const uint8_t* chroma = nv12 + H_IN * W_IN;

    const float yf_uv = (static_cast<float>(y_out) + 0.5f) * scale_uv_h - 0.5f;
    const float xf_uv = (static_cast<float>(x_out) + 0.5f) * scale_uv_w - 0.5f;
    const float2 uv = bilinearSampleUV(chroma, H_UV, W_UV, W_IN, yf_uv, xf_uv);

    float R, G, B;
    ycbcrToRgbNorm(Y, uv.x - 128.0f, uv.y - 128.0f, R, G, B);

    const int idx = y_out * W_OUT + x_out;
    const int chStride = H_OUT * W_OUT;
    output[idx                ] = R;
    output[idx +     chStride ] = G;
    output[idx + 2 * chStride ] = B;
}

// ---- FP16 output kernel ---------------------------------------------------
//
// Same parallelism as the fp32 kernel, but converts to __half on the way
// out so TRT can keep the downstream conv stem in fp16 with no reformat.
__global__ void nv12DecodeKernelF16(
    const uint8_t* __restrict__ nv12,
    __half* __restrict__ output,
    int H_IN, int W_IN,
    int H_OUT, int W_OUT,
    float scale_y_h, float scale_y_w,
    float scale_uv_h, float scale_uv_w)
{
    const int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_out >= W_OUT || y_out >= H_OUT) return;

    const float yf_y = (static_cast<float>(y_out) + 0.5f) * scale_y_h - 0.5f;
    const float xf_y = (static_cast<float>(x_out) + 0.5f) * scale_y_w - 0.5f;
    const float Y = bilinearSampleY(nv12, H_IN, W_IN, W_IN, yf_y, xf_y);

    const int H_UV = H_IN >> 1;
    const int W_UV = W_IN >> 1;
    const uint8_t* chroma = nv12 + H_IN * W_IN;

    const float yf_uv = (static_cast<float>(y_out) + 0.5f) * scale_uv_h - 0.5f;
    const float xf_uv = (static_cast<float>(x_out) + 0.5f) * scale_uv_w - 0.5f;
    const float2 uv = bilinearSampleUV(chroma, H_UV, W_UV, W_IN, yf_uv, xf_uv);

    float R, G, B;
    ycbcrToRgbNorm(Y, uv.x - 128.0f, uv.y - 128.0f, R, G, B);

    const int idx = y_out * W_OUT + x_out;
    const int chStride = H_OUT * W_OUT;
    output[idx                ] = __float2half_rn(R);
    output[idx +     chStride ] = __float2half_rn(G);
    output[idx + 2 * chStride ] = __float2half_rn(B);
}

// ---------------------------------------------------------------------------
// Launch wrapper.
//
// The TRT graph declares the input as INT8 (IPluginV3 doesn't expose UINT8),
// but the underlying bytes are uint8 NV12 values — the plugin reinterprets
// the device pointer before calling in.
// ---------------------------------------------------------------------------
template <typename TOut>
cudaError_t launchNv12Decode(
    const uint8_t* nv12,
    TOut* output,
    int H_IN,
    int W_IN,
    int H_OUT,
    int W_OUT,
    cudaStream_t stream)
{
    dim3 block(32, 8, 1);
    dim3 grid(
        (W_OUT + block.x - 1) / block.x,
        (H_OUT + block.y - 1) / block.y,
        1);

    const float scale_y_h  = static_cast<float>(H_IN)      / static_cast<float>(H_OUT);
    const float scale_y_w  = static_cast<float>(W_IN)      / static_cast<float>(W_OUT);
    const float scale_uv_h = static_cast<float>(H_IN >> 1) / static_cast<float>(H_OUT);
    const float scale_uv_w = static_cast<float>(W_IN >> 1) / static_cast<float>(W_OUT);

    if constexpr (std::is_same_v<TOut, __half>) {
        nv12DecodeKernelF16<<<grid, block, 0, stream>>>(
            nv12, output,
            H_IN, W_IN, H_OUT, W_OUT,
            scale_y_h, scale_y_w, scale_uv_h, scale_uv_w);
    } else {
        nv12DecodeKernelF32<<<grid, block, 0, stream>>>(
            nv12, output,
            H_IN, W_IN, H_OUT, W_OUT,
            scale_y_h, scale_y_w, scale_uv_h, scale_uv_w);
    }
    return lastLaunchError(__func__);
}

template cudaError_t launchNv12Decode<__half>(
    const uint8_t*, __half*, int, int, int, int, cudaStream_t);
template cudaError_t launchNv12Decode<float>(
    const uint8_t*, float*, int, int, int, int, cudaStream_t);

} // namespace whirlwind
