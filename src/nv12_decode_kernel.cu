#include "nv12_decode_plugin.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <NvInfer.h>

namespace liteany {

// IPluginV3 cannot expose UINT8 as an I/O datatype, so the ONNX graph emits a
// uint8 -> float cast immediately before this plugin (one tiny TRT op). The
// kernel reads the NV12 byte values as floats; arithmetic is unchanged.

// ---------------------------------------------------------------------------
// Bilinear sample of a single plane (already in float) at fractional coords.
// PyTorch align_corners=False -> source = (dst + 0.5) * src_dim/dst_dim - 0.5
// ---------------------------------------------------------------------------
__device__ __forceinline__ float bilinearSamplePlane(
    const float* __restrict__ plane,
    int H,
    int W,
    int rowStride,
    float yf,
    float xf)
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

    float v00 = plane[y0c * rowStride + x0c];
    float v01 = plane[y0c * rowStride + x1c];
    float v10 = plane[y1c * rowStride + x0c];
    float v11 = plane[y1c * rowStride + x1c];

    return wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
}

// Bilinear sample of U or V inside the interleaved NV12 chroma row block.
// `chroma` points to the start of the UV section (`nv12 + H_IN * W_IN`).
// `chOff` is 0 for U, 1 for V.
__device__ __forceinline__ float bilinearSampleNv12Chroma(
    const float* __restrict__ chroma,
    int H_UV,
    int W_UV,
    int W_IN,
    int chOff,
    float yf,
    float xf)
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

    // Each chroma row has W_IN floats (W_UV interleaved U/V pairs).
    float v00 = chroma[y0c * W_IN + 2 * x0c + chOff];
    float v01 = chroma[y0c * W_IN + 2 * x1c + chOff];
    float v10 = chroma[y1c * W_IN + 2 * x0c + chOff];
    float v11 = chroma[y1c * W_IN + 2 * x1c + chOff];

    return wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
}

template <typename T>
__device__ __forceinline__ T floatToOut(float v) {
    return static_cast<T>(v);
}

template <>
__device__ __forceinline__ __half floatToOut<__half>(float v) {
    return __float2half_rn(v);
}

// ---------------------------------------------------------------------------
// One thread per output (y_out, x_out). Writes all 3 channels.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void nv12DecodeKernel(
    const float* __restrict__ nv12,
    T* __restrict__ output,
    int H_IN,
    int W_IN,
    int H_OUT,
    int W_OUT,
    float scale_y_h,
    float scale_y_w,
    float scale_uv_h,
    float scale_uv_w)
{
    const int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_out >= W_OUT || y_out >= H_OUT) return;

    // ---- Y plane: bilinear from (H_IN, W_IN) at row offset 0.
    const float yf_y = (static_cast<float>(y_out) + 0.5f) * scale_y_h - 0.5f;
    const float xf_y = (static_cast<float>(x_out) + 0.5f) * scale_y_w - 0.5f;
    const float Y = bilinearSamplePlane(nv12, H_IN, W_IN, W_IN, yf_y, xf_y);

    // ---- UV plane: starts at row H_IN, half resolution interleaved.
    const int H_UV = H_IN >> 1;
    const int W_UV = W_IN >> 1;
    const float* chroma = nv12 + H_IN * W_IN;

    const float yf_uv = (static_cast<float>(y_out) + 0.5f) * scale_uv_h - 0.5f;
    const float xf_uv = (static_cast<float>(x_out) + 0.5f) * scale_uv_w - 0.5f;
    const float U = bilinearSampleNv12Chroma(chroma, H_UV, W_UV, W_IN, 0, yf_uv, xf_uv);
    const float V = bilinearSampleNv12Chroma(chroma, H_UV, W_UV, W_IN, 1, yf_uv, xf_uv);

    const float Cb = U - 128.0f;
    const float Cr = V - 128.0f;

    // BT.601 full-range YCbCr -> RGB, fused with `2*(rgb/255) - 1`.
    constexpr float s = 2.0f / 255.0f;
    float R = fmaf(s, Y + 1.402f * Cr, -1.0f);
    float G = fmaf(s, Y - 0.344136f * Cb - 0.714136f * Cr, -1.0f);
    float B = fmaf(s, Y + 1.772f * Cb, -1.0f);

    R = fminf(1.0f, fmaxf(-1.0f, R));
    G = fminf(1.0f, fmaxf(-1.0f, G));
    B = fminf(1.0f, fmaxf(-1.0f, B));

    const int idx = y_out * W_OUT + x_out;
    const int chStride = H_OUT * W_OUT;
    output[idx]                 = floatToOut<T>(R);
    output[idx + chStride]      = floatToOut<T>(G);
    output[idx + 2 * chStride]  = floatToOut<T>(B);
}

// ---------------------------------------------------------------------------
// Dispatcher.
// ---------------------------------------------------------------------------
void launchNv12Decode(
    const void* nv12,
    void* output,
    int H_IN,
    int W_IN,
    int H_OUT,
    int W_OUT,
    nvinfer1::DataType outDtype,
    cudaStream_t stream)
{
    dim3 block(16, 16, 1);
    dim3 grid(
        (W_OUT + block.x - 1) / block.x,
        (H_OUT + block.y - 1) / block.y,
        1);

    const float scale_y_h  = static_cast<float>(H_IN) / static_cast<float>(H_OUT);
    const float scale_y_w  = static_cast<float>(W_IN) / static_cast<float>(W_OUT);
    const float scale_uv_h = static_cast<float>(H_IN >> 1) / static_cast<float>(H_OUT);
    const float scale_uv_w = static_cast<float>(W_IN >> 1) / static_cast<float>(W_OUT);

    if (outDtype == nvinfer1::DataType::kHALF) {
        nv12DecodeKernel<__half><<<grid, block, 0, stream>>>(
            static_cast<const float*>(nv12),
            static_cast<__half*>(output),
            H_IN, W_IN, H_OUT, W_OUT,
            scale_y_h, scale_y_w, scale_uv_h, scale_uv_w);
    } else {
        nv12DecodeKernel<float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(nv12),
            static_cast<float*>(output),
            H_IN, W_IN, H_OUT, W_OUT,
            scale_y_h, scale_y_w, scale_uv_h, scale_uv_w);
    }
}

} // namespace liteany
