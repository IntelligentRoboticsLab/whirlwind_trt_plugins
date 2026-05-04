#include "ms_deform_attn_plugin.hpp"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace liteany {

template <typename T>
__device__ __forceinline__ float loadValue(const T* ptr, int idx)
{
    return static_cast<float>(ptr[idx]);
}

template <>
__device__ __forceinline__ float loadValue<half>(const half* ptr, int idx)
{
    return __half2float(ptr[idx]);
}

template <typename T>
__device__ __forceinline__ T storeValue(float value)
{
    return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half storeValue<half>(float value)
{
    return __float2half_rn(value);
}

template <typename T>
__device__ __forceinline__ float bilinearGridSample(
    const T* __restrict__ value,
    int bh,
    int c,
    int C,
    int H,
    int W,
    float gx,
    float gy)
{
    // PyTorch grid_sample align_corners=False:
    // source = ((grid + 1) * size - 1) / 2.
    const float x = 0.5f * ((gx + 1.0f) * static_cast<float>(W) - 1.0f);
    const float y = 0.5f * ((gy + 1.0f) * static_cast<float>(H) - 1.0f);

    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float wx1 = x - static_cast<float>(x0);
    const float wy1 = y - static_cast<float>(y0);
    const float wx0 = 1.0f - wx1;
    const float wy0 = 1.0f - wy1;

    float acc = 0.0f;
    const int base = ((bh * C + c) * H) * W;

    if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) {
        acc += loadValue(value, base + y0 * W + x0) * wx0 * wy0;
    }
    if (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H) {
        acc += loadValue(value, base + y0 * W + x1) * wx1 * wy0;
    }
    if (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H) {
        acc += loadValue(value, base + y1 * W + x0) * wx0 * wy1;
    }
    if (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H) {
        acc += loadValue(value, base + y1 * W + x1) * wx1 * wy1;
    }

    return acc;
}

template <typename T>
__device__ __forceinline__ float bilinearUnitSample(
    const T* __restrict__ value,
    int bh,
    int c,
    int C,
    int H,
    int W,
    float locX,
    float locY)
{
    const float x = locX * static_cast<float>(W) - 0.5f;
    const float y = locY * static_cast<float>(H) - 0.5f;

    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float wx1 = x - static_cast<float>(x0);
    const float wy1 = y - static_cast<float>(y0);
    const float wx0 = 1.0f - wx1;
    const float wy0 = 1.0f - wy1;

    float acc = 0.0f;
    const int base = ((bh * C + c) * H) * W;

    if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) {
        acc += loadValue(value, base + y0 * W + x0) * wx0 * wy0;
    }
    if (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H) {
        acc += loadValue(value, base + y0 * W + x1) * wx1 * wy0;
    }
    if (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H) {
        acc += loadValue(value, base + y1 * W + x0) * wx0 * wy1;
    }
    if (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H) {
        acc += loadValue(value, base + y1 * W + x1) * wx1 * wy1;
    }

    return acc;
}

template <typename T>
__global__ void msDeformAttnLocationKernel(
    const T* __restrict__ value0,
    const T* __restrict__ value1,
    const T* __restrict__ value2,
    const T* __restrict__ locations,
    const T* __restrict__ weights,
    T* __restrict__ output,
    int B,
    int C,
    int Q,
    int H0,
    int W0,
    int H1,
    int W1,
    int H2,
    int W2,
    int numHeads)
{
    __shared__ float sLocX[12];
    __shared__ float sLocY[12];
    __shared__ float sWeight[12];

    const int block = blockIdx.x;
    const int q = block % Q;
    const int bh = block / Q;
    if (bh >= B * numHeads) return;

    const int b = bh / numHeads;
    const int head = bh - b * numHeads;
    constexpr int totalPoints = 12;

    if (threadIdx.x < totalPoints) {
        const int p = threadIdx.x;
        const int locIdx = ((((b * Q + q) * numHeads + head) * totalPoints + p) * 2);
        sLocX[p] = loadValue(locations, locIdx);
        sLocY[p] = loadValue(locations, locIdx + 1);
        sWeight[p] = loadValue(weights, (((b * Q + q) * numHeads + head) * totalPoints + p));
    }
    __syncthreads();

    const int c = threadIdx.x;
    if (c >= C) return;

    float acc = 0.0f;
    #pragma unroll
    for (int p = 0; p < 3; ++p) {
        acc += sWeight[p] * bilinearUnitSample(value0, bh, c, C, H0, W0, sLocX[p], sLocY[p]);
    }
    #pragma unroll
    for (int p = 3; p < 9; ++p) {
        acc += sWeight[p] * bilinearUnitSample(value1, bh, c, C, H1, W1, sLocX[p], sLocY[p]);
    }
    #pragma unroll
    for (int p = 9; p < 12; ++p) {
        acc += sWeight[p] * bilinearUnitSample(value2, bh, c, C, H2, W2, sLocX[p], sLocY[p]);
    }

    output[((b * Q + q) * numHeads + head) * C + c] = storeValue<T>(acc);
}

template <typename T>
__device__ __forceinline__ float bilinearFlatSample(
    const T* __restrict__ value,
    int b,
    int head,
    int c,
    int H,
    int W,
    int levelStart,
    int numHeads,
    int C,
    float locX,
    float locY)
{
    const float x = locX * static_cast<float>(W) - 0.5f;
    const float y = locY * static_cast<float>(H) - 0.5f;

    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float wx1 = x - static_cast<float>(x0);
    const float wy1 = y - static_cast<float>(y0);
    const float wx0 = 1.0f - wx1;
    const float wy0 = 1.0f - wy1;

    float acc = 0.0f;
    constexpr int spatialSize = 8400;
    const int batchBase = b * spatialSize * numHeads * C;
    const int hc = head * C + c;

    if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) {
        const int spatial = levelStart + y0 * W + x0;
        acc += loadValue(value, batchBase + spatial * numHeads * C + hc) * wx0 * wy0;
    }
    if (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H) {
        const int spatial = levelStart + y0 * W + x1;
        acc += loadValue(value, batchBase + spatial * numHeads * C + hc) * wx1 * wy0;
    }
    if (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H) {
        const int spatial = levelStart + y1 * W + x0;
        acc += loadValue(value, batchBase + spatial * numHeads * C + hc) * wx0 * wy1;
    }
    if (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H) {
        const int spatial = levelStart + y1 * W + x1;
        acc += loadValue(value, batchBase + spatial * numHeads * C + hc) * wx1 * wy1;
    }

    return acc;
}

template <typename T>
__global__ void msDeformAttnFlatKernel(
    const T* __restrict__ value,
    const T* __restrict__ locations,
    const T* __restrict__ weights,
    T* __restrict__ output,
    int B,
    int C,
    int Q,
    int numHeads)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * Q * numHeads * C;
    if (idx >= total) return;

    int tmp = idx;
    const int c = tmp % C;
    tmp /= C;
    const int head = tmp % numHeads;
    tmp /= numHeads;
    const int q = tmp % Q;
    tmp /= Q;
    const int b = tmp;

    constexpr int totalPoints = 12;
    float acc = 0.0f;

    #pragma unroll
    for (int p = 0; p < 3; ++p) {
        const int locIdx = ((((b * Q + q) * numHeads + head) * totalPoints + p) * 2);
        const float locX = loadValue(locations, locIdx);
        const float locY = loadValue(locations, locIdx + 1);
        const float weight = loadValue(weights, (((b * Q + q) * numHeads + head) * totalPoints + p));
        acc += weight * bilinearFlatSample(value, b, head, c, 80, 80, 0, numHeads, C, locX, locY);
    }
    #pragma unroll
    for (int p = 3; p < 9; ++p) {
        const int locIdx = ((((b * Q + q) * numHeads + head) * totalPoints + p) * 2);
        const float locX = loadValue(locations, locIdx);
        const float locY = loadValue(locations, locIdx + 1);
        const float weight = loadValue(weights, (((b * Q + q) * numHeads + head) * totalPoints + p));
        acc += weight * bilinearFlatSample(value, b, head, c, 40, 40, 6400, numHeads, C, locX, locY);
    }
    #pragma unroll
    for (int p = 9; p < 12; ++p) {
        const int locIdx = ((((b * Q + q) * numHeads + head) * totalPoints + p) * 2);
        const float locX = loadValue(locations, locIdx);
        const float locY = loadValue(locations, locIdx + 1);
        const float weight = loadValue(weights, (((b * Q + q) * numHeads + head) * totalPoints + p));
        acc += weight * bilinearFlatSample(value, b, head, c, 20, 20, 8000, numHeads, C, locX, locY);
    }

    output[idx] = storeValue<T>(acc);
}

template <typename TValue, typename TGrid, typename TWeight>
__global__ void msDeformAttnElementKernel(
    const TValue* __restrict__ value0,
    const TValue* __restrict__ value1,
    const TValue* __restrict__ value2,
    const TGrid* __restrict__ grid0,
    const TGrid* __restrict__ grid1,
    const TGrid* __restrict__ grid2,
    const TWeight* __restrict__ weights,
    TValue* __restrict__ output,
    int BH,
    int C,
    int Q,
    int H0,
    int W0,
    int H1,
    int W1,
    int H2,
    int W2,
    int numHeads)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = (BH / numHeads) * Q * numHeads * C;
    if (idx >= total) return;

    const int p0 = 3;
    const int p1 = 6;
    const int p2 = 3;
    const int totalPoints = p0 + p1 + p2;

    int tmp = idx;
    const int c = tmp % C;
    tmp /= C;
    const int head = tmp % numHeads;
    tmp /= numHeads;
    const int q = tmp % Q;
    tmp /= Q;
    const int b = tmp;
    const int bh = b * numHeads + head;

    float acc = 0.0f;

    #pragma unroll
    for (int p = 0; p < 3; ++p) {
        const int gridIdx = ((bh * Q + q) * p0 + p) * 2;
        const float gx = loadValue(grid0, gridIdx);
        const float gy = loadValue(grid0, gridIdx + 1);
        const float weight = loadValue(weights, (((b * Q + q) * numHeads + head) * totalPoints + p));
        acc += weight * bilinearGridSample(value0, bh, c, C, H0, W0, gx, gy);
    }
    #pragma unroll
    for (int p = 0; p < 6; ++p) {
        const int point = p0 + p;
        const int gridIdx = ((bh * Q + q) * p1 + p) * 2;
        const float gx = loadValue(grid1, gridIdx);
        const float gy = loadValue(grid1, gridIdx + 1);
        const float weight = loadValue(weights, (((b * Q + q) * numHeads + head) * totalPoints + point));
        acc += weight * bilinearGridSample(value1, bh, c, C, H1, W1, gx, gy);
    }
    #pragma unroll
    for (int p = 0; p < 3; ++p) {
        const int point = p0 + p1 + p;
        const int gridIdx = ((bh * Q + q) * p2 + p) * 2;
        const float gx = loadValue(grid2, gridIdx);
        const float gy = loadValue(grid2, gridIdx + 1);
        const float weight = loadValue(weights, (((b * Q + q) * numHeads + head) * totalPoints + point));
        acc += weight * bilinearGridSample(value2, bh, c, C, H2, W2, gx, gy);
    }

    output[idx] = storeValue<TValue>(acc);
}

template <typename T>
void launchMSDeformAttnFromLocationsTyped(
    const void* value0,
    const void* value1,
    const void* value2,
    const void* locations,
    const void* weights,
    void* output,
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
    cudaStream_t stream)
{
    dim3 grid(B * numHeads * Q, 1, 1);
    dim3 block(64, 1, 1);
    msDeformAttnLocationKernel<T><<<grid, block, 0, stream>>>(
        static_cast<const T*>(value0),
        static_cast<const T*>(value1),
        static_cast<const T*>(value2),
        static_cast<const T*>(locations),
        static_cast<const T*>(weights),
        static_cast<T*>(output),
        B, C, Q, H0, W0, H1, W1, H2, W2, numHeads);
}

void launchMSDeformAttnFromLocations(
    const void* value0,
    const void* value1,
    const void* value2,
    const void* locations,
    const void* weights,
    void* output,
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
    nvinfer1::DataType dtype,
    cudaStream_t stream)
{
    if (dtype == nvinfer1::DataType::kHALF) {
        launchMSDeformAttnFromLocationsTyped<half>(
            value0, value1, value2, locations, weights, output,
            B, C, Q, H0, W0, H1, W1, H2, W2, numHeads, stream);
    } else {
        launchMSDeformAttnFromLocationsTyped<float>(
            value0, value1, value2, locations, weights, output,
            B, C, Q, H0, W0, H1, W1, H2, W2, numHeads, stream);
    }
}

template <typename T>
void launchMSDeformAttnFlatTyped(
    const void* value,
    const void* locations,
    const void* weights,
    void* output,
    int B,
    int C,
    int Q,
    int numHeads,
    cudaStream_t stream)
{
    const int total = B * Q * numHeads * C;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    msDeformAttnFlatKernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(value),
        static_cast<const T*>(locations),
        static_cast<const T*>(weights),
        static_cast<T*>(output),
        B,
        C,
        Q,
        numHeads);
}

void launchMSDeformAttnFlat(
    const void* value,
    const void* locations,
    const void* weights,
    void* output,
    int B,
    int C,
    int Q,
    int numHeads,
    nvinfer1::DataType dtype,
    cudaStream_t stream)
{
    if (dtype == nvinfer1::DataType::kHALF) {
        launchMSDeformAttnFlatTyped<half>(value, locations, weights, output, B, C, Q, numHeads, stream);
    } else {
        launchMSDeformAttnFlatTyped<float>(value, locations, weights, output, B, C, Q, numHeads, stream);
    }
}

template <typename TValue, typename TGrid, typename TWeight>
void launchMSDeformAttnTyped(
    const void* value0,
    const void* value1,
    const void* value2,
    const void* grid0,
    const void* grid1,
    const void* grid2,
    const void* weights,
    void* output,
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
    cudaStream_t stream)
{
    const int total = (BH / numHeads) * Q * numHeads * C;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    msDeformAttnElementKernel<TValue, TGrid, TWeight><<<blocks, threads, 0, stream>>>(
        static_cast<const TValue*>(value0),
        static_cast<const TValue*>(value1),
        static_cast<const TValue*>(value2),
        static_cast<const TGrid*>(grid0),
        static_cast<const TGrid*>(grid1),
        static_cast<const TGrid*>(grid2),
        static_cast<const TWeight*>(weights),
        static_cast<TValue*>(output),
        BH, C, Q, H0, W0, H1, W1, H2, W2, numHeads);
}

void launchMSDeformAttn(
    const void* value0,
    const void* value1,
    const void* value2,
    const void* grid0,
    const void* grid1,
    const void* grid2,
    const void* weights,
    void* output,
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
    nvinfer1::DataType valueDtype,
    nvinfer1::DataType gridDtype,
    nvinfer1::DataType weightDtype,
    cudaStream_t stream)
{
    if (valueDtype == nvinfer1::DataType::kHALF) {
        if (gridDtype == nvinfer1::DataType::kHALF && weightDtype == nvinfer1::DataType::kHALF) {
            launchMSDeformAttnTyped<half, half, half>(
                value0, value1, value2, grid0, grid1, grid2, weights, output,
                BH, C, Q, H0, W0, H1, W1, H2, W2, numHeads, stream);
        } else if (gridDtype == nvinfer1::DataType::kFLOAT && weightDtype == nvinfer1::DataType::kHALF) {
            launchMSDeformAttnTyped<half, float, half>(
                value0, value1, value2, grid0, grid1, grid2, weights, output,
                BH, C, Q, H0, W0, H1, W1, H2, W2, numHeads, stream);
        } else if (gridDtype == nvinfer1::DataType::kHALF && weightDtype == nvinfer1::DataType::kFLOAT) {
            launchMSDeformAttnTyped<half, half, float>(
                value0, value1, value2, grid0, grid1, grid2, weights, output,
                BH, C, Q, H0, W0, H1, W1, H2, W2, numHeads, stream);
        } else {
            launchMSDeformAttnTyped<half, float, float>(
                value0, value1, value2, grid0, grid1, grid2, weights, output,
                BH, C, Q, H0, W0, H1, W1, H2, W2, numHeads, stream);
        }
    } else {
        if (gridDtype == nvinfer1::DataType::kHALF && weightDtype == nvinfer1::DataType::kHALF) {
            launchMSDeformAttnTyped<float, half, half>(
                value0, value1, value2, grid0, grid1, grid2, weights, output,
                BH, C, Q, H0, W0, H1, W1, H2, W2, numHeads, stream);
        } else if (gridDtype == nvinfer1::DataType::kFLOAT && weightDtype == nvinfer1::DataType::kHALF) {
            launchMSDeformAttnTyped<float, float, half>(
                value0, value1, value2, grid0, grid1, grid2, weights, output,
                BH, C, Q, H0, W0, H1, W1, H2, W2, numHeads, stream);
        } else if (gridDtype == nvinfer1::DataType::kHALF && weightDtype == nvinfer1::DataType::kFLOAT) {
            launchMSDeformAttnTyped<float, half, float>(
                value0, value1, value2, grid0, grid1, grid2, weights, output,
                BH, C, Q, H0, W0, H1, W1, H2, W2, numHeads, stream);
        } else {
            launchMSDeformAttnTyped<float, float, float>(
                value0, value1, value2, grid0, grid1, grid2, weights, output,
                BH, C, Q, H0, W0, H1, W1, H2, W2, numHeads, stream);
        }
    }
}

} // namespace liteany
