#pragma once

#include <cuda_runtime.h>

#include <cstdio>

namespace whirlwind {

// Reports the pending launch error, if any, and returns it so the caller can
// propagate failure up to enqueue(). Call this right after every kernel
// launch: a bad launch configuration is otherwise invisible until it
// resurfaces as unrelated corruption further down the stream.
//
// `where` is normally __func__ of the launch wrapper.
inline cudaError_t lastLaunchError(const char* where) noexcept
{
    const cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[whirlwind] %s: CUDA launch failed: %s\n",
                     where, cudaGetErrorString(err));
    }
    return err;
}

} // namespace whirlwind
