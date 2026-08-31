#pragma once

#include <NvInferRuntime.h>

#include <cuda_fp16.h>

namespace whirlwind {

// Carries a type through a generic lambda without needing to construct a
// value of it (__half is awkward to default-construct on the host).
template <typename T>
struct TypeTag {
    using type = T;
};

// Turns a runtime nvinfer1::DataType into a compile-time type, so one generic
// lambda body covers every element type a plugin accepts:
//
//     dispatchFloatType(desc.type, [&](auto tag) {
//         using T = typename decltype(tag)::type;
//         err = launchFoo<T>(...);
//     });
//
// Returns false without invoking `fn` for anything other than fp16/fp32 —
// supportsFormatCombination should already have excluded those, so a false
// return means the plugin and its kernels disagree about what is supported.
// Nests for kernels whose inputs are independently typed.
template <typename Fn>
bool dispatchFloatType(nvinfer1::DataType dt, Fn&& fn)
{
    switch (dt) {
        case nvinfer1::DataType::kHALF:
            fn(TypeTag<__half>{});
            return true;
        case nvinfer1::DataType::kFLOAT:
            fn(TypeTag<float>{});
            return true;
        default:
            return false;
    }
}

} // namespace whirlwind
