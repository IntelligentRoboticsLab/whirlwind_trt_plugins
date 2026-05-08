cmake_minimum_required(VERSION 3.18)
project(ww_trt_plugins LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build for Turing, Orin AGX (sm_87), and Ada/RTX 4090 (sm_89) by default.
# Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;87;89")
endif()

find_package(CUDAToolkit REQUIRED)

set(TENSORRT_INCLUDE_DIR /usr/include/x86_64-linux-gnu)
set(TENSORRT_LIB_DIR /usr/lib/x86_64-linux-gnu)

include_directories(
    include
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

link_directories(${TENSORRT_LIB_DIR})

add_library(ww_trt_plugins SHARED
    src/corr_volume_plugin.cpp
    src/corr_volume_kernel.cu
    src/cost_stem_3d_plugin.cpp
    src/cost_stem_3d_kernel.cu
    src/ms_deform_attn_plugin.cpp
    src/ms_deform_attn_kernel.cu
    src/nv12_decode_plugin.cpp
    src/nv12_decode_kernel.cu
    src/plugin_register.cpp
)

target_link_libraries(ww_trt_plugins
    nvinfer
    cudart
)

set_target_properties(ww_trt_plugins PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
