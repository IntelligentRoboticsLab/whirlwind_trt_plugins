cmake_minimum_required(VERSION 3.18)
project(liteany_trt_plugins LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build for Turing (dev box) and Orin AGX (sm_87) by default.
# Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;87")
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

add_library(corr_volume_plugin SHARED
    src/corr_volume_plugin.cpp
    src/corr_volume_kernel.cu
    src/nv12_decode_plugin.cpp
    src/nv12_decode_kernel.cu
    src/plugin_register.cpp
)

target_link_libraries(corr_volume_plugin
    nvinfer
    cudart
)

set_target_properties(corr_volume_plugin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
