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

# ---------------------------------------------------------------------------
# TensorRT. No upstream CMake package config ships with the .deb layout, so
# locate the pieces we need and wrap them in an imported target — that keeps
# the dependency on the target instead of a global link_directories().
# ---------------------------------------------------------------------------
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/include)
find_library(TENSORRT_NVINFER_LIBRARY nvinfer
    HINTS /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TENSORRT_INCLUDE_DIR TENSORRT_NVINFER_LIBRARY)

add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
set_target_properties(TensorRT::nvinfer PROPERTIES
    IMPORTED_LOCATION "${TENSORRT_NVINFER_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
)

# ---------------------------------------------------------------------------
# Plugin library. One directory per plugin under src/plugins; shared
# boilerplate in src/common; static registration in src/register_plugins.cpp.
# ---------------------------------------------------------------------------
add_library(ww_trt_plugins SHARED
    src/register_plugins.cpp

    src/plugins/corr_volume/corr_volume.cpp
    src/plugins/corr_volume/corr_volume_kernel.cu

    src/plugins/cost_stem_3d/cost_stem_3d.cpp
    src/plugins/cost_stem_3d/cost_stem_3d_kernel.cu

    src/plugins/ms_deform_attn/ms_deform_attn.cpp
    src/plugins/ms_deform_attn/ms_deform_attn_kernel.cu

    src/plugins/nv12_decode/nv12_decode.cpp
    src/plugins/nv12_decode/nv12_decode_kernel.cu
)

# Everything here is internal: includes are written relative to src/ so they
# read as "common/..." / "plugins/<name>/...". Nothing is exported.
target_include_directories(ww_trt_plugins PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(ww_trt_plugins
    PRIVATE
        TensorRT::nvinfer
        CUDA::cudart
)

set_target_properties(ww_trt_plugins PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

include(GNUInstallDirs)
install(TARGETS ww_trt_plugins LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
