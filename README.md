# whirlwind_trt_plugins

TensorRT plugin and CUDA kernels for whIRLwind inference pipelines.
The build produces a shared plugin library:

```bash
build/libww_trt_plugins.so
```

## Requirements

- Linux with TensorRT headers/libraries installed under `/usr/include/x86_64-linux-gnu` and `/usr/lib/x86_64-linux-gnu`
- CUDA toolkit available at `/usr/local/cuda`
- CMake and a C++17 compiler

## Build

```bash
pixi run build
```

The Pixi task intentionally uses the host compiler and CUDA toolkit (`/usr/bin/c++`
and `/usr/local/cuda/bin/nvcc`) so it matches the manual CMake build.
