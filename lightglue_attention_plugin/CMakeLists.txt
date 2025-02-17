cmake_minimum_required(VERSION 3.5.0)
project(lightglue_attention_plugin LANGUAGES CXX)   # CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(CUDA)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -w --use_fast_math")

if(POLICY CMP0146) 
    cmake_policy(SET CMP0146 OLD) 
endif()
if(NOT CUDA_INCLUDE_DIRS)
    find_package(CUDA REQUIRED)
endif()
message(STATUS "CUDA_LIBRARIES        : ${CUDA_LIBRARIES}")
message(STATUS "CUDA_cublas_LIBRARY   : ${CUDA_cublas_LIBRARY}")
message(STATUS "CUDA_cusolver_LIBRARY : ${CUDA_cusolver_LIBRARY}")
message(STATUS "CUDA_INCLUDE_DIRS     : ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_CUDART_LIBRARY   : ${CUDA_CUDART_LIBRARY}")

# Find dependencies
# To get LIBTORCH_DIR in python3 console : 
# >> import torch 
# >> print(torch.utils.cmake_prefix_path)
set(LIBTORCH_DIR "/usr/local/lib/python3.8/dist-packages/torch/share/cmake")
# set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${LIBTORCH_DIR}")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
find_package(Torch REQUIRED)
set(TORCH_PYTHON_LIBRARY "/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_python.so")
message(STATUS "CMAKE_PREFIX_PATH    : ${CMAKE_PREFIX_PATH}")
message(STATUS "TORCH_LIBRARIES      : ${TORCH_LIBRARIES}")
message(STATUS "TORCH_PYTHON_LIBRARY : ${TORCH_PYTHON_LIBRARY}")
message(STATUS "TORCH_INCLUDE_DIRS   : ${TORCH_INCLUDE_DIRS}")

# set(CUTLASS_DIRS "../third_party/cutlass-v3.4.1/include")
# message(STATUS "CUTLASS_DIRS   : ${CUTLASS_DIRS}")

get_filename_component(CUTLASS_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass-v3.4.1/include" ABSOLUTE)
message(STATUS "CUTLASS_DIRS : ${CUTLASS_DIRS}")

set(
    TRT_INCLUDE_DIRS
    "/usr/local/cuda/include/"
)
# set(
#     TRT_LIBRARIES 
#     "/usr/lib/aarch-linux-gnu/libnvinfer.so"
#     "/usr/lib/aarch-linux-gnu/libnvinfer_plugin.so"
#     "/usr/lib/aarch-linux-gnu/libnvonnxparser.so"
#     "/usr/lib/aarch-linux-gnu/libnvparsers.so"
# )   # maybe more

# for tensorrt libs
link_directories(/usr/lib/aarch-linux-gnu/)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${CUTLASS_DIRS}
    ${TRT_INCLUDE_DIRS}
)

add_library(
    lightglue_attention_plugin 
    SHARED 
    attention_headdim_64_fp16in_fp16out.cu 
    attention_headdim_64_fp16in_fp32out.cu 
    lightglue_attention_plugin.cpp
    common/checkMacrosPlugin.cpp
)
target_link_libraries(
    lightglue_attention_plugin 
    PUBLIC 
    # ${TORCH_LIBRARIES} 
    # ${TORCH_PYTHON_LIBRARY} 
    ${CUDA_LIBRARIES} 
    ${CUDA_cublas_LIBRARY} 
    ${CUDA_cusolver_LIBRARY}
    # ${TRT_LIBRARIES}
    nvinfer
    nvinfer_plugin
    # nvonnxparser
    # nvparsers
)
set_target_properties(lightglue_attention_plugin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
