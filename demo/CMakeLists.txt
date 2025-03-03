cmake_minimum_required(VERSION 3.5)
project(infer_with_trt_engine_superpoint)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

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
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
find_package(Torch REQUIRED)
message(STATUS "CMAKE_PREFIX_PATH  : ${CMAKE_PREFIX_PATH}")
message(STATUS "TORCH_LIBRARIES    : ${TORCH_LIBRARIES}")
message(STATUS "TORCH_INCLUDE_DIRS : ${TORCH_INCLUDE_DIRS}")

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

set(TRT_LINK_DIRS "/usr/lib/aarch-linux-gnu/")
link_directories(${TRT_LINK_DIRS})
message(STATUS "TRT_LINK_DIRS : ${TRT_LINK_DIRS}")

get_filename_component(ATTENTION_PLUGIN_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../lightglue_attention_plugin/build" ABSOLUTE)
link_directories(${ATTENTION_PLUGIN_DIRS})
message(STATUS "ATTENTION_PLUGIN_DIRS : ${ATTENTION_PLUGIN_DIRS}")

message(STATUS "CMAKE_DL_LIBS : ${CMAKE_DL_LIBS}") # for dlopen(), dlclose()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TRT_INCLUDE_DIRS}
)

add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/tensorrtbuffer) # lib's name is tensorrtbuffer

add_library(
    superpoint_and_lightglue_with_trt 
    SHARED 
    superpoint_mono_trt.cpp
    lightglue_trt.cpp
    buffer_trt.cpp
    utils_trt.cpp
)
target_link_libraries(
    superpoint_and_lightglue_with_trt 
    ${CMAKE_DL_LIBS}
    ${OpenCV_LIBRARIES} 
    ${TORCH_LIBRARIES} 
    ${CUDA_LIBRARIES} 
    nvinfer
    nvinfer_plugin
    # nvonnxparser
    # nvparsers
    tensorrtbuffer
    lightglue_attention_plugin
)

add_executable(
    demo_mono
    demo_mono.cpp
)
target_link_libraries(
    demo_mono
    ${OpenCV_LIBRARIES} 
    ${TORCH_LIBRARIES} 
    ${CUDA_LIBRARIES} 
    tensorrtbuffer
    superpoint_and_lightglue_with_trt
)

