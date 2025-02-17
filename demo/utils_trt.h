//
// Created by https://github.com/qdLMF on 25-02-16.
//

#ifndef UTILS_TRT_H
#define UTILS_TRT_H

#include <cstdlib>
#include <vector>

#include <cuda_runtime_api.h>

#include <NvInfer.h>


nvinfer1::Dims IntVectorToDims(const std::vector<int64_t>& dims_vec);

std::vector<int64_t> DimsToIntVector(const nvinfer1::Dims& dims);

#endif