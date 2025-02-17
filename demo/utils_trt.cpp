//
// Created by https://github.com/qdLMF on 25-02-16.
//

#include "./utils_trt.h"


nvinfer1::Dims IntVectorToDims(const std::vector<int64_t>& dims_vec) {
    nvinfer1::Dims dims;
    dims.nbDims = dims_vec.size();

    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = dims_vec[i];
    }

    return dims;
}

std::vector<int64_t> DimsToIntVector(const nvinfer1::Dims& dims) {
    std::vector<int64_t> dims_vec;
    dims_vec.resize(dims.nbDims);

    for (int i = 0; i < dims.nbDims; ++i) {
        dims_vec[i] = dims.d[i];
    }

    return dims_vec;
}