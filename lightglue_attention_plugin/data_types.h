//
// Created by https://github.com/qdLMF on 25-02-16.
//

#ifndef ATTENTION_HEADDIM_64_DATA_TYPES_H
#define ATTENTION_HEADDIM_64_DATA_TYPES_H

#include <cuda_fp16.h>

struct halfx4 {
    __half x;
    __half y;
    __half z;
    __half w;
};

struct halfx8 {
    __half d_0;
    __half d_1;
    __half d_2;
    __half d_3;
    __half d_4;
    __half d_5;
    __half d_6;
    __half d_7;
};

using FP16   = half;
using FP16x4 = halfx4;
using FP16x8 = halfx8;
using FP32   = float;
using FP32x2 = float2;
using FP32x4 = float4;

#endif

