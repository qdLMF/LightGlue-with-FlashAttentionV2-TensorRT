//
// Created by https://github.com/qdLMF on 25-02-16.
//

#ifndef ATTENTION_HEADDIM_64_FP16IN_FP32OUT_CUH
#define ATTENTION_HEADDIM_64_FP16IN_FP32OUT_CUH

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <torch/torch.h>

#include "./data_types.h"


namespace AttentionHeadDim64 {

void launch_QKV_convert_from_fp32_to_fp16(
    const FP32* Q_ptr_fp32, 
    const FP32* K_ptr_fp32, 
    const FP32* V_ptr_fp32, 
    FP16* Q_ptr_fp16, 
    FP16* K_ptr_fp16, 
    FP16* V_ptr_fp16, 
    int batch, 
    int num_heads, 
    int num_Q_real, 
    int num_KV_real, 
    int num_Q_padded, 
    int num_KV_padded, 
    cudaStream_t stream
);


// slower, don't use it.
void launch_QKV_convert_from_fp32_to_fp16_unused(
    const FP32* Q_ptr_fp32, 
    const FP32* K_ptr_fp32, 
    const FP32* V_ptr_fp32, 
    FP16* Q_ptr_fp16, 
    FP16* K_ptr_fp16, 
    FP16* V_ptr_fp16, 
    int batch, 
    int num_heads, 
    int num_Q_real, 
    int num_KV_real, 
    int num_Q_padded, 
    int num_KV_padded, 
    cudaStream_t stream
);


void launch_O_move_from_padded_to_unpadded_fp32(
    const FP32* O_ptr_src, 
    FP32* O_ptr_dst, 
    int batch, 
    int num_heads, 
    int num_O_real, 
    int num_O_padded, 
    cudaStream_t stream
);


void launch_attention_kernel_headdim_64_no_remainder_fp16in_fp32out(
    const FP16* Q_ptr, 
    const FP16* K_ptr, 
    const FP16* V_ptr, 
    FP32* O_ptr, 
    int batch, 
    int num_heads, 
    int num_Q_padded, 
    int num_KV_padded, 
    int num_Q_real, 
    int num_KV_real, 
    cudaStream_t stream
);

}   // namespace AttentionHeadDim64

#endif