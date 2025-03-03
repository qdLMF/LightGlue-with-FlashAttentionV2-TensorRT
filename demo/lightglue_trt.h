//
// Created by https://github.com/qdLMF on 25-02-16.
//

#ifndef LIGHTGLUE_TRT_H
#define LIGHTGLUE_TRT_H

#include <cstdlib>
#include <string>
#include <memory>
#include <fstream>
#include <cassert>
#include <vector>
#include <unordered_map>
#include <tuple>

#include <torch/torch.h>

#include <cuda_runtime_api.h>

#include <NvInfer.h>

#include "3rdparty/tensorrtbuffer/include/logger.h"

#include "./buffer_trt.h"
#include "./utils_trt.h"


class LightGlueTRT {
public : 
    explicit LightGlueTRT(
        const std::string& trt_engine_file_path, 
        const std::string& lightglue_plugin_file_path, 
        tensorrt_log::Logger& gLogger
    );

    ~LightGlueTRT();

    void SetMaxInputShape(const std::unordered_map<std::string, std::vector<int64_t>>& max_input_shape);
    void SetInputShape(const std::unordered_map<std::string, std::vector<int64_t>>& input_shape);
    void SetInputAddress();
    void CopyInputTensor(const std::unordered_map<std::string, torch::Tensor>& input_tensor);
    void Forward();
    void Forward(const cudaStream_t& cuda_stream);
    std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> PostProcess(float threshold);
    void RecordCUDAGraph();
    void LaunchCUDAGraph();
    void Sync();

private : 
    std::string m_trt_engine_file_path;

    std::shared_ptr<nvinfer1::IRuntime> m_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    std::map<std::string, int> m_bindings_name_to_index;

    BufferTRT m_buffer;

    std::unordered_map<std::string, std::vector<int64_t>> m_max_input_shape;
    std::unordered_map<std::string, std::vector<int64_t>> m_max_output_shape;
    std::unordered_map<std::string, std::vector<int64_t>> m_input_shape;
    std::unordered_map<std::string, std::vector<int64_t>> m_output_shape;

private : 
    void* m_lightglue_plugin_handle = nullptr;

private : 
    torch::Tensor m_scores_fp32_cuda;
    torch::Tensor m_descriptors_0_fp32_cuda;
    torch::Tensor m_descriptors_1_fp32_cuda;
    std::tuple<
        torch::Tensor, 
        torch::Tensor
    > m_max_scores_0_tuple;
    std::tuple<
        torch::Tensor, 
        torch::Tensor
    > m_max_scores_1_tuple;
    torch::Tensor m_max_scores_values_0_fp32_cuda;
    torch::Tensor m_max_scores_indices_0_int64_cuda;
    torch::Tensor m_max_scores_values_1_fp32_cuda;
    torch::Tensor m_max_scores_indices_1_int64_cuda;
    torch::Tensor m_linear_indices_0_int64_cuda;
    torch::Tensor m_temp_int64_cuda;
    torch::Tensor m_mutual_max_scores_flag_bool_cuda;
    torch::Tensor m_zeros_fp32_cuda;
    torch::Tensor m_mutual_max_scores_fp32_cuda;
    torch::Tensor m_mutual_max_scores_mask_bool_cuda;
    torch::Tensor m_match_indices_0_int64_cuda;
    torch::Tensor m_match_indices_1_int64_cuda;
    torch::Tensor m_match_indices_int64_cuda;
    torch::Tensor m_match_scores_fp32_cuda;

public : 
    cudaStream_t cuda_stream;
    cudaGraph_t cuda_graph;
    cudaGraphExec_t cuda_graph_exec;
};

#endif