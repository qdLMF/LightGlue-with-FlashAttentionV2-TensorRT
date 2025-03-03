//
// Created by https://github.com/qdLMF on 25-02-16.
//

#ifndef SUPERPOINT_MONO_TRT_H
#define SUPERPOINT_MONO_TRT_H

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


class SuperPointMonoTRT {
public : 
    explicit SuperPointMonoTRT(
        const std::string& trt_engine_file_path,
        tensorrt_log::Logger& gLogger
    );
    
    ~SuperPointMonoTRT();

    void SetMaxInputShape(const std::unordered_map<std::string, std::vector<int64_t>>& max_input_shape);
    void SetInputShape(const std::unordered_map<std::string, std::vector<int64_t>>& input_shape);
    void SetInputAddress();
    void CopyInputTensor(const std::unordered_map<std::string, torch::Tensor>& input_tensor);
    void Forward();
    void Forward(const cudaStream_t& cuda_stream);
    std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor> PostProcess(float threshold, int max_num_keypoints);
    void RecordCUDAGraph();
    void LaunchCUDAGraph();
    void Sync();

public : 
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

public : 
    int m_k;
    float m_image_rows;
    float m_image_cols;
    float m_scale;
    torch::Tensor m_scores_fp32_cuda;
    torch::Tensor m_descriptors_fp32_cuda;
    torch::Tensor m_mask_gt_threshold_bool_cuda;
    torch::Tensor m_keypoints_gt_threshold_int32_cuda;
    torch::Tensor m_scores_gt_threshold_fp32_cuda;
    std::tuple<
        torch::Tensor, 
        torch::Tensor
    > m_scores_indices_topk_cuda;
    torch::Tensor m_scores_topk_fp32_cuda;
    torch::Tensor m_indices_topk_int32_cuda;
    torch::Tensor m_keypoints_topk_int32_cuda;
    torch::Tensor m_divider;
    torch::Tensor m_keypoints_topk_fp32_cuda;
    torch::Tensor m_descriptors_topk_fp32_cuda;
    torch::Tensor m_shift;
    torch::Tensor m_keypoints_topk_normalized_fp32_cuda;

public : 
    cudaStream_t cuda_stream;
    cudaGraph_t cuda_graph;
    cudaGraphExec_t cuda_graph_exec;
};

#endif