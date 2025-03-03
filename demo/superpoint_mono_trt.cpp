//
// Created by https://github.com/qdLMF on 25-02-16.
//

#include "./superpoint_mono_trt.h"


// #define SUPERPOINT_MAX_NUM_KEYPOINTS  1024
// #define SUPERPOINT_KEYPOINT_THRESHOLD 0.00005f

SuperPointMonoTRT::SuperPointMonoTRT(
    const std::string& trt_engine_file_path,
    tensorrt_log::Logger& gLogger
) : m_trt_engine_file_path(trt_engine_file_path) {
    std::ifstream file_stream(trt_engine_file_path.c_str(), std::ios::binary);
    if (!file_stream.is_open()) {
        std::cerr << "SuperPoint : Failed to open engine file!" << std::endl;
        return;
    }
    file_stream.seekg(0, std::ifstream::end);
    size_t engine_size = file_stream.tellg();
    file_stream.seekg(0, std::ifstream::beg);
    char* engine_stream = new char[engine_size];
    file_stream.read(engine_stream, engine_size);
    file_stream.close();

    m_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    if (m_runtime == nullptr) {
        std::cerr << "SuperPoint : Failed to construct IRuntime!" << std::endl;
        delete [] engine_stream;
        return;
    }
    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(engine_stream, engine_size));
    if (m_engine == nullptr) {
        std::cerr << "SuperPoint : Failed to construct ICudaEngine!" << std::endl;
        delete [] engine_stream;
        return;
    }
    delete [] engine_stream;
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());

    m_bindings_name_to_index.clear();
    m_bindings_name_to_index["image"]       = m_engine->getBindingIndex("image");
    m_bindings_name_to_index["scores"]      = m_engine->getBindingIndex("scores");
    m_bindings_name_to_index["descriptors"] = m_engine->getBindingIndex("descriptors");

    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);
}

SuperPointMonoTRT::~SuperPointMonoTRT() {
    // cudaStreamSynchronize(cuda_stream);
    // cudaStreamDestroy(cuda_stream);
    // cudaGraphDestroy(cuda_graph);
    // cudaGraphExecDestroy(cuda_graph_exec);
}

void SuperPointMonoTRT::SetMaxInputShape(const std::unordered_map<std::string, std::vector<int64_t>>& max_input_shape) {
    assert(max_input_shape.find("image") != max_input_shape.end());
    assert(max_input_shape.at("image").size() == 4);
    assert(
        max_input_shape.at("image")[0] == 1    && 
        max_input_shape.at("image")[1] == 1    &&
        max_input_shape.at("image")[2] == 480  &&
        max_input_shape.at("image")[3] >= 640
    );

    m_max_input_shape = max_input_shape;
    m_context->setBindingDimensions(
        m_bindings_name_to_index.at("image"), 
        IntVectorToDims(m_max_input_shape.at("image"))
    );

    m_buffer.Set(
        m_engine,
        0,
        m_context.get()
    );

    m_max_output_shape["scores"]      = DimsToIntVector(m_context->getBindingDimensions(m_bindings_name_to_index.at("scores")));
    m_max_output_shape["descriptors"] = DimsToIntVector(m_context->getBindingDimensions(m_bindings_name_to_index.at("descriptors")));
}

void SuperPointMonoTRT::SetInputShape(const std::unordered_map<std::string, std::vector<int64_t>>& input_shape) {
    assert(input_shape.find("image") != input_shape.end());
    assert(input_shape.at("image").size() == 4);
    assert(
        input_shape.at("image")[0] == m_max_input_shape.at("image")[0]  && 
        input_shape.at("image")[1] == m_max_input_shape.at("image")[1]  &&
        input_shape.at("image")[2] <= m_max_input_shape.at("image")[2]  &&
        input_shape.at("image")[3] <= m_max_input_shape.at("image")[3]
    );

    m_input_shape = input_shape;
    m_context->setBindingDimensions(
        m_bindings_name_to_index.at("image"), 
        IntVectorToDims(m_input_shape.at("image"))
    );

    m_buffer.Set(
        m_engine,
        0,
        m_context.get()
    );

    m_output_shape["scores"]      = DimsToIntVector(m_context->getBindingDimensions(m_bindings_name_to_index.at("scores")));
    m_output_shape["descriptors"] = DimsToIntVector(m_context->getBindingDimensions(m_bindings_name_to_index.at("descriptors")));
}

void SuperPointMonoTRT::SetInputAddress() {
    m_context->setTensorAddress(
        "image", 
        m_buffer.GetDeviceBuffer(m_bindings_name_to_index.at("image"))
    );
    m_context->setTensorAddress(
        "scores", 
        m_buffer.GetDeviceBuffer(m_bindings_name_to_index.at("scores"))
    );
    m_context->setTensorAddress(
        "descriptors", 
        m_buffer.GetDeviceBuffer(m_bindings_name_to_index.at("descriptors"))
    );
}

void SuperPointMonoTRT::CopyInputTensor(const std::unordered_map<std::string, torch::Tensor>& input_tensor) {
    assert(input_tensor.find("image") != input_tensor.end());
    assert(input_tensor.at("image").sizes() == m_input_shape.at("image"));
    assert(input_tensor.at("image").dtype() == torch::kFloat32);

    m_buffer.CopyInputFromPtr(
        m_bindings_name_to_index.at("image"), 
        input_tensor.at("image").data_ptr(), 
        input_tensor.at("image").numel() * input_tensor.at("image").element_size(), 
        input_tensor.at("image").device() == torch::kCUDA ? true : false
    );
}

// void SuperPointMonoTRT::Forward() {
//     assert(m_context->executeV2(m_buffer.GetDeviceBindings().data()));  // using default stream
// }

// void SuperPointMonoTRT::Forward() {
//     assert(m_context->enqueueV2(m_buffer.GetDeviceBindings().data(), cuda_stream, nullptr));
// }

void SuperPointMonoTRT::Forward() {
    assert(m_context->enqueueV3(cuda_stream));  // have to call m_context->setTensorAddress() before
}

// void SuperPointMonoTRT::Forward(const cudaStream_t& cuda_stream) {
//     assert(m_context->enqueueV3(cuda_stream));  // have to call m_context->setTensorAddress() before
// }

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor> SuperPointMonoTRT::PostProcess(float threshold, int max_num_keypoints) {
    torch::NoGradGuard torch_no_grad;

    m_scores_fp32_cuda = torch::from_blob(
        m_buffer.GetDeviceBuffer(m_bindings_name_to_index.at("scores")), 
        m_output_shape.at("scores"), 
        torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA).requires_grad(false)
    ).squeeze(0);   // [480, 640]
    m_descriptors_fp32_cuda = torch::from_blob(
        m_buffer.GetDeviceBuffer(m_bindings_name_to_index.at("descriptors")), 
        m_output_shape.at("descriptors"), 
        torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA).requires_grad(false)
    ).squeeze(0);   // [256, 60, 80]

    m_scores_fp32_cuda.index_put_(
        {
            torch::indexing::Slice(torch::indexing::None, 4),
            torch::indexing::Slice()
        },
        -1.0f
    );
    m_scores_fp32_cuda.index_put_(
        {
            torch::indexing::Slice(),
            torch::indexing::Slice(torch::indexing::None, 4)
        },
        -1.0f
    );
    m_scores_fp32_cuda.index_put_(
        {
            torch::indexing::Slice(m_output_shape.at("scores")[1] - 4, torch::indexing::None),
            torch::indexing::Slice()
        },
        -1.0f
    );
    m_scores_fp32_cuda.index_put_(
        {
            torch::indexing::Slice(),
            torch::indexing::Slice(m_output_shape.at("scores")[2] - 4, torch::indexing::None)
        },
        -1.0f
    );

    m_mask_gt_threshold_bool_cuda = torch::where(torch::gt(m_scores_fp32_cuda, threshold), 1, 0);       // all-zero m_scores_fp32_cuda is impossible
    m_keypoints_gt_threshold_int32_cuda = m_mask_gt_threshold_bool_cuda.nonzero().to(torch::kInt32);    // r, c

    m_scores_gt_threshold_fp32_cuda = m_scores_fp32_cuda.view(-1).index_select(
        0, 
        m_mask_gt_threshold_bool_cuda.view(-1).nonzero().squeeze()
    );

    if (m_scores_gt_threshold_fp32_cuda.size(0) == 0) {
        return std::make_tuple(
            int(0), 
            m_keypoints_topk_int32_cuda.clone().contiguous(), 
            m_keypoints_topk_normalized_fp32_cuda.clone().contiguous(), 
            m_descriptors_topk_fp32_cuda.clone().contiguous()
        );
    }

    m_k = m_scores_gt_threshold_fp32_cuda.size(0) < max_num_keypoints ? m_scores_gt_threshold_fp32_cuda.size(0) : max_num_keypoints;
    m_scores_indices_topk_cuda  = torch::topk(m_scores_gt_threshold_fp32_cuda, m_k, 0, true, true);
    m_scores_topk_fp32_cuda     = std::get<0>(m_scores_indices_topk_cuda);
    m_indices_topk_int32_cuda   = std::get<1>(m_scores_indices_topk_cuda).to(torch::kInt32);    // [row, col] = [480, 640]
    m_keypoints_topk_int32_cuda = m_keypoints_gt_threshold_int32_cuda.index_select(0, m_indices_topk_int32_cuda).flip(1).to(torch::kInt32); // [col, row] = [640, 480], same as opencv

    int s = 8;
    int w = m_descriptors_fp32_cuda.size(2);
    int h = m_descriptors_fp32_cuda.size(1);
    m_divider = torch::tensor(
        {(w * s - s / 2 - 0.5f), (h * s - s / 2 - 0.5f)}, 
        torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA).requires_grad(false)
    );

    m_keypoints_topk_fp32_cuda = torch::div(m_keypoints_topk_int32_cuda.to(torch::kFloat32) - float(s / 2) + 0.5f, m_divider) * 2.0f - 1.0f;
    m_descriptors_topk_fp32_cuda = torch::nn::functional::grid_sample(
        m_descriptors_fp32_cuda.unsqueeze(0), 
        m_keypoints_topk_fp32_cuda.unsqueeze(0).unsqueeze(0), 
        torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true)
    ).squeeze({0, 2});
    m_descriptors_topk_fp32_cuda = torch::nn::functional::normalize(
        m_descriptors_topk_fp32_cuda, 
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(0)
    ).permute({1, 0});

    m_image_rows = float(m_input_shape.at("image")[2]);
    m_image_cols = float(m_input_shape.at("image")[3]);
    m_scale = std::max(m_image_cols, m_image_rows) / 2.0f;
    m_shift = torch::tensor(
        {m_image_cols / 2.0f, m_image_rows / 2.0f}, 
        torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA).requires_grad(false)
    );
    m_keypoints_topk_normalized_fp32_cuda = (m_keypoints_topk_int32_cuda.to(torch::kFloat32) - m_shift) / m_scale;  // lightglue takes in normalized keypoint coordinats.

    return std::make_tuple(
        int(m_keypoints_topk_int32_cuda.size(0)), 
        m_keypoints_topk_int32_cuda.clone().contiguous(), 
        m_keypoints_topk_normalized_fp32_cuda.clone().contiguous(), 
        m_descriptors_topk_fp32_cuda.clone().contiguous()
    );
}

void SuperPointMonoTRT::RecordCUDAGraph() {
    // cudaStreamSynchronize(cuda_stream);
    cudaDeviceSynchronize();
    // cudaStreamCaptureModeGlobal works
    // cudaStreamCaptureModeThreadLocal ???
    // cudaStreamCaptureModeRelaxed ???
    cudaStreamBeginCapture(cuda_stream, cudaStreamCaptureModeRelaxed);

    assert(m_context->enqueueV3(cuda_stream));

    cudaStreamEndCapture(cuda_stream, &cuda_graph);
    cudaGraphInstantiate(&cuda_graph_exec, cuda_graph, nullptr, nullptr, 0);
    // cudaStreamSynchronize(cuda_stream);
    cudaDeviceSynchronize();
}

void SuperPointMonoTRT::LaunchCUDAGraph() {
    cudaGraphLaunch(cuda_graph_exec, cuda_stream);
}

void SuperPointMonoTRT::Sync() {
    cudaStreamSynchronize(cuda_stream);
}

