//
// Created by https://github.com/qdLMF on 25-02-16.
//

#include "./lightglue_trt.h"


#define LIGHTGLUE_SCORE_THRESHOLD 0.1f

LightGlueTRT::LightGlueTRT(
    const std::string& trt_engine_file_path, 
    const std::string& lightglue_plugin_file_path, 
    tensorrt_log::Logger& gLogger
) : m_trt_engine_file_path(trt_engine_file_path) {
    m_lightglue_plugin_handle = dlopen(lightglue_plugin_file_path.c_str(), RTLD_LAZY); 
    if (!m_lightglue_plugin_handle) {
        std::cerr << "LightGlue : Failed to load attention plugin library! dlerror() : " << dlerror() << std::endl;
        return;
    }

    std::ifstream file_stream(trt_engine_file_path.c_str(), std::ios::binary);
    if (!file_stream.is_open()) {
        std::cerr << "LightGlue : Failed to open engine file!" << std::endl;
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
        std::cerr << "LightGlue : Failed to construct IRuntime!" << std::endl;
        delete [] engine_stream;
        return;
    }
    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(engine_stream, engine_size));
    if (m_engine == nullptr) {
        std::cerr << "LightGlue : Failed to construct ICudaEngine!" << std::endl;
        delete [] engine_stream;
        return;
    }
    delete [] engine_stream;
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());

    m_bindings_name_to_index.clear();
    m_bindings_name_to_index["keypoints_0"] = m_engine->getBindingIndex("keypoints_0");
    m_bindings_name_to_index["keypoints_1"] = m_engine->getBindingIndex("keypoints_1");
    m_bindings_name_to_index["descriptors_0"] = m_engine->getBindingIndex("descriptors_0");
    m_bindings_name_to_index["descriptors_1"] = m_engine->getBindingIndex("descriptors_1");
    m_bindings_name_to_index["lightglue_scores"] = m_engine->getBindingIndex("lightglue_scores");
    m_bindings_name_to_index["lightglue_descriptors_0"] = m_engine->getBindingIndex("lightglue_descriptors_0");
    m_bindings_name_to_index["lightglue_descriptors_1"] = m_engine->getBindingIndex("lightglue_descriptors_1");
}

void LightGlueTRT::SetInputShape(const std::unordered_map<std::string, std::vector<int64_t>>& input_shape) {
    assert(input_shape.find("keypoints_0") != input_shape.end());
    assert(input_shape.find("keypoints_0")->second.size() == 3);
    assert(
        input_shape.find("keypoints_0")->second[0] == 1    && 
        input_shape.find("keypoints_0")->second[1] <= 1024 &&
        input_shape.find("keypoints_0")->second[2] == 2
    );

    assert(input_shape.find("keypoints_1") != input_shape.end());
    assert(input_shape.find("keypoints_1")->second.size() == 3);
    assert(
        input_shape.find("keypoints_1")->second[0] == 1    && 
        input_shape.find("keypoints_1")->second[1] <= 1024 &&
        input_shape.find("keypoints_1")->second[2] == 2
    );

    assert(input_shape.find("descriptors_0") != input_shape.end());
    assert(input_shape.find("descriptors_0")->second.size() == 3);
    assert(
        input_shape.find("descriptors_0")->second[0] == 1    && 
        input_shape.find("descriptors_0")->second[1] <= 1024 &&
        input_shape.find("descriptors_0")->second[2] == 256
    );

    assert(input_shape.find("descriptors_1") != input_shape.end());
    assert(input_shape.find("descriptors_1")->second.size() == 3);
    assert(
        input_shape.find("descriptors_1")->second[0] == 1    && 
        input_shape.find("descriptors_1")->second[1] <= 1024 &&
        input_shape.find("descriptors_1")->second[2] == 256
    );

    assert(
        input_shape.find("keypoints_0")->second[1] == input_shape.find("descriptors_0")->second[1] && 
        input_shape.find("keypoints_1")->second[1] == input_shape.find("descriptors_1")->second[1]
    );

    m_input_shape = input_shape;
    m_context->setBindingDimensions(
        m_bindings_name_to_index.at("keypoints_0"), 
        IntVectorToDims(m_input_shape.at("keypoints_0"))
    );
    m_context->setBindingDimensions(
        m_bindings_name_to_index.at("keypoints_1"), 
        IntVectorToDims(m_input_shape.at("keypoints_1"))
    );
    m_context->setBindingDimensions(
        m_bindings_name_to_index.at("descriptors_0"), 
        IntVectorToDims(m_input_shape.at("descriptors_0"))
    );
    m_context->setBindingDimensions(
        m_bindings_name_to_index.at("descriptors_1"), 
        IntVectorToDims(m_input_shape.at("descriptors_1"))
    );

    m_buffer.Set(
        m_engine,
        0,
        m_context.get()
    );
    
    m_output_shape["lightglue_descriptors_0"] = DimsToIntVector(m_context->getBindingDimensions(m_bindings_name_to_index.at("lightglue_descriptors_0")));
    m_output_shape["lightglue_descriptors_1"] = DimsToIntVector(m_context->getBindingDimensions(m_bindings_name_to_index.at("lightglue_descriptors_1")));
    m_output_shape["lightglue_scores"]        = DimsToIntVector(m_context->getBindingDimensions(m_bindings_name_to_index.at("lightglue_scores")));
}

void LightGlueTRT::CopyInputTensor(const std::unordered_map<std::string, torch::Tensor>& input_tensor) {
    assert(input_tensor.find("keypoints_0") != input_tensor.end());
    assert(input_tensor.at("keypoints_0").sizes() == m_input_shape.at("keypoints_0"));
    assert(input_tensor.at("keypoints_0").dtype() == torch::kFloat32);

    assert(input_tensor.find("keypoints_1") != input_tensor.end());
    assert(input_tensor.at("keypoints_1").sizes() == m_input_shape.at("keypoints_1"));
    assert(input_tensor.at("keypoints_1").dtype() == torch::kFloat32);

    assert(input_tensor.find("descriptors_0") != input_tensor.end());
    assert(input_tensor.at("descriptors_0").sizes() == m_input_shape.at("descriptors_0"));
    assert(input_tensor.at("descriptors_0").dtype() == torch::kFloat32);
    
    assert(input_tensor.find("descriptors_1") != input_tensor.end());
    assert(input_tensor.at("descriptors_1").sizes() == m_input_shape.at("descriptors_1"));
    assert(input_tensor.at("descriptors_1").dtype() == torch::kFloat32);

    m_buffer.CopyInputFromPtr(
        m_bindings_name_to_index.at("keypoints_0"), 
        input_tensor.at("keypoints_0").data_ptr(), 
        input_tensor.at("keypoints_0").numel() * input_tensor.at("keypoints_0").element_size(), 
        input_tensor.at("keypoints_0").device() == torch::kCUDA ? true : false
    );
    m_buffer.CopyInputFromPtr(
        m_bindings_name_to_index.at("keypoints_1"), 
        input_tensor.at("keypoints_1").data_ptr(), 
        input_tensor.at("keypoints_1").numel() * input_tensor.at("keypoints_1").element_size(), 
        input_tensor.at("keypoints_1").device() == torch::kCUDA ? true : false
    );
    m_buffer.CopyInputFromPtr(
        m_bindings_name_to_index.at("descriptors_0"), 
        input_tensor.at("descriptors_0").data_ptr(), 
        input_tensor.at("descriptors_0").numel() * input_tensor.at("descriptors_0").element_size(), 
        input_tensor.at("descriptors_0").device() == torch::kCUDA ? true : false
    );
    m_buffer.CopyInputFromPtr(
        m_bindings_name_to_index.at("descriptors_1"), 
        input_tensor.at("descriptors_1").data_ptr(), 
        input_tensor.at("descriptors_1").numel() * input_tensor.at("descriptors_1").element_size(), 
        input_tensor.at("descriptors_1").device() == torch::kCUDA ? true : false
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> LightGlueTRT::Forward() {
    assert(m_context->executeV2(m_buffer.GetDeviceBindings().data()));

    torch::NoGradGuard torch_no_grad;

    m_descriptors_0_fp32_cuda = torch::from_blob(
        m_buffer.GetDeviceBuffer(m_bindings_name_to_index.at("lightglue_descriptors_0")), 
        m_output_shape.at("lightglue_descriptors_0"), 
        torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA).requires_grad(false)
    ).squeeze(0);
    m_descriptors_1_fp32_cuda = torch::from_blob(
        m_buffer.GetDeviceBuffer(m_bindings_name_to_index.at("lightglue_descriptors_1")), 
        m_output_shape.at("lightglue_descriptors_1"), 
        torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA).requires_grad(false)
    ).squeeze(0);
    m_scores_fp32_cuda = torch::from_blob(
        m_buffer.GetDeviceBuffer(m_bindings_name_to_index.at("lightglue_scores")), 
        m_output_shape.at("lightglue_scores"), 
        torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA).requires_grad(false)
    ).squeeze(0);

    m_max_scores_0_tuple = torch::topk(m_scores_fp32_cuda, 1, 1, true, false);
    m_max_scores_1_tuple = torch::topk(m_scores_fp32_cuda, 1, 0, true, false);
    m_max_scores_values_0_fp32_cuda  = std::get<0>(m_max_scores_0_tuple).squeeze(1);
    m_max_scores_indices_0_int64_cuda = std::get<1>(m_max_scores_0_tuple).squeeze(1);
    m_max_scores_values_1_fp32_cuda  = std::get<0>(m_max_scores_1_tuple).squeeze(0);
    m_max_scores_indices_1_int64_cuda = std::get<1>(m_max_scores_1_tuple).squeeze(0);

    m_linear_indices_0_int64_cuda = torch::arange(
        m_max_scores_indices_0_int64_cuda.size(0), 
        torch::TensorOptions().dtype(torch::kInt64).layout(torch::kStrided).device(torch::kCUDA).requires_grad(false)
    );  // pass

    m_temp_int64_cuda = torch::gather(m_max_scores_indices_1_int64_cuda, 0, m_max_scores_indices_0_int64_cuda);

    m_mutual_max_scores_flag_bool_cuda = m_linear_indices_0_int64_cuda == m_temp_int64_cuda;

    m_max_scores_values_0_fp32_cuda = torch::exp(m_max_scores_values_0_fp32_cuda);
    m_zeros_fp32_cuda = torch::zeros_like(m_max_scores_values_0_fp32_cuda);
    m_mutual_max_scores_fp32_cuda = torch::where(m_mutual_max_scores_flag_bool_cuda, m_max_scores_values_0_fp32_cuda, m_zeros_fp32_cuda);
    m_mutual_max_scores_mask_bool_cuda = m_mutual_max_scores_fp32_cuda > LIGHTGLUE_SCORE_THRESHOLD;

    m_match_indices_0_int64_cuda = torch::masked_select(m_temp_int64_cuda, m_mutual_max_scores_mask_bool_cuda);
    m_match_indices_1_int64_cuda = torch::gather(m_max_scores_indices_0_int64_cuda, 0, m_match_indices_0_int64_cuda);

    m_match_scores_fp32_cuda = torch::gather(m_mutual_max_scores_fp32_cuda, 0, m_match_indices_0_int64_cuda);

    return std::make_tuple(
        m_descriptors_0_fp32_cuda.clone().contiguous(), 
        m_descriptors_1_fp32_cuda.clone().contiguous(), 
        m_scores_fp32_cuda.clone().contiguous(), 
        m_match_indices_0_int64_cuda.clone().contiguous(), 
        m_match_indices_1_int64_cuda.clone().contiguous(), 
        m_match_scores_fp32_cuda.clone().contiguous()
    );
}


