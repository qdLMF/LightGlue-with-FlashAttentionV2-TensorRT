//
// Created by https://github.com/qdLMF on 25-02-16.
//

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <NvInfer.h>
#include <NvInferPlugin.h>

#define ATTENTION_BATCH       1
#define ATTENTION_NUM_HEADS   4
#define ATTENTION_MAX_SEQ_LEN 2048
#define ATTENTION_HEAD_DIM    64

namespace nvinfer1 {

namespace plugin {

class LightGlueAttentionPlugin : public nvinfer1::IPluginV2DynamicExt {
public : 
    LightGlueAttentionPlugin();

    LightGlueAttentionPlugin(const void* serialized_data, size_t serialized_length);

    ~LightGlueAttentionPlugin() override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    void destroy() noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    void setPluginNamespace(char const* plugin_namespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void attachToContext(
        cudnnContext* cudnn_context, 
        cublasContext* cublas_context, 
        nvinfer1::IGpuAllocator* gpu_allocator
    ) noexcept override;

    void detachFromContext() noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    int32_t getNbOutputs() const noexcept override;

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t output_index, 
        const nvinfer1::DimsExprs* inputs_dim, 
        int32_t nb_inputs, 
        nvinfer1::IExprBuilder& expr_builder
    ) noexcept override;

    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs_desc, 
        int32_t nb_inputs, 
        const nvinfer1::PluginTensorDesc* outputs_desc, 
        int32_t nb_outputs
    ) const noexcept override;

    int32_t enqueue(
        const nvinfer1::PluginTensorDesc* inputs_desc, 
        const nvinfer1::PluginTensorDesc* outputs_desc, 
        const void* const* inputs_ptr, 
        void* const* outputs_ptr, 
        void* workspace, 
        cudaStream_t stream
    ) noexcept override;

    bool supportsFormatCombination(
        int32_t index, 
        const nvinfer1::PluginTensorDesc* inputs_and_outputs_desc, 
        int32_t nb_inputs, 
        int32_t nb_outputs
    ) noexcept override; 

    nvinfer1::DataType getOutputDataType(
        int32_t output_index, 
        const nvinfer1::DataType* inputs_type, 
        int32_t nb_inputs
    ) const noexcept override;

    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* inputs_desc, 
        int32_t nb_inputs, 
        const nvinfer1::DynamicPluginTensorDesc* outputs_desc, 
        int32_t nb_outputs
    ) noexcept override;

private : 
    std::string m_plugin_namespace;

    static constexpr size_t workspace_Q_num_bytes = (ATTENTION_BATCH * ATTENTION_NUM_HEADS * ATTENTION_MAX_SEQ_LEN * ATTENTION_HEAD_DIM * sizeof(half));
    static constexpr size_t workspace_K_num_bytes = (ATTENTION_BATCH * ATTENTION_NUM_HEADS * ATTENTION_MAX_SEQ_LEN * ATTENTION_HEAD_DIM * sizeof(half));
    static constexpr size_t workspace_V_num_bytes = (ATTENTION_BATCH * ATTENTION_NUM_HEADS * ATTENTION_MAX_SEQ_LEN * ATTENTION_HEAD_DIM * sizeof(half));
    static constexpr size_t workspace_O_num_bytes = (ATTENTION_BATCH * ATTENTION_NUM_HEADS * ATTENTION_MAX_SEQ_LEN * ATTENTION_HEAD_DIM * sizeof(float));

    static constexpr size_t workspace_Q_offset_in_bytes = 0;
    static constexpr size_t workspace_K_offset_in_bytes = 1 * (ATTENTION_BATCH * ATTENTION_NUM_HEADS * ATTENTION_MAX_SEQ_LEN * ATTENTION_HEAD_DIM * sizeof(half));
    static constexpr size_t workspace_V_offset_in_bytes = 2 * (ATTENTION_BATCH * ATTENTION_NUM_HEADS * ATTENTION_MAX_SEQ_LEN * ATTENTION_HEAD_DIM * sizeof(half));
    static constexpr size_t workspace_O_offset_in_bytes = 3 * (ATTENTION_BATCH * ATTENTION_NUM_HEADS * ATTENTION_MAX_SEQ_LEN * ATTENTION_HEAD_DIM * sizeof(half));
};

// ----------------------------------------------------------------------------------------

class LightGlueAttentionPluginCreator : public nvinfer1::IPluginCreator {
public : 
    LightGlueAttentionPluginCreator();

    ~LightGlueAttentionPluginCreator() override;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    void setPluginNamespace(char const* plugin_namespace) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override;

    nvinfer1::IPluginV2DynamicExt* createPlugin(
        char const* name,
        const nvinfer1::PluginFieldCollection* fc
    ) noexcept override;

    nvinfer1::IPluginV2DynamicExt* deserializePlugin(
        char const* name, 
        void const* serialized_data,
        size_t serialized_length
    ) noexcept override;

private : 

    static nvinfer1::PluginFieldCollection m_plugin_field_collection;

    static std::vector<nvinfer1::PluginField> m_plugin_field_vector;

    std::string m_plugin_namespace;
};

}   // namespace plugin

}   // namespace nvinfer1

