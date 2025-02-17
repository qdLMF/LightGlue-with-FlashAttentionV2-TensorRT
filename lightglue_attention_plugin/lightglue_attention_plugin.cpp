//
// Created by https://github.com/qdLMF on 25-02-16.
//

#include "./common/checkMacrosPlugin.h"

#include "./attention_headdim_64_fp16in_fp16out.cuh"
#include "./attention_headdim_64_fp16in_fp32out.cuh"

#include "./lightglue_attention_plugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using nvinfer1::plugin::LightGlueAttentionPlugin;
using nvinfer1::plugin::LightGlueAttentionPluginCreator;

namespace {

constexpr char const* LIGHTGLUE_ATTENTION_PLUGIN_NAME    { "MHAHeadDim64" };
constexpr char const* LIGHTGLUE_ATTENTION_PLUGIN_VERSION { "1"            };

}   // namespace


nvinfer1::PluginFieldCollection LightGlueAttentionPluginCreator::m_plugin_field_collection{};
std::vector<nvinfer1::PluginField> LightGlueAttentionPluginCreator::m_plugin_field_vector;

LightGlueAttentionPlugin::LightGlueAttentionPlugin() {}

LightGlueAttentionPlugin::LightGlueAttentionPlugin(const void* serialized_data, size_t serialized_length) {}

LightGlueAttentionPlugin::~LightGlueAttentionPlugin() { 
    terminate(); 
}

int32_t LightGlueAttentionPlugin::initialize() noexcept { return 0; }

void LightGlueAttentionPlugin::terminate() noexcept { 
}

void LightGlueAttentionPlugin::destroy() noexcept { 
    delete this; 
}

size_t LightGlueAttentionPlugin::getSerializationSize() const noexcept { return 0; }

void LightGlueAttentionPlugin::serialize(void* buffer) const noexcept {}

char const* LightGlueAttentionPlugin::getPluginType() const noexcept { return LIGHTGLUE_ATTENTION_PLUGIN_NAME; }

char const* LightGlueAttentionPlugin::getPluginVersion() const noexcept { return LIGHTGLUE_ATTENTION_PLUGIN_VERSION; }

void LightGlueAttentionPlugin::setPluginNamespace(char const* plugin_namespace) noexcept { m_plugin_namespace = plugin_namespace; }

char const* LightGlueAttentionPlugin::getPluginNamespace() const noexcept { return m_plugin_namespace.c_str(); }

void LightGlueAttentionPlugin::attachToContext(
    cudnnContext* cudnn_context, 
    cublasContext* cublas_context, 
    nvinfer1::IGpuAllocator* gpu_allocator
) noexcept {}

void LightGlueAttentionPlugin::detachFromContext() noexcept {}

nvinfer1::IPluginV2DynamicExt* LightGlueAttentionPlugin::clone() const noexcept { 
    try {
        auto* plugin = new LightGlueAttentionPlugin();
        plugin->setPluginNamespace(m_plugin_namespace.c_str());
        plugin->initialize();
        return plugin;
    } catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

int32_t LightGlueAttentionPlugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::DimsExprs LightGlueAttentionPlugin::getOutputDimensions(
    int32_t output_index, 
    const nvinfer1::DimsExprs* inputs_dim, 
    int32_t nb_inputs, 
    nvinfer1::IExprBuilder& expr_builder
) noexcept {
    PLUGIN_ASSERT(
           inputs_dim != nullptr 
        && nb_inputs == 3 
        && output_index == 0
    );
    PLUGIN_ASSERT(inputs_dim[0].nbDims == 4 && inputs_dim[1].nbDims == 4 && inputs_dim[2].nbDims == 4);

    DimsExprs output(inputs_dim[0]);
    return output;
}

size_t LightGlueAttentionPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs_desc, 
    int32_t nb_inputs, 
    const nvinfer1::PluginTensorDesc* outputs_desc, 
    int32_t nb_outputs
) const noexcept {
    PLUGIN_ASSERT(
           inputs_desc  != nullptr 
        && outputs_desc != nullptr 
        && nb_inputs    == 3 
        && nb_outputs   == 1
    );

    return \
    (ATTENTION_BATCH * ATTENTION_NUM_HEADS * ATTENTION_MAX_SEQ_LEN * ATTENTION_HEAD_DIM * sizeof(half)) * 3 
    + (ATTENTION_BATCH * ATTENTION_NUM_HEADS * ATTENTION_MAX_SEQ_LEN * ATTENTION_HEAD_DIM * sizeof(float));
}

int32_t LightGlueAttentionPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputs_desc, 
    const nvinfer1::PluginTensorDesc* outputs_desc, 
    const void* const* inputs_ptr, 
    void* const* outputs_ptr, 
    void* workspace, 
    cudaStream_t stream
) noexcept {
    PLUGIN_ASSERT(
           inputs_desc  != nullptr 
        && outputs_desc != nullptr 
        && inputs_ptr   != nullptr 
        && outputs_ptr  != nullptr 
        && workspace    != nullptr
    );

    PLUGIN_ASSERT( inputs_desc[0].dims.nbDims == 4);
    PLUGIN_ASSERT( inputs_desc[1].dims.nbDims == 4);
    PLUGIN_ASSERT( inputs_desc[2].dims.nbDims == 4);
    PLUGIN_ASSERT(outputs_desc[0].dims.nbDims == 4);

    PLUGIN_ASSERT(inputs_desc[0].dims.d[0] == ATTENTION_BATCH);
    PLUGIN_ASSERT(inputs_desc[0].dims.d[0] ==  inputs_desc[1].dims.d[0]);
    PLUGIN_ASSERT(inputs_desc[1].dims.d[0] ==  inputs_desc[2].dims.d[0]);
    PLUGIN_ASSERT(inputs_desc[0].dims.d[0] == outputs_desc[0].dims.d[0]);

    PLUGIN_ASSERT(inputs_desc[0].dims.d[1] == ATTENTION_NUM_HEADS);
    PLUGIN_ASSERT(inputs_desc[0].dims.d[1] ==  inputs_desc[1].dims.d[1]);
    PLUGIN_ASSERT(inputs_desc[1].dims.d[1] ==  inputs_desc[2].dims.d[1]);
    PLUGIN_ASSERT(inputs_desc[0].dims.d[1] == outputs_desc[0].dims.d[1]);

    PLUGIN_ASSERT(inputs_desc[0].dims.d[2] <= ATTENTION_MAX_SEQ_LEN);
    PLUGIN_ASSERT(inputs_desc[1].dims.d[2] <= ATTENTION_MAX_SEQ_LEN);
    PLUGIN_ASSERT(inputs_desc[1].dims.d[2] ==  inputs_desc[2].dims.d[2]);
    PLUGIN_ASSERT(inputs_desc[0].dims.d[2] == outputs_desc[0].dims.d[2]);

    PLUGIN_ASSERT(inputs_desc[0].dims.d[3] == ATTENTION_HEAD_DIM);
    PLUGIN_ASSERT(inputs_desc[0].dims.d[3] ==  inputs_desc[1].dims.d[3]);
    PLUGIN_ASSERT(inputs_desc[1].dims.d[3] ==  inputs_desc[2].dims.d[3]);
    PLUGIN_ASSERT(inputs_desc[0].dims.d[3] == outputs_desc[0].dims.d[3]);

    PLUGIN_ASSERT( inputs_desc[0].type == nvinfer1::DataType::kFLOAT || inputs_desc[0].type == nvinfer1::DataType::kHALF);
    PLUGIN_ASSERT( inputs_desc[1].type == inputs_desc[0].type);
    PLUGIN_ASSERT( inputs_desc[2].type == inputs_desc[0].type);
    PLUGIN_ASSERT(outputs_desc[0].type == inputs_desc[0].type);

    PLUGIN_ASSERT( inputs_desc[0].format == nvinfer1::TensorFormat::kLINEAR);
    PLUGIN_ASSERT( inputs_desc[1].format == nvinfer1::TensorFormat::kLINEAR);
    PLUGIN_ASSERT( inputs_desc[2].format == nvinfer1::TensorFormat::kLINEAR);
    PLUGIN_ASSERT(outputs_desc[0].format == nvinfer1::TensorFormat::kLINEAR);

    int batch         = ATTENTION_BATCH;
    int num_heads     = ATTENTION_NUM_HEADS;
    int num_Q_real    = inputs_desc[0].dims.d[2];
    int num_KV_real   = inputs_desc[1].dims.d[2];
    int num_Q_padded  = ((num_Q_real  + 64 - 1) / 64) * 64;
    int num_KV_padded = ((num_KV_real + 64 - 1) / 64) * 64;

    uint8_t* Q_ptr = reinterpret_cast<uint8_t*>(workspace) + workspace_Q_offset_in_bytes;
    uint8_t* K_ptr = reinterpret_cast<uint8_t*>(workspace) + workspace_K_offset_in_bytes;
    uint8_t* V_ptr = reinterpret_cast<uint8_t*>(workspace) + workspace_V_offset_in_bytes;
    uint8_t* O_ptr = reinterpret_cast<uint8_t*>(workspace) + workspace_O_offset_in_bytes;

    if (inputs_desc[0].type == nvinfer1::DataType::kHALF) {
        AttentionHeadDim64::launch_QKV_move_from_unpadded_to_padded_fp16(
            reinterpret_cast<const half*>(inputs_ptr[0]), 
            reinterpret_cast<const half*>(inputs_ptr[1]), 
            reinterpret_cast<const half*>(inputs_ptr[2]), 
            reinterpret_cast<half*>(Q_ptr), 
            reinterpret_cast<half*>(K_ptr), 
            reinterpret_cast<half*>(V_ptr), 
            batch, 
            num_heads, 
            num_Q_real, 
            num_KV_real, 
            num_Q_padded, 
            num_KV_padded, 
            stream
        );
        // cudaStreamSynchronize(stream);

        AttentionHeadDim64::launch_attention_kernel_headdim_64_no_remainder_fp16in_fp16out(
            num_Q_real  != num_Q_padded  ? reinterpret_cast<const half*>(Q_ptr) : reinterpret_cast<const half*>( inputs_ptr[0]), 
            num_KV_real != num_KV_padded ? reinterpret_cast<const half*>(K_ptr) : reinterpret_cast<const half*>( inputs_ptr[1]), 
            num_KV_real != num_KV_padded ? reinterpret_cast<const half*>(V_ptr) : reinterpret_cast<const half*>( inputs_ptr[2]), 
            num_Q_real  != num_Q_padded  ? reinterpret_cast<      half*>(O_ptr) : reinterpret_cast<      half*>(outputs_ptr[0]), 
            batch, 
            num_heads, 
            num_Q_padded, 
            num_KV_padded, 
            num_Q_real, 
            num_KV_real, 
            stream
        );
        // cudaStreamSynchronize(stream);

        if (num_Q_real != num_Q_padded) {
            AttentionHeadDim64::launch_O_move_from_padded_to_unpadded_fp16(
                reinterpret_cast<const half*>(O_ptr), 
                reinterpret_cast<half*>(outputs_ptr[0]), 
                batch, 
                num_heads, 
                num_Q_real, 
                num_Q_padded, 
                stream
            );
            // cudaStreamSynchronize(stream);
        }
    } else if (inputs_desc[0].type == nvinfer1::DataType::kFLOAT) {
        AttentionHeadDim64::launch_QKV_convert_from_fp32_to_fp16(
            reinterpret_cast<const float*>(inputs_ptr[0]), 
            reinterpret_cast<const float*>(inputs_ptr[1]), 
            reinterpret_cast<const float*>(inputs_ptr[2]), 
            reinterpret_cast<half*>(Q_ptr), 
            reinterpret_cast<half*>(K_ptr), 
            reinterpret_cast<half*>(V_ptr), 
            batch, 
            num_heads, 
            num_Q_real, 
            num_KV_real, 
            num_Q_padded, 
            num_KV_padded, 
            stream
        );
        // cudaStreamSynchronize(stream);

        AttentionHeadDim64::launch_attention_kernel_headdim_64_no_remainder_fp16in_fp32out(
            reinterpret_cast<const half*>(Q_ptr), 
            reinterpret_cast<const half*>(K_ptr), 
            reinterpret_cast<const half*>(V_ptr), 
            num_Q_real != num_Q_padded ? reinterpret_cast<float*>(O_ptr) : reinterpret_cast<float*>(outputs_ptr[0]), 
            batch, 
            num_heads, 
            num_Q_padded, 
            num_KV_padded, 
            num_Q_real, 
            num_KV_real, 
            stream
        );
        // cudaStreamSynchronize(stream);

        if (num_Q_real != num_Q_padded) {
            AttentionHeadDim64::launch_O_move_from_padded_to_unpadded_fp32(
                reinterpret_cast<const float*>(O_ptr), 
                reinterpret_cast<float*>(outputs_ptr[0]), 
                batch, 
                num_heads, 
                num_Q_real, 
                num_Q_padded, 
                stream
            );
            // cudaStreamSynchronize(stream);
        }
    }

    return 0;
}

bool LightGlueAttentionPlugin::supportsFormatCombination(
    int32_t index, 
    const nvinfer1::PluginTensorDesc* inputs_and_outputs_desc, 
    int32_t nb_inputs, 
    int32_t nb_outputs
) noexcept {
    PLUGIN_ASSERT(
           inputs_and_outputs_desc != nullptr 
        && nb_inputs  == 3 
        && nb_outputs == 1 
        && (index == 0 || index == 1 || index == 2 || index == 3)
    );

    if (index == 0) {
        return \
        (inputs_and_outputs_desc[index].type == nvinfer1::DataType::kFLOAT || inputs_and_outputs_desc[index].type == nvinfer1::DataType::kHALF) \
        && inputs_and_outputs_desc[index].format == nvinfer1::TensorFormat::kLINEAR;
    } else if (index == 1 || index == 2 || index == 3) {
        return \
        inputs_and_outputs_desc[index].type == inputs_and_outputs_desc[0].type \
        && inputs_and_outputs_desc[index].format == nvinfer1::TensorFormat::kLINEAR;
    }

    return false;
}

nvinfer1::DataType LightGlueAttentionPlugin::getOutputDataType(
    int32_t output_index, 
    const nvinfer1::DataType* inputs_type, 
    int32_t nb_inputs
) const noexcept {
    PLUGIN_ASSERT(
           inputs_type != nullptr 
        && nb_inputs == 3 
        && output_index == 0
    );

    return inputs_type[0];
}

void LightGlueAttentionPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs_desc, 
    int32_t nb_inputs, 
    const nvinfer1::DynamicPluginTensorDesc* outputs_desc, 
    int32_t nb_outputs
) noexcept {
    PLUGIN_ASSERT(
           inputs_desc  != nullptr
        && outputs_desc != nullptr
        && nb_inputs  == 3
        && nb_outputs == 1
    );

    PLUGIN_ASSERT( inputs_desc[0].desc.dims.nbDims == 4);
    PLUGIN_ASSERT( inputs_desc[1].desc.dims.nbDims == 4);
    PLUGIN_ASSERT( inputs_desc[2].desc.dims.nbDims == 4);
    PLUGIN_ASSERT(outputs_desc[0].desc.dims.nbDims == 4);

    PLUGIN_ASSERT(inputs_desc[0].desc.dims.d[0] == ATTENTION_BATCH);
    PLUGIN_ASSERT(inputs_desc[0].desc.dims.d[0] ==  inputs_desc[1].desc.dims.d[0]);
    PLUGIN_ASSERT(inputs_desc[1].desc.dims.d[0] ==  inputs_desc[2].desc.dims.d[0]);
    PLUGIN_ASSERT(inputs_desc[0].desc.dims.d[0] == outputs_desc[0].desc.dims.d[0]);

    PLUGIN_ASSERT(inputs_desc[0].desc.dims.d[1] == ATTENTION_NUM_HEADS);
    PLUGIN_ASSERT(inputs_desc[0].desc.dims.d[1] ==  inputs_desc[1].desc.dims.d[1]);
    PLUGIN_ASSERT(inputs_desc[1].desc.dims.d[1] ==  inputs_desc[2].desc.dims.d[1]);
    PLUGIN_ASSERT(inputs_desc[0].desc.dims.d[1] == outputs_desc[0].desc.dims.d[1]);

    PLUGIN_ASSERT(inputs_desc[0].desc.dims.d[2] <= ATTENTION_MAX_SEQ_LEN);
    PLUGIN_ASSERT(inputs_desc[1].desc.dims.d[2] <= ATTENTION_MAX_SEQ_LEN);
    PLUGIN_ASSERT(inputs_desc[1].desc.dims.d[2] ==  inputs_desc[2].desc.dims.d[2]);
    PLUGIN_ASSERT(inputs_desc[0].desc.dims.d[2] == outputs_desc[0].desc.dims.d[2]);

    PLUGIN_ASSERT(inputs_desc[0].desc.dims.d[3] == ATTENTION_HEAD_DIM);
    PLUGIN_ASSERT(inputs_desc[0].desc.dims.d[3] ==  inputs_desc[1].desc.dims.d[3]);
    PLUGIN_ASSERT(inputs_desc[1].desc.dims.d[3] ==  inputs_desc[2].desc.dims.d[3]);
    PLUGIN_ASSERT(inputs_desc[0].desc.dims.d[3] == outputs_desc[0].desc.dims.d[3]);

    PLUGIN_ASSERT( inputs_desc[0].desc.type == nvinfer1::DataType::kFLOAT || inputs_desc[0].desc.type == nvinfer1::DataType::kHALF);
    PLUGIN_ASSERT( inputs_desc[1].desc.type == inputs_desc[0].desc.type);
    PLUGIN_ASSERT( inputs_desc[2].desc.type == inputs_desc[0].desc.type);
    PLUGIN_ASSERT(outputs_desc[0].desc.type == inputs_desc[0].desc.type);

    PLUGIN_ASSERT( inputs_desc[0].desc.format == nvinfer1::TensorFormat::kLINEAR);
    PLUGIN_ASSERT( inputs_desc[1].desc.format == nvinfer1::TensorFormat::kLINEAR);
    PLUGIN_ASSERT( inputs_desc[2].desc.format == nvinfer1::TensorFormat::kLINEAR);
    PLUGIN_ASSERT(outputs_desc[0].desc.format == nvinfer1::TensorFormat::kLINEAR);
}

// ----------------------------------------------------------------------------------------

LightGlueAttentionPluginCreator::LightGlueAttentionPluginCreator() { 
    m_plugin_field_vector.clear();
    m_plugin_field_collection.nbFields = m_plugin_field_vector.size();
    m_plugin_field_collection.fields   = m_plugin_field_vector.data();
}

LightGlueAttentionPluginCreator::~LightGlueAttentionPluginCreator() {};

char const* LightGlueAttentionPluginCreator::getPluginName() const noexcept {
    return LIGHTGLUE_ATTENTION_PLUGIN_NAME;
}

char const* LightGlueAttentionPluginCreator::getPluginVersion() const noexcept {
    return LIGHTGLUE_ATTENTION_PLUGIN_VERSION;
}

void LightGlueAttentionPluginCreator::setPluginNamespace(char const* plugin_namespace) noexcept {
    m_plugin_namespace = plugin_namespace;
}

char const* LightGlueAttentionPluginCreator::getPluginNamespace() const noexcept {
    return m_plugin_namespace.c_str();
}

nvinfer1::PluginFieldCollection const* LightGlueAttentionPluginCreator::getFieldNames() noexcept {
    return &m_plugin_field_collection;
}

nvinfer1::IPluginV2DynamicExt* LightGlueAttentionPluginCreator::createPlugin(
    char const* name,
    const nvinfer1::PluginFieldCollection* plugin_field_collection
) noexcept {
    try {
        auto* plugin = new LightGlueAttentionPlugin();
        plugin->setPluginNamespace(m_plugin_namespace.c_str());
        plugin->initialize();
        return plugin;
    } catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::IPluginV2DynamicExt* LightGlueAttentionPluginCreator::deserializePlugin(
    char const* name, 
    void const* serialized_data,
    size_t serialized_length
) noexcept {
    try {
        auto* plugin = new LightGlueAttentionPlugin{serialized_data, serialized_length};
        plugin->setPluginNamespace(m_plugin_namespace.c_str());
        plugin->initialize();
        return plugin;
    } catch (std::exception const &e) {
        caughtError(e);
    }
    return nullptr;
}

REGISTER_TENSORRT_PLUGIN(LightGlueAttentionPluginCreator);