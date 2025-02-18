//
// Created by https://github.com/qdLMF on 25-02-16.
//

#include "./buffer_trt.h"


using namespace tensorrt_log;
using namespace tensorrt_buffer;

void BufferTRT::Set(
    std::shared_ptr<nvinfer1::ICudaEngine> engine, 
    const int batch_size,
    const nvinfer1::IExecutionContext* context
) {
    assert(engine != nullptr);
    assert(engine->hasImplicitBatchDimension() || batch_size == 0);
    // Create host and device buffers
    m_cur_buffer_numel.clear();
    m_cur_buffer_dtype.clear();
    m_i_indicies.clear();
    m_o_indicies.clear();
    for (int i = 0; i < engine->getNbBindings(); i++) {
        auto dims = context ? context->getBindingDimensions(i) : engine->getBindingDimensions(i);
        size_t vol = context || !batch_size ? 1 : static_cast<size_t>(batch_size);
        nvinfer1::DataType type = engine->getBindingDataType(i);
        int vecDim = engine->getBindingVectorizedDim(i);
        if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
        {
            int scalarsPerVec = engine->getBindingComponentsPerElement(i);
            dims.d[vecDim] = tensorrt_common::divUp(dims.d[vecDim], scalarsPerVec);
            vol *= scalarsPerVec;
        }
        vol *= tensorrt_common::volume(dims);
        m_cur_buffer_numel.push_back(vol);  // * tensorrt_common::getElementSize(type));
        m_cur_buffer_dtype.push_back(type);
        if (engine->bindingIsInput(i)) {
            m_i_indicies.push_back(i);
        } else {
            m_o_indicies.push_back(i);
        }
    }

    bool reset = false;
    if (m_buffer_numel.size() != m_cur_buffer_numel.size()) {
        reset = true;
    } else {
        for (int i = 0; i < m_cur_buffer_numel.size(); ++i) {
            if ((m_buffer_numel[i] != m_cur_buffer_numel[i]) || (m_buffer_dtype[i] != m_cur_buffer_dtype[i])) {
                reset = true;
                break;
            }
        }
    }

    if (reset) {
        if (m_buffer_numel.size() != m_cur_buffer_numel.size()) {
            m_managed_buffers.clear();
            m_device_bindings.clear();
            for (int i = 0; i < m_cur_buffer_numel.size(); i++) {
                std::unique_ptr<ManagedBuffer> managed_buffer{new ManagedBuffer()};
                managed_buffer->deviceBuffer = DeviceBuffer(m_cur_buffer_numel[i], m_cur_buffer_dtype[i]);
                managed_buffer->hostBuffer = HostBuffer(m_cur_buffer_numel[i], m_cur_buffer_dtype[i]);
                m_device_bindings.emplace_back(managed_buffer->deviceBuffer.data());
                m_managed_buffers.emplace_back(std::move(managed_buffer));
            }
        } else {
            for (int i = 0; i < m_cur_buffer_numel.size(); i++) {
                if ((m_buffer_dtype[i] == m_cur_buffer_dtype[i]) && (m_buffer_numel[i] != m_cur_buffer_numel[i])) {
                    m_managed_buffers[i]->deviceBuffer.resize(m_cur_buffer_numel[i]);
                    m_managed_buffers[i]->hostBuffer.resize(m_cur_buffer_numel[i]);
                } else if (m_buffer_dtype[i] != m_cur_buffer_dtype[i]) {
                    std::unique_ptr<ManagedBuffer> managed_buffer{new ManagedBuffer()};
                    managed_buffer->deviceBuffer = DeviceBuffer(m_cur_buffer_numel[i], m_cur_buffer_dtype[i]);
                    managed_buffer->hostBuffer = HostBuffer(m_cur_buffer_numel[i], m_cur_buffer_dtype[i]);
                    m_managed_buffers[i] = std::move(managed_buffer);
                }
                m_device_bindings[i] = m_managed_buffers[i]->deviceBuffer.data();
            }
        }
        m_buffer_numel = m_cur_buffer_numel;
        m_buffer_dtype = m_cur_buffer_dtype;
    }
}

std::vector<void*> BufferTRT::GetDeviceBindings() const {
    return m_device_bindings;
}

void* BufferTRT::GetDeviceBuffer(const int index) const {
    return _GetBuffer(false, index);
}

void* BufferTRT::GetHostBuffer(const int index) const {
    return _GetBuffer(true, index);
}

void BufferTRT::CopyInputToDevice() {
    _MemcpyBuffers(true, false, false);
}

void BufferTRT::CopyOutputToHost() {
    _MemcpyBuffers(false, true, false);
}

void BufferTRT::CopyInputToDeviceAsync(const cudaStream_t& stream) {
    _MemcpyBuffers(true, false, true, stream);
}

void BufferTRT::CopyOutputToHostAsync(const cudaStream_t& stream) {
    _MemcpyBuffers(false, true, true, stream);
}

size_t BufferTRT::GetByteSize(const int index) const {
    if (index < 0 || index >= m_managed_buffers.size())
        return ~size_t(0);
    return m_managed_buffers[index]->hostBuffer.nbBytes();
}

void BufferTRT::CopyInputFromPtr(
    const int index, 
    const void* ptr, 
    const size_t byte_size, 
    bool is_device_ptr, 
    const cudaStream_t& stream
) {
    assert(index >= 0 && index < m_managed_buffers.size());
    assert(ptr != nullptr);
    assert(byte_size == m_managed_buffers[index]->deviceBuffer.nbBytes());
    if (stream == 0) {
        CHECK(cudaMemcpy(m_managed_buffers[index]->deviceBuffer.data(), ptr, byte_size, is_device_ptr ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(m_managed_buffers[index]->hostBuffer.data(), ptr, byte_size, is_device_ptr ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost));
    } else {
        CHECK(cudaMemcpyAsync(m_managed_buffers[index]->deviceBuffer.data(), ptr, byte_size, is_device_ptr ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(m_managed_buffers[index]->hostBuffer.data(), ptr, byte_size, is_device_ptr ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost, stream));
    }
}

void* BufferTRT::_GetBuffer(const bool is_host, const int index) const {
    if (index < 0 || index >= m_managed_buffers.size()) {
        return nullptr;
    }
    return (is_host ? m_managed_buffers[index]->hostBuffer.data() : m_managed_buffers[index]->deviceBuffer.data());
}

void BufferTRT::_MemcpyBuffers(const bool copy_input, const bool device_to_host, const bool async, const cudaStream_t& stream) {
    std::vector<size_t> indicies = copy_input ? m_i_indicies : m_o_indicies;
    for (int i = 0; i < indicies.size(); i++) {
        size_t index = indicies[i];
        void* dst_ptr       = device_to_host ? m_managed_buffers[index]->hostBuffer.data()   : m_managed_buffers[index]->deviceBuffer.data();
        const void* src_ptr = device_to_host ? m_managed_buffers[index]->deviceBuffer.data() : m_managed_buffers[index]->hostBuffer.data();
        const size_t byte_size = m_managed_buffers[index]->hostBuffer.nbBytes();
        const cudaMemcpyKind memcpy_type = device_to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
        if (async) {
            CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, byte_size, memcpy_type, stream));
        } else {
            CHECK(cudaMemcpy(dst_ptr, src_ptr, byte_size, memcpy_type));
        }
    }
}

