#ifndef BUFFER_TRT_H
#define BUFFER_TRT_H

#include <cstdlib>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include <NvInfer.h>

#include "3rdparty/tensorrtbuffer/include/logger.h"
#include "3rdparty/tensorrtbuffer/include/buffers.h"


class BufferTRT {

public : 

    BufferTRT() = default;
    BufferTRT(const BufferTRT&) = delete;
    BufferTRT& operator=(const BufferTRT&) = delete;
    BufferTRT(BufferTRT&&) = delete;
    BufferTRT& operator=(BufferTRT&&) = delete;
    ~BufferTRT() = default;

    void Set(
        std::shared_ptr<nvinfer1::ICudaEngine> engine, 
        const int batch_size = 0,
        const nvinfer1::IExecutionContext *context = nullptr
    );

    //!
    //! \brief Returns a vector of device buffers that you can use directly as
    //!        bindings for the execute and enqueue methods of IExecutionContext.
    //!
    std::vector<void*> GetDeviceBindings() const;

    //!
    //! \brief Returns the device buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void* GetDeviceBuffer(const int index) const;

    //!
    //! \brief Returns the host buffer corresponding to tensorName.
    //!        Returns nullptr if no such tensor can be found.
    //!
    void* GetHostBuffer(const int index) const;

    //!
    //! \brief Copy the contents of input host buffers to input device buffers synchronously.
    //!
    void CopyInputToDevice();

    //!
    //! \brief Copy the contents of output device buffers to output host buffers synchronously.
    //!
    void CopyOutputToHost();

    //!
    //! \brief Copy the contents of input host buffers to input device buffers asynchronously.
    //!
    void CopyInputToDeviceAsync(const cudaStream_t& stream = 0);

    //!
    //! \brief Copy the contents of output device buffers to output host buffers asynchronously.
    //!
    void CopyOutputToHostAsync(const cudaStream_t& stream = 0);

    //!
    //! \brief Returns the size of the host and device buffers that correspond to tensorName.
    //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
    //!
    size_t GetByteSize(const int index) const;

    void CopyInputFromPtr(
        const int index, 
        const void* ptr, 
        const size_t byte_size, 
        bool is_device_ptr, 
        const cudaStream_t& stream = 0
    );

private:

    void* _GetBuffer(const bool is_host, const int index) const;

    void _MemcpyBuffers(const bool copy_input, const bool device_to_host, const bool async, const cudaStream_t& stream = 0);

private : 

    std::vector<size_t> m_i_indicies;
    std::vector<size_t> m_o_indicies;
    std::vector<size_t> m_cur_buffer_numel;
    std::vector<size_t> m_buffer_numel;
    std::vector<nvinfer1::DataType> m_cur_buffer_dtype;
    std::vector<nvinfer1::DataType> m_buffer_dtype;
    std::vector<std::unique_ptr<tensorrt_buffer::ManagedBuffer>> m_managed_buffers;
    std::vector<void*> m_device_bindings;
};

#endif