#pragma once

#include <vector>
#include "helpers.h"

using std::vector;

template<typename T>
class device_memory
{
    T* d_ptr_ = nullptr;
    size_t size_ = 0;

public:
    device_memory();
    explicit device_memory(size_t size);
    explicit device_memory(const vector<T>& vector, size_t start_index = 0, size_t end_index = 0);

    device_memory(const device_memory& other);
    device_memory(device_memory&& other) noexcept;
    device_memory& operator=(const device_memory& other);
    device_memory& operator=(const vector<T>& other);
    // ReSharper disable once CppMoveOperationWithoutNoexceptSpecification
    device_memory& operator=(device_memory&& other);
    ~device_memory();

    void resize(size_t size);
    void fill_with_zeroes();
    void insert_at_index(const vector<T>& vector, size_t index);
    void swap_with(device_memory<T>& other);
    void set_values(T* ptr, size_t size);

    size_t size() const;
    T* begin();
    T* end();
    const T* begin() const;
    const T* end() const;
    T last_elem() const;
    vector<T> copy_to_host() const;
};

template <typename T>
device_memory<T>::device_memory()
{
    d_ptr_ = nullptr;
    size_ = 0;
}

template <typename T>
device_memory<T>::device_memory(size_t size)
{
    if (size == 0)
    {
        d_ptr_ = nullptr;
        size_ = 0;
        return;
    }

    auto error = cudaMalloc(&d_ptr_, size * sizeof(T));
    check_status(error, "Device memory allocation error.");
    size_ = size;
}

template <typename T>
device_memory<T>::device_memory(const vector<T>& vector, const size_t start_index, size_t end_index)
{
    if (end_index == 0)
        end_index = vector.size();

    if (end_index > vector.size() || start_index > vector.size() || start_index > end_index)
        throw std::runtime_error("Trying to copy vector to device with out of bounds start or end index.");

    size_ = end_index - start_index;
    if (size_ == 0)
    {
        d_ptr_ = nullptr;
        return;
    }

    auto error = cudaMalloc(&d_ptr_, size_ * sizeof(T));
    check_status(error, "Device memory allocation error.");

    error = cudaMemcpy(d_ptr_, vector.data() + start_index, size_ * sizeof(T), cudaMemcpyHostToDevice);
    check_status(error, "Device memory copying error.");
}

template <typename T>
void device_memory<T>::resize(size_t size)
{
    if (size == size_)
        return;

    T* new_ptr = nullptr;

    if (size != 0)
    {
        cudaError_t error = cudaMalloc(&new_ptr, size * sizeof(T));
        check_status(error, "Device memory allocation error.");

        if (d_ptr_ != nullptr)
        {
            size_t smaller_size = size < size_ ? size : size_;

            error = cudaMemcpy(new_ptr, d_ptr_, smaller_size * sizeof(T), cudaMemcpyDeviceToDevice);
            check_status(error, "Device memory copying error.");

            error = cudaFree(d_ptr_);
            check_status(error, "Device memory deallocation error.");
        }
    }
    else
    {
        auto error = cudaFree(d_ptr_);
        check_status(error, "Device memory deallocation error.");
    }

    d_ptr_ = new_ptr;
    size_ = size;
}

template <typename T>
void device_memory<T>::fill_with_zeroes()
{
    if (size_ != 0)
    {
        auto error = cudaMemset(d_ptr_, 0, sizeof(T)*size_);
        check_status(error, "Device memory filling with zeroes error.");
    }
}

template <typename T>
void device_memory<T>::insert_at_index(const vector<T>& vector, const size_t index)
{
    if (index + vector.size() > size_)
        throw std::runtime_error("Attempt to write outside of device_memory bounds.");

    if (vector.size() > 0)
    {
        auto error = cudaMemcpy(d_ptr_ + index, vector.data(), vector.size() * sizeof(T), cudaMemcpyHostToDevice);
        check_status(error, "Device memory copying error.");
    }
}

template <typename T>
void device_memory<T>::swap_with(device_memory<T>& other)
{
    T* tmp_ptr = d_ptr_;
    const size_t tmp_size = size_;

    d_ptr_ = other.begin();
    size_ = other.size();
    other.set_values(tmp_ptr, tmp_size);
}

template <typename T>
void device_memory<T>::set_values(T* ptr, const size_t size)
{
    d_ptr_ = ptr;
    size_ = size;
}

template <typename T>
device_memory<T>::~device_memory()
{
    if (d_ptr_ == nullptr)
        return;

    auto error = cudaFree(d_ptr_);
    check_status(error, "Device memory deallocation error.");
}

template <typename T>
device_memory<T>::device_memory(const device_memory& other)
{
    size_ = other.size_;
    if (size_ == 0)
    {
        d_ptr_ = nullptr;
        return;
    }

    auto error = cudaMalloc(&d_ptr_, size_ * sizeof(T));
    check_status(error, "Device memory allocation error.");

    error = cudaMemcpy(d_ptr_, other.d_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
    check_status(error, "Device memory copying error.");
}

template <typename T>
device_memory<T>::device_memory(device_memory&& other) noexcept :
    d_ptr_(other.d_ptr_),
    size_(other.size_)
{
    other.size_ = 0;
    other.d_ptr_ = nullptr;
}

template <typename T>
device_memory<T>& device_memory<T>::operator=(const device_memory& other)
{
    if (this == &other)
        return *this;

    cudaError_t error;
    if (size_ == other.size() && size_ != 0)
    {
        error = cudaMemcpy(d_ptr_, other.d_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
        check_status(error, "Device memory copying error.");
        return *this;
    }

    if (d_ptr_ != nullptr)
    {
        error = cudaFree(d_ptr_);
        check_status(error, "Device memory deallocation error.");
        d_ptr_ = nullptr;
    }

    size_ = other.size_;

    if (other.size_ != 0)
    {
        error = cudaMalloc(&d_ptr_, size_ * sizeof(T));
        check_status(error, "Device memory allocation error.");

        error = cudaMemcpy(d_ptr_, other.d_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
        check_status(error, "Device memory copying error.");
    }

    return *this;
}

template <typename T>
device_memory<T>& device_memory<T>::operator=(const vector<T>& other)
{
    cudaError_t error;
    if (size_ == other.size() && size_ != 0)
    {
        error = cudaMemcpy(d_ptr_, other.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
        check_status(error, "Device memory copying error.");
        return *this;
    }

    if (d_ptr_ != nullptr)
    {
        error = cudaFree(d_ptr_);
        check_status(error, "Device memory deallocation error.");
        d_ptr_ = nullptr;
    }

    size_ = other.size();

    if (size_ != 0)
    {
        error = cudaMalloc(&d_ptr_, size_ * sizeof(T));
        check_status(error, "Device memory allocation error.");

        error = cudaMemcpy(d_ptr_, other.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
        check_status(error, "Device memory copying error.");
    }

    return *this;
}

template <typename T>
device_memory<T>& device_memory<T>::operator=(device_memory&& other)
{
    if (this == &other)
        return *this;

    if (d_ptr_ != nullptr)
    {
        auto error = cudaFree(d_ptr_);
        check_status(error, "Device memory deallocation error.");
    }

    d_ptr_ = other.d_ptr_;
    size_ = other.size_;

    other.size_ = 0;
    other.d_ptr_ = nullptr;
    return *this;
}

template <typename T>
size_t device_memory<T>::size() const
{
    return size_;
}

template <typename T>
const T* device_memory<T>::begin() const
{
    return d_ptr_;
}

template <typename T>
T* device_memory<T>::begin()
{
    return d_ptr_;
}

template <typename T>
const T* device_memory<T>::end() const
{
    return d_ptr_ + size_;
}

template <typename T>
T* device_memory<T>::end()
{
    return d_ptr_ + size_;
}

template <typename T>
T device_memory<T>::last_elem() const
{
    if (size() == 0)
        throw std::runtime_error("Trying to get last value from empty memory block.");

    T elem;
    auto error = cudaMemcpy(&elem, d_ptr_ + size_ - 1, sizeof(T), cudaMemcpyDeviceToHost);
    check_status(error, "Device memory copying error.");
    return elem;
}

template <typename T>
vector<T> device_memory<T>::copy_to_host() const
{
    vector<T> vec(size_);
    if (size_ != 0)
    {
        auto error = cudaMemcpy(vec.data(), d_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
        check_status(error, "Device memory copying error.");
    }

    return vec;
}
