#pragma once
#include <crt/host_defines.h>
#include <iterator>

template<typename T>
struct device_reverse_iterator
{
    T* d_ptr;

    using value_type = T;
    using pointer = T * ;
    using reference = T & ;
    using difference_type = ptrdiff_t;
    using iterator_category = std::_Really_trivial_ptr_iterator_tag;

    __host__ __device__ __forceinline__ explicit device_reverse_iterator(T* d_ptr) : d_ptr{ d_ptr } {}

    __device__ __forceinline__ device_reverse_iterator& operator++()
    {
        --d_ptr;
        return *this;
    }

    __device__ __forceinline__ T& operator*()
    {
        return *d_ptr;
    }

    __device__ __forceinline__ T& operator [] (const int index)
    {
        return  *(d_ptr - index);
    }

    __device__ __forceinline__ device_reverse_iterator operator + (const int val)
    {
        return device_reverse_iterator(d_ptr - val);
    }
};
