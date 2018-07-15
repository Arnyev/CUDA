#pragma once

#include <thrust/system/cuda/detail/core/util.h>
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>
#include <helper_cuda.h>

void radix_sort(unsigned int* keys, unsigned int* items, int count)
{
	void*        d_temp_storage = NULL;
	size_t       temp_storage_bytes = 0;
	cudaStream_t stream = (cudaStream_t)0;
	bool         debug_sync = THRUST_DEBUG_SYNC_FLAG;

	thrust::cuda_cub::cub::DoubleBuffer<unsigned int>  keys_buffer(keys, NULL);
	thrust::cuda_cub::cub::DoubleBuffer<unsigned int> items_buffer(items, NULL);

	int keys_count = count;
	int items_count = count;

	cudaError_t status;

	status = thrust::cuda_cub::cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys_buffer, items_buffer,
		count, 0, static_cast<int>(sizeof(unsigned int) * 8), stream, debug_sync);

	thrust::cuda_cub::throw_on_error(status, "radix_sort: failed on 1st step");

	size_t keys_temp_storage = thrust::cuda_cub::core::align_to(sizeof(unsigned int) * keys_count, 128);
	size_t items_temp_storage = thrust::cuda_cub::core::align_to(sizeof(unsigned int) * items_count, 128);

	size_t temp_storage_total = keys_temp_storage + items_temp_storage + temp_storage_bytes;

	checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_total));

	keys_buffer.d_buffers[1] = (unsigned int*)d_temp_storage;
	items_buffer.d_buffers[1] = (unsigned int*)((char*)d_temp_storage + keys_temp_storage);

	void* d_temp_storage1 = (char*)d_temp_storage + keys_temp_storage + items_temp_storage;

	status = thrust::cuda_cub::cub::DeviceRadixSort::SortPairs(d_temp_storage1, temp_storage_bytes, keys_buffer, items_buffer,
		count, 0, static_cast<int>(sizeof(unsigned int) * 8), stream, debug_sync);

	thrust::cuda_cub::throw_on_error(status, "radix_sort: failed on 2nd step");

	if (keys_buffer.selector != 0)
		checkCudaErrors(cudaMemcpy(keys, keys_buffer.d_buffers[1], sizeof(unsigned int)*keys_count, cudaMemcpyDeviceToDevice));

	if (items_buffer.selector != 0)
		checkCudaErrors(cudaMemcpy(items, items_buffer.d_buffers[1], sizeof(unsigned int)*items_count, cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaFree(d_temp_storage));
}
