#pragma once
#include <driver_types.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#define GPU_ERRCHK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

inline void check_status(cudaError_t status, const std::string& message)
{
	if (status == cudaSuccess)
		return;

	const auto error_string = cudaGetErrorString(status);
	std::cout << message << " " << error_string;
	throw std::runtime_error(message + error_string);
}

inline void gpu_assert(const cudaError_t code, const char* file, const int line)
{
	if (code == cudaSuccess)
		return;

	std::stringstream stream;
	stream << "Error in file " << file << ", line " << line << '\n';
	check_status(code, stream.str());
}

inline void print_memory_usage()
{
	size_t free_byte;
	size_t total_byte;

	const auto status = cudaMemGetInfo(&free_byte, &total_byte);
	check_status(status, "Error during getting memory usage info.");

	const auto used = total_byte - free_byte;

	std::cout << "GPU memory usage: used = " << used / 1024 / 1024 << " MB, free = " << free_byte / 1024 / 1024 << " MB, total = " << total_byte / 1024 / 1024 << " MB\n";
}

#define  THREAD_ID() ((blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x)

#if __CUDACC__
#define kernel_init(...) <<<__VA_ARGS__>>>
#else
#define kernel_init(...)
#endif

#define STARTKERNEL(function,thread_count,...)																		\
{																													\
	const unsigned threads_x = static_cast<unsigned>(BLOCKSIZE < thread_count ? BLOCKSIZE : thread_count);			\
	unsigned num_blocks = thread_count % BLOCKSIZE != 0 ? thread_count / BLOCKSIZE + 1 : thread_count / BLOCKSIZE;	\
	const unsigned grid_x = static_cast<unsigned>(GRIDDIM < num_blocks ? GRIDDIM : num_blocks);						\
	num_blocks = num_blocks % GRIDDIM != 0 ? num_blocks / GRIDDIM + 1 : num_blocks / GRIDDIM;						\
	const unsigned grid_y = static_cast<unsigned>(GRIDDIM < num_blocks ? GRIDDIM : num_blocks);						\
	const dim3 threads(threads_x, 1, 1);																			\
	const dim3 blocks(grid_x, grid_y, 1);																		\
	function kernel_init(blocks, threads) (__VA_ARGS__);															\
	GPU_ERRCHK(cudaGetLastError());																					\
}

struct measure
{
	template<typename F, typename ...Args>
	static float execution_gpu(F&& func, Args&& ... args)
	{
		cudaEvent_t start;
		cudaEvent_t stop;
		float milliseconds = 0;

		GPU_ERRCHK(cudaEventCreate(&start));
		GPU_ERRCHK(cudaEventCreate(&stop));
		GPU_ERRCHK(cudaEventRecord(start));

		std::forward<decltype(func)>(func)(std::forward<Args>(args)...);

		GPU_ERRCHK(cudaEventRecord(stop));
		GPU_ERRCHK(cudaEventSynchronize(stop));
		GPU_ERRCHK(cudaEventElapsedTime(&milliseconds, start, stop));
		GPU_ERRCHK(cudaEventDestroy(start));
		GPU_ERRCHK(cudaEventDestroy(stop));

		return milliseconds;
	}
};
