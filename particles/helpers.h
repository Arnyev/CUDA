#pragma once

#include <driver_types.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <sstream>
#include <chrono>
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

inline void gpu_assert(const cudaError_t code, const char *file, const int line)
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

    const auto free_db = static_cast<double>(free_byte);
    const auto total_db = static_cast<double>(total_byte);
    const double used_db = total_db - free_db;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

__host__ __device__ __forceinline__ size_t align_to(const size_t n, const size_t align)
{
    return (n + align - 1) / align * align;
}

#define  THREAD_ID() ((blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.x * blockDim.y+ threadIdx.y * blockDim.x+ threadIdx.x)

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
	num_blocks = num_blocks % GRIDDIM != 0 ? num_blocks / GRIDDIM + 1 : num_blocks / GRIDDIM;						\
	unsigned grid_z = static_cast<unsigned>(num_blocks);															\
	const dim3 threads(threads_x, 1, 1);																			\
	const dim3 blocks(grid_x, grid_y, grid_z);																		\
	function kernel_init(blocks, threads) (__VA_ARGS__);															\
	GPU_ERRCHK(cudaGetLastError());																					\
}

struct measure
{
    template<typename F, typename ...Args>
    static std::chrono::microseconds::rep execution(F&& func, Args&&... args)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>
            (std::chrono::high_resolution_clock::now() - start);
        return duration.count();
    }

    template<typename F, typename ...Args>
    static float execution_gpu(F&& func, Args&&... args)
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

        return milliseconds * 1000;
    }
};
