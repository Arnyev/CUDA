#pragma once
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <chrono>

typedef unsigned long long ullong;
typedef unsigned int uint;

#define LOGOFILE "logof.jpg"

#define BOUND 100.0f
#define RADIUS 1
#define DRAWDISTSQR (RADIUS*RADIUS)
#define COLLISIONDIST (2*RADIUS+1.0f)

#define SPRINGFORCE (-0.2f)
#define GRAVITY (-0.00000005f)
#define DAMPING 0.012f
#define BOUNDARYFORCE 0.2f
#define SHEARFORCE 0.06f

#define DOWNSAMPLING 3
#define IMAGEW 1920
#define IMAGEH 1080
#define IMAGEWFULL (IMAGEW*DOWNSAMPLING)
#define IMAGEHFULL (IMAGEH*DOWNSAMPLING)

#define CELLSIZE (static_cast<int>(COLLISIONDIST))
#define CELLCOUNTX (IMAGEWFULL/CELLSIZE)
#define CELLCOUNTY (IMAGEHFULL/CELLSIZE)
#define CELLCOUNT (CELLCOUNTX * CELLCOUNTY)

#define FRAMEDIFF 400
#define FRAMECOUNT 9999

#define LOGOW 1016
#define LOGOH 856
#define DPPX 1
#define PARTICLECOUNTX (DPPX*LOGOW)
#define PARTICLECOUNTY (DPPX*LOGOH)

#define STARTINGDIST (COLLISIONDIST*1.1f)
#define STARTINGHEIGHT (BOUND+RADIUS)
#define STARTINGX ((IMAGEWFULL - 2 * BOUND - STARTINGDIST * PARTICLECOUNTX) / 2)

#define PARTICLECOUNT (PARTICLECOUNTX*PARTICLECOUNTY)

#define SPEEDFACTORY 0.0004f
#define SPEEDFACTORX 0.0004f
#define THREADCOUNT 16
#define BLOCKSIZE 256
#define GRIDDIM 2048

#define  THREAD_ID() ((blockIdx.x + blockIdx.y * static_cast<size_t>(gridDim.x) + static_cast<size_t>(gridDim.x) * gridDim.y * blockIdx.z)\
 * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.x * blockDim.y+ threadIdx.y * blockDim.x+ threadIdx.x)

#if __CUDACC__
#define kernel_init(...) <<<__VA_ARGS__>>>
#else
#define kernel_init(...)
#endif

#define GPU_ERRCHK(ans) { gpuAssert((ans),__FILE__,__LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		cudaDeviceReset();
		system("pause");
		exit(code);
	}
}

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

struct particle_data
{
	thrust::device_vector<float2> positions_stable;
	thrust::device_vector<float2> speeds_stable;
	thrust::device_vector<float2> positions_sort;
	thrust::device_vector<float2> speeds_sort;
	thrust::device_vector<uint> cell_starts;
	thrust::device_vector<uint> indices;
	thrust::device_vector<uint> particle_cell;
	thrust::device_vector<uchar3> logo;
	thrust::device_vector<uchar3> image;
	thrust::device_vector<uchar3> image_downsampled;

	explicit particle_data(const thrust::host_vector<uchar3>& logo_host);
};

struct measure
{
	template<typename F, typename ...Args>
	static std::chrono::microseconds::rep execution(F&& func, Args&&... args);

	template<typename F, typename ...Args>
	static float execution_gpu(F&& func, Args&&... args);
};

void put_circles(particle_data& data);
void downsample_image(const uchar3* input, uchar3* output);
void process_step(particle_data& particles);
void sort_particles(particle_data& particles);
