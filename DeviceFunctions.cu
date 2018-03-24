#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "device_launch_parameters.h"

__global__ void CUDADrawPixels(uchar4 *pixels)
{
	int y = threadIdx.x + blockIdx.x * 1000;
	pixels[y] = { (unsigned char)255,(unsigned char)255,(unsigned char)255,(unsigned char)255 };
}

void RunCUDA(uchar4 *dst)
{
	dim3 threads(1000, 1, 1);
	dim3 grid(100);

	CUDADrawPixels << <grid, threads >> >(dst);

	getLastCudaError("Mandelbrot1 kernel execution failed.\n");
}
