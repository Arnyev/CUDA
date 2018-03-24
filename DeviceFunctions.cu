#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "device_launch_parameters.h"
#define BOUND 10

float2* d_accelerations = 0;
float2* d_speeds = 0;
float2* d_positions = 0;
float2* h_accelerations = 0;
float2* h_speeds = 0;
float2* h_positions = 0;
int lastHeight = 0;
int lastWidth = 0;
bool initialized = false;
int circleCount = 100;

__global__ void CUDADrawPixels(uchar4 *pixels, int imageWidth, int imageHeight, float2* positions, float2* speeds, float2* accelerations)
{
	int threadNum = threadIdx.x;

	positions[threadNum].x += speeds[threadNum].x;
	if (positions[threadNum].x > (imageWidth - BOUND))
	{
		positions[threadNum].x -= 2 * (positions[threadNum].x - imageWidth + BOUND);
		speeds[threadNum].x *= -1;
		accelerations[threadNum].x *= -1;
	}
	if (positions[threadNum].x < BOUND)
	{
		positions[threadNum].x -= BOUND;
		positions[threadNum].x *= -1;
		positions[threadNum].x += BOUND;
		speeds[threadNum].x *= -1;
		accelerations[threadNum].x *= -1;
	}

	positions[threadNum].y += speeds[threadNum].y;
	if (positions[threadNum].y > (imageHeight - BOUND))
	{
		positions[threadNum].y -= 2 * (positions[threadNum].y - imageHeight + BOUND);
		speeds[threadNum].y *= -1;
		accelerations[threadNum].y *= -1;
	}

	if (positions[threadNum].y < BOUND)
	{
		positions[threadNum].y -= BOUND;
		positions[threadNum].y *= -1;
		positions[threadNum].y += BOUND;
		speeds[threadNum].y *= -1;
		accelerations[threadNum].y *= -1;
	}

	speeds[threadNum].x += accelerations[threadNum].x;
	speeds[threadNum].y += accelerations[threadNum].y;

	int arrayIndex = (int)positions[threadNum].x + imageWidth * (int)positions[threadNum].y;

	pixels[arrayIndex - imageWidth] = { (unsigned char)255,(unsigned char)255,(unsigned char)255,(unsigned char)255 };
	pixels[arrayIndex - 1] = { (unsigned char)255,(unsigned char)255,(unsigned char)255,(unsigned char)255 };
	pixels[arrayIndex] = { (unsigned char)255,(unsigned char)255,(unsigned char)255,(unsigned char)255 };
	pixels[arrayIndex + 1] = { (unsigned char)255,(unsigned char)255,(unsigned char)255,(unsigned char)255 };
	pixels[arrayIndex + imageWidth] = { (unsigned char)255,(unsigned char)255,(unsigned char)255,(unsigned char)255 };
}

void RunCUDA(uchar4 *d_destinationBitmap, int imageWidth, int imageHeight)
{
	checkCudaErrors(cudaMemset(d_destinationBitmap, 0, sizeof(uchar4)* imageWidth* imageHeight));

	const size_t arraySize = circleCount * sizeof(float2);

	if (!initialized)
	{
		initialized = true;

		h_positions = (float2*)malloc(arraySize);
		h_speeds = (float2*)malloc(arraySize);
		h_accelerations = (float2*)malloc(arraySize);

		checkCudaErrors(cudaMalloc((void **)&d_positions, arraySize));
		checkCudaErrors(cudaMalloc((void **)&d_speeds, arraySize));
		checkCudaErrors(cudaMalloc((void **)&d_accelerations, arraySize));
	}

	if (lastWidth != imageWidth || lastHeight != imageHeight)
	{
		lastWidth = imageWidth;
		lastHeight = imageHeight;

		srand(0);
		float dividerX = ((float)RAND_MAX) / (imageWidth - 1);
		float dividerY = ((float)RAND_MAX) / (imageHeight - 10);

		for (int i = 0; i < circleCount; i++)
		{
			h_positions[i].x = abs((float)rand() / dividerX);
			h_positions[i].y = abs((float)rand() / dividerY);
			h_speeds[i].x = (float)rand() / (dividerX * 10000);
			h_speeds[i].y = (float)rand() / (dividerY * 10000);
			h_accelerations[i].x = (float)rand() / (dividerX * 1000000);
			h_accelerations[i].y = (float)rand() / (dividerY * 1000000);
		}

		checkCudaErrors(cudaMemcpy(d_positions, h_positions, arraySize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_speeds, h_speeds, arraySize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_accelerations, h_accelerations, arraySize, cudaMemcpyHostToDevice));
	}

	dim3 threads(circleCount, 1, 1);
	dim3 grid(1);

	CUDADrawPixels << <grid, threads >> > (d_destinationBitmap, imageWidth, imageHeight, d_positions, d_speeds, d_accelerations);

	getLastCudaError("CUDADrawPixels kernel execution failed.\n");
}
