#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <math.h>
#include <cooperative_groups.h>
#include "helper_math.h"
#include "math_constants.h"

#define BOUND 100.0f
#define BOUNDINT (int)BOUND
#define RADIUS 2
#define CIRCLECOUNT 300
#define COLLISIONDIST 10.0f
#define SPRINGFORCE 0.5f
#define GRAVITY -0.1f
#define DAMPING 0.02f
#define BOUNDARYFORCE 0.5f
#define SHEARFORCE 0.2f

float2* d_speeds = 0;
float2* d_positions = 0;
float2* h_speeds = 0;
float2* h_positions = 0;

int lastHeight = 0;
int lastWidth = 0;
bool initialized = false;

__device__ void PutCircle(int middleIndex, int imageWidth, uchar4* pixels, int threadNum);
__device__ void ProcessCollisions(int threadNum, float2* positions, float2* speeds);

__global__ void CUDADrawPixels(uchar4 *pixels, int imageWidth, int imageHeight, float2* positions, float2* speeds)
{
	int threadNum = threadIdx.x;
	if (threadNum >= CIRCLECOUNT)
		return;

	positions[threadNum].x += speeds[threadNum].x;
	positions[threadNum].y += speeds[threadNum].y;

	if (positions[threadNum].x > imageWidth - BOUND)
	{
		speeds[threadNum].x += -1 * BOUNDARYFORCE*(positions[threadNum].x - imageWidth + BOUND);
	}

	if (positions[threadNum].x < BOUND)
	{
		speeds[threadNum].x += BOUNDARYFORCE * (BOUND - positions[threadNum].x);
	}

	if (positions[threadNum].y > imageHeight - BOUND)
	{
		speeds[threadNum].y += -1 * BOUNDARYFORCE*(positions[threadNum].y - imageHeight + BOUND);
	}

	if (positions[threadNum].y < BOUND)
	{
		speeds[threadNum].y += BOUNDARYFORCE * (BOUND - positions[threadNum].y);
	}

	int middleIndex = (int)positions[threadNum].x + imageWidth * (int)positions[threadNum].y;
	PutCircle(middleIndex, imageWidth, pixels, threadNum);

	speeds[threadNum].y += GRAVITY;

	ProcessCollisions(threadNum, positions, speeds);
}

__device__ void ProcessCollisions( int threadNum, float2* positions, float2* speeds)
{
	float2 force = { 0.0f,0.0f };

	float2 posA = positions[threadNum];
	float2 velA = speeds[threadNum];

	for (int i = 0; i < CIRCLECOUNT; i++)
	{
		if (i == threadNum)
			continue;

		float2 posB = positions[i];
		float2 velB = speeds[i];
		float2 relPos = posB - posA;

		float dist = length(relPos);

		if (dist < COLLISIONDIST)
		{
			float2 norm = relPos / dist;

			// relative velocity
			float2 relVel = velB - velA;
			float lenVel=length(relVel);
			float2 relVelN = relVel / lenVel;

			// relative tangential velocity
			float2 tanVel = relVel - (dot(relVel, norm) * norm);

			// spring force
			force = -SPRINGFORCE*(COLLISIONDIST - dist) * norm;
			// dashpot (damping) force
			force += DAMPING*relVel;
			// tangential shear force
			force += SHEARFORCE*tanVel;
			// attraction
		}
	}
	__syncthreads();
	speeds[threadNum] += force;
}

__device__ void PutCircle(int middleIndex, int imageWidth, uchar4* pixels, int threadNum)
{
	uchar4 colorWhite = { (unsigned char)255,(unsigned char)threadNum ,(unsigned char)255 ,(unsigned char)255 };

	pixels[middleIndex - (imageWidth * 3) - 1] = colorWhite;
	pixels[middleIndex - (imageWidth * 3)] = colorWhite;
	pixels[middleIndex - (imageWidth * 3) + 1] = colorWhite;
	pixels[middleIndex - (imageWidth * 2) - 2] = colorWhite;
	pixels[middleIndex - (imageWidth * 2) - 1] = colorWhite;
	pixels[middleIndex - (imageWidth * 2)] = colorWhite;
	pixels[middleIndex - (imageWidth * 2) + 1] = colorWhite;
	pixels[middleIndex - (imageWidth * 2) + 2] = colorWhite;
	pixels[middleIndex - imageWidth - 3] = colorWhite;
	pixels[middleIndex - imageWidth - 2] = colorWhite;
	pixels[middleIndex - imageWidth - 1] = colorWhite;
	pixels[middleIndex - imageWidth] = colorWhite;
	pixels[middleIndex - imageWidth + 1] = colorWhite;
	pixels[middleIndex - imageWidth + 2] = colorWhite;
	pixels[middleIndex - imageWidth + 3] = colorWhite;
	pixels[middleIndex - 3] = colorWhite;
	pixels[middleIndex - 2] = colorWhite;
	pixels[middleIndex - 1] = colorWhite;
	pixels[middleIndex] = colorWhite;
	pixels[middleIndex + 1] = colorWhite;
	pixels[middleIndex + 2] = colorWhite;
	pixels[middleIndex + 3] = colorWhite;
	pixels[middleIndex + imageWidth - 3] = colorWhite;
	pixels[middleIndex + imageWidth - 2] = colorWhite;
	pixels[middleIndex + imageWidth - 1] = colorWhite;
	pixels[middleIndex + imageWidth] = colorWhite;
	pixels[middleIndex + imageWidth + 1] = colorWhite;
	pixels[middleIndex + imageWidth + 2] = colorWhite;
	pixels[middleIndex + imageWidth + 3] = colorWhite;
	pixels[middleIndex + (imageWidth * 2) - 2] = colorWhite;
	pixels[middleIndex + (imageWidth * 2) - 1] = colorWhite;
	pixels[middleIndex + (imageWidth * 2)] = colorWhite;
	pixels[middleIndex + (imageWidth * 2) + 1] = colorWhite;
	pixels[middleIndex + (imageWidth * 2) + 2] = colorWhite;
	pixels[middleIndex + (imageWidth * 3) - 1] = colorWhite;
	pixels[middleIndex + (imageWidth * 3)] = colorWhite;
	pixels[middleIndex + (imageWidth * 3) + 1] = colorWhite;
}

void RunCUDA(uchar4 *d_destinationBitmap, int imageWidth, int imageHeight)
{
	const size_t arraySize = CIRCLECOUNT * sizeof(float2);
	int h = 0;
	int k = h;
	if (!initialized)
	{
		initialized = true;

		h_positions = (float2*)malloc(arraySize);
		h_speeds = (float2*)malloc(arraySize);
		memset(h_positions, 0, arraySize);
		memset(h_speeds, 0, arraySize);

		checkCudaErrors(cudaMalloc((void **)&d_positions, arraySize));
		checkCudaErrors(cudaMalloc((void **)&d_speeds, arraySize));
	}

	if (lastWidth != imageWidth || lastHeight != imageHeight)
	{
		lastWidth = imageWidth;
		lastHeight = imageHeight;

		srand(0);
		float dividerX = ((float)RAND_MAX) / (imageWidth - 2 * BOUNDINT);
		float dividerY = ((float)RAND_MAX) / (imageHeight / 10 - 2 * BOUNDINT);

		for (int i = 0; i < CIRCLECOUNT; i++)
		{
			h_positions[i].x = 10 * (i % ((imageWidth - 2 * BOUNDINT) / 10)) + BOUNDINT;
			h_positions[i].y = 10 * (i / ((imageWidth - 2 * BOUNDINT) / 10)) + BOUNDINT;
			h_speeds[i].x = 0;
			h_speeds[i].y = 0;
			//h_positions[i].x = abs((float)rand() / dividerX) + BOUND;
			//h_positions[i].y = abs((float)rand() / dividerY) + BOUND;
			h_speeds[i].x = (float)rand() / (dividerX * 2000000);
			//h_speeds[i].y = (float)rand() / (dividerY * 2000000);
		}

		checkCudaErrors(cudaMemcpy(d_positions, h_positions, arraySize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_speeds, h_speeds, arraySize, cudaMemcpyHostToDevice));
	}

	checkCudaErrors(cudaMemset(d_destinationBitmap, 0, sizeof(uchar4)* imageWidth* imageHeight));

	dim3 threads(CIRCLECOUNT, 1, 1);
	dim3 grid(1);

	CUDADrawPixels << <grid, threads >> > (d_destinationBitmap, imageWidth, imageHeight, d_positions, d_speeds);

	getLastCudaError("CUDADrawPixels kernel execution failed.\n");
}
