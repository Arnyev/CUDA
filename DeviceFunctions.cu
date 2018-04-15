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
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#define BOUND 20.0f
#define BOUNDINT (int)BOUND
#define RADIUS 1
#define COLLISIONDIST (float)(2*RADIUS+1)
#define COLLISIONDISTINT (2*RADIUS+2)
#define SPRINGFORCE 0.5f
#define GRAVITY -0.000005f
#define DAMPING 0.002f
#define BOUNDARYFORCE 0.5f
#define SHEARFORCE 0.01f
#define CELLSIZE 8
#define LOGGRIDSIZE 8
#define CELLCOUNTX 256
#define CELLCOUNTY 128
#define CELLCOUNT (CELLCOUNTX * CELLCOUNTY )

float2* d_speeds1 = 0;
float2* d_positions1 = 0;
float2* d_speeds2 = 0;
float2* d_positions2 = 0;

uint* d_indices = 0;
uint* d_hashes = 0;

uint* d_cellStarts = 0;
uint* d_cellEnds = 0;

int lastHeight = 0;
int lastWidth = 0;

__device__ void PutCircle(int middleIndex, int imageWidth, uchar4* pixels, int threadNum, uchar4*logo);
__device__ float2 ProcessCollisions(int threadNum, float2* oldPositions, float2* oldSpeeds, uint *cellStarts, uint *cellEnds);

__global__ void ProcessForces(float2* oldPositions, float2* oldSpeeds, float2* newSpeeds, int particleCount, uint * gridParticleIndices, uint *cellStarts, uint *cellEnds)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= particleCount)
		return;

	float2 force = ProcessCollisions(threadNum, oldPositions, oldSpeeds, cellStarts, cellEnds);
	float2 newSpeed = oldSpeeds[threadNum] + force;
	newSpeed.y += GRAVITY;

	int originalIndex = gridParticleIndices[threadNum];

	newSpeeds[originalIndex] = newSpeed;
}

__global__ void CUDADrawPixels(uchar4 *pixels, int imageWidth, int imageHeight,float2* positions, float2* speeds, uchar4*logo, int particleCount)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= particleCount)
		return;

	float2 position = positions[threadNum];
	float2 speed = speeds[threadNum];

	if (position.x > imageWidth - BOUND)
	{
		speed.x += -1 * BOUNDARYFORCE*(position.x - imageWidth + BOUND);
	}

	if (position.x < BOUND)
	{
		speed.x += BOUNDARYFORCE * (BOUND - position.x);
	}

	if (position.y > imageHeight - BOUND)
	{
		speed.y += -1 * BOUNDARYFORCE*(position.y - imageHeight + BOUND);
	}

	if (position.y < BOUND)
	{
		speed.y += BOUNDARYFORCE * (BOUND - position.y);
	}

	positions[threadNum] = position + speed;
	speeds[threadNum] = speed;

	int middleIndex = (int)position.x + imageWidth * (int)position.y;
	PutCircle(middleIndex, imageWidth, pixels, threadNum, logo);
}

__device__ uint GetHashFromPosition(float2 p)
{
	return (((uint)floor(p.y / CELLSIZE)) << LOGGRIDSIZE) | ((uint)floor(p.x / CELLSIZE));
}

__device__ float2 ProcessCollisions( int threadNum, float2* oldPositions, float2* oldSpeeds, uint *cellStarts, uint *cellEnds)
{
	float2 force = { 0.0f,0.0f };

	float2 posA = oldPositions[threadNum];
	float2 velA = oldSpeeds[threadNum];
	int cellIndex = GetHashFromPosition(posA);

	for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			int neighbourCellIndex = cellIndex + CELLCOUNTX * y + x;
			if (neighbourCellIndex < 0 || neighbourCellIndex >= CELLCOUNT)
				continue;

			int neighbourCellStart = cellStarts[neighbourCellIndex];
			int neighbourCellEnd = cellEnds[neighbourCellIndex];

			if (neighbourCellStart == 0xffffffff)
				continue;

			for (int i = neighbourCellStart; i < neighbourCellEnd; i++)
			{
				if (i == threadNum)
					continue;

				float2 posB = oldPositions[i];
				float2 velB = oldSpeeds[i];
				float2 relPos = posB - posA;

				float dist = length(relPos);

				if (dist < COLLISIONDIST)
				{
					float2 norm = relPos / dist;

					// relative velocity
					float2 relVel = velB - velA;
					float lenVel = length(relVel);
					float2 relVelN = relVel / lenVel;

					// relative tangential velocity
					float2 tanVel = relVel - (dot(relVel, norm) * norm);

					// spring force
					force -= SPRINGFORCE * (COLLISIONDIST - dist) * norm;
					force += DAMPING*relVel;
					// tangential shear force
					force += SHEARFORCE*tanVel;
					// attraction
				}
			}
		}
	}
	return force;
}

__device__ int2 calcGridPos(float2 p)
{
	int2 gridPos;
	gridPos.x = floor(p.x / CELLSIZE);
	gridPos.y = floor(p.y / CELLSIZE);
	return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int2 gridPos)
{
	gridPos.x = gridPos.x & (CELLCOUNTX - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (CELLCOUNTY - 1);
	return __umul24(gridPos.y, CELLCOUNTX) + gridPos.x;
}

__global__ void CalcHashesD(int particleCount, float2* positions, uint* hashes, uint* indices )
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= particleCount) return;

	volatile float2 p = positions[index];

	// get address in grid
	int2 gridPos = calcGridPos(make_float2(p.x, p.y));
	uint hash = calcGridHash(gridPos);

	// store grid hash and particle index
	hashes[index] = hash;
	indices[index] = index;
}

__device__ void PutCircle(int middleIndex, int imageWidth, uchar4* pixels, int threadNum, uchar4*logo)
{
	uchar4 color = logo[threadNum];

	if (RADIUS >= 3)
	{
		pixels[middleIndex - (imageWidth * 3) - 1] = color;
		pixels[middleIndex - (imageWidth * 3)] = color;
		pixels[middleIndex - (imageWidth * 3) + 1] = color;
		pixels[middleIndex - (imageWidth * 2) - 2] = color;
	}
	if (RADIUS >= 2)
	{
		pixels[middleIndex - (imageWidth * 2) - 1] = color;
		pixels[middleIndex - (imageWidth * 2)] = color;
		pixels[middleIndex - (imageWidth * 2) + 1] = color;
	}
	if (RADIUS >= 3)
	{
		pixels[middleIndex - (imageWidth * 2) + 2] = color;
		pixels[middleIndex - imageWidth - 3] = color;
	}
	if (RADIUS >= 2)
	{
		pixels[middleIndex - imageWidth - 2] = color;
		pixels[middleIndex - imageWidth - 1] = color;
	}
	pixels[middleIndex - imageWidth] = color;
	if (RADIUS >= 2)
	{
		pixels[middleIndex - imageWidth + 1] = color;
		pixels[middleIndex - imageWidth + 2] = color;
	}
	if (RADIUS >= 3)
	{
		pixels[middleIndex - imageWidth + 3] = color;
		pixels[middleIndex - 3] = color;
	}
	if (RADIUS >= 2)
	{
		pixels[middleIndex - 2] = color;
	}
	pixels[middleIndex - 1] = color;
	pixels[middleIndex] = color;
	pixels[middleIndex + 1] = color;
	if (RADIUS >= 2)
	{
		pixels[middleIndex + 2] = color;
	}
	if (RADIUS >= 3)
	{
		pixels[middleIndex + 3] = color;
		pixels[middleIndex + imageWidth - 3] = color;
	}
	if (RADIUS >= 2)
	{
		pixels[middleIndex + imageWidth - 2] = color;
		pixels[middleIndex + imageWidth - 1] = color;
	}
	pixels[middleIndex + imageWidth] = color;
	if (RADIUS >= 2)
	{
		pixels[middleIndex + imageWidth + 1] = color;
		pixels[middleIndex + imageWidth + 2] = color;
	}
	if (RADIUS >= 3)
	{
		pixels[middleIndex + imageWidth + 3] = color;
		pixels[middleIndex + (imageWidth * 2) - 2] = color;
	}
	if (RADIUS >= 2)
	{
		pixels[middleIndex + (imageWidth * 2) - 1] = color;
		pixels[middleIndex + (imageWidth * 2)] = color;
		pixels[middleIndex + (imageWidth * 2) + 1] = color;
	}
	if (RADIUS >= 3)
	{
		pixels[middleIndex + (imageWidth * 2) + 2] = color;
		pixels[middleIndex + (imageWidth * 3) - 1] = color;
		pixels[middleIndex + (imageWidth * 3)] = color;
		pixels[middleIndex + (imageWidth * 3) + 1] = color;
	}
}

void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
}

void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
{
	thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
		thrust::device_ptr<uint>(dGridParticleHash + numParticles),
		thrust::device_ptr<uint>(dGridParticleIndex));
}

void CalcHashes(int particleCount, float2* positions, uint* hashes, uint* indices)
{
	uint numThreads, numBlocks;
	computeGridSize(particleCount, 1024, numBlocks, numThreads);

	// execute the kernel
	CalcHashesD << < numBlocks, numThreads >> > (particleCount, positions, hashes, indices);

	// check if kernel invocation generated an error
	getLastCudaError("Kernel execution failed");
}

__global__ void reorderDataAndFindCellStartD(
	uint*cellStart, uint*cellEnd,
	float2*sortedPos, float2 *sortedVel,
	uint *gridParticleHash, uint *gridParticleIndex,
	float2 *oldPos, float2 *oldVel,
	uint numParticles)
{
	uint index = blockIdx.x* blockDim.x + threadIdx.x;

	if (index >= numParticles)
		return;

	uint hash = gridParticleHash[index];
	uint lastHash = 0;

	if (index > 0)
		lastHash = gridParticleHash[index - 1];

	if (index == 0 || hash != lastHash)
	{
		cellStart[hash] = index;

		if (index > 0)
			cellEnd[lastHash] = index;
	}

	if (index == numParticles - 1)
	{
		cellEnd[hash] = index + 1;
	}

	uint sortedIndex = gridParticleIndex[index];

	sortedPos[index] = oldPos[sortedIndex];
	sortedVel[index] = oldVel[sortedIndex];
}

void ReorderDataAndFindCellStart(
	uint  *cellStart, uint  *cellEnd,
	float2 *sortedPos, float2 *sortedVel,
	uint  *gridParticleHash, uint  *gridParticleIndex,
	float2 *oldPos, float2 *oldVel,
	uint   numParticles, uint numCells)
{
	uint numThreads, numBlocks;
	computeGridSize(numParticles, 1024, numBlocks, numThreads);

	checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells * sizeof(uint)));

	reorderDataAndFindCellStartD << < numBlocks, numThreads >> > (cellStart, cellEnd, sortedPos, sortedVel, gridParticleHash, gridParticleIndex, oldPos, oldVel, numParticles);

	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
}

void InitializeParticles(size_t arraySize, int logoWidth, int logoHeight)
{
	float2* positions = (float2*)malloc(arraySize);
	float2* speeds = (float2*)malloc(arraySize);
	memset(positions, 0, arraySize);
	memset(speeds, 0, arraySize);

	if (d_positions1)
	{
		cudaFree(d_positions1);
		cudaFree(d_speeds1);
		cudaFree(d_positions2);
		cudaFree(d_speeds2);
		cudaFree(d_hashes);
		cudaFree(d_indices);
	}

	size_t indicesSize = sizeof(uint)*arraySize / (sizeof(float2)) * 8;

	checkCudaErrors(cudaMalloc((void **)&d_positions1, arraySize));
	checkCudaErrors(cudaMalloc((void **)&d_speeds1, arraySize));
	checkCudaErrors(cudaMalloc((void **)&d_positions2, arraySize));
	checkCudaErrors(cudaMalloc((void **)&d_speeds2, arraySize));
	checkCudaErrors(cudaMalloc((void **)&d_indices, indicesSize));
	checkCudaErrors(cudaMalloc((void **)&d_hashes, indicesSize));

	srand(0);

	for (int i = 0; i < logoHeight; i++)
	{
		for (int j = 0; j < logoWidth; j++)
		{
			int index = i * logoWidth + j;
			positions[index].x = BOUND + COLLISIONDISTINT * j;
			positions[index].y = BOUND + COLLISIONDISTINT * i;
			speeds[index].x = 0;
			speeds[index].y = 0;
			speeds[index].x = (float)rand() / 20000000;
		}
	}
	checkCudaErrors(cudaMemcpy(d_positions1, positions, arraySize, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_speeds1, speeds, arraySize, cudaMemcpyHostToDevice));

	free(speeds);
	free(positions);
}

void RunCUDA(uchar4 *d_destinationBitmap, uchar4 *d_logo, int logoWidth, int logoHeight, int imageWidth, int imageHeight)
{
	static size_t arraySize = 0;
	int particleCount = logoHeight * logoWidth;

	if (arraySize != particleCount * sizeof(float2))
	{
		arraySize = particleCount * sizeof(float2);
		InitializeParticles(arraySize, logoWidth, logoHeight);
	}

	if (!d_cellStarts)
	{
		checkCudaErrors(cudaMalloc((void **)&d_cellEnds, sizeof(uint)*CELLCOUNT));
		checkCudaErrors(cudaMalloc((void **)&d_cellStarts, sizeof(uint)*CELLCOUNT));
	}

	checkCudaErrors(cudaMemset(d_destinationBitmap, 0, sizeof(uchar4)* imageWidth* imageHeight));

	uint numThreads, numBlocks;
	computeGridSize(particleCount, 256, numBlocks, numThreads);

	CalcHashes(particleCount, d_positions1, d_hashes, d_indices);

	sortParticles(d_hashes, d_indices, particleCount);

	ReorderDataAndFindCellStart(d_cellStarts, d_cellEnds, d_positions2, d_speeds2, d_hashes, d_indices, d_positions1, d_speeds1, particleCount, CELLCOUNT);

	ProcessForces << <numBlocks, numThreads >> > (d_positions2, d_speeds2, d_speeds1, particleCount, d_indices, d_cellStarts, d_cellEnds);

	getLastCudaError("CUDADrawPixels kernel execution failed.\n");

	CUDADrawPixels << <numBlocks, numThreads >> > (d_destinationBitmap, imageWidth, imageHeight, d_positions1, d_speeds1, d_logo, particleCount);

	getLastCudaError("CUDADrawPixels kernel execution failed.\n");
}
