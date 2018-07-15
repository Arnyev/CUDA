#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "helper_math.h"
#include "thrust/sort.h"
#include <thrust/execution_policy.h>
#include "thrust/device_ptr.h"
#include <parameters.h> 

float2* d_speeds1 = 0;
float2* d_positions1 = 0;
float2* d_speeds2 = 0;
float2* d_positions2 = 0;

uint* d_indices = 0;
uint* d_hashes = 0;

uint* d_cellStarts = 0;
uint* d_cellEnds = 0;

__device__ void PutCircle(int middleIndex, uchar3* pixels, int threadNum, uchar4*logo);
__device__ float2 ProcessCollisions(int threadNum, float2* oldPositions, float2* oldSpeeds, uint *cellStarts, uint *cellEnds);
void radix_sort(unsigned int* keys, unsigned int* items, int count);

__global__ void ProcessForces(float2* oldPositions, float2* oldSpeeds, float2* newSpeeds, uint * gridParticleIndices, uint *cellStarts, uint *cellEnds)
{
	uint threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= PARTICLECOUNT)
		return;

	float2 force = ProcessCollisions(threadNum, oldPositions, oldSpeeds, cellStarts, cellEnds);
	float2 newSpeed = oldSpeeds[threadNum] + force;
	newSpeed.y += GRAVITY;

	uint originalIndex = gridParticleIndices[threadNum];

	newSpeeds[originalIndex] = newSpeed;
}

__global__ void CUDADrawPixels(uchar3 *pixels, float2* positions, float2* speeds, uchar4*logo, bool putCircle, float rightBound)
{
	uint threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= PARTICLECOUNT)
		return;

	float2 position = positions[threadNum];
	float2 speed = speeds[threadNum];

	if (position.x > rightBound)
	{
		speed.x += -1 * BOUNDARYFORCE*(position.x - rightBound);
	}

	if (position.x < BOUND)
	{
		speed.x += BOUNDARYFORCE * (BOUND - position.x);
	}

	if (position.y >  IMAGEHFULL - BOUND)
	{
		speed.y += -1 * BOUNDARYFORCE*(position.y - IMAGEHFULL + BOUND);
	}

	if (position.y < BOUND)
	{
		speed.y += BOUNDARYFORCE * (BOUND - position.y);
	}

	positions[threadNum] = position + speed;
	speeds[threadNum] = speed;

	int middleIndex = (int)position.x + IMAGEWFULL * (int)position.y;
	if (putCircle)
		PutCircle(middleIndex, pixels, threadNum, logo);
}

__device__ uint GetHashFromPosition(float2 p)
{
	uint gridPosx = floor(p.x  / CELLSIZE);
	uint gridPosy = floor(p.y  / CELLSIZE);

	gridPosx = gridPosx & (CELLCOUNTX - 1);
	gridPosy = gridPosy & (CELLCOUNTX - 1);

	return  __umul24(gridPosy, CELLCOUNTX) + gridPosx;
}

__device__ float2 ProcessCollisions(int threadNum, float2* oldPositions, float2* oldSpeeds, uint *cellStarts, uint *cellEnds)
{
	float2 force = { 0.0f,0.0f };

	float2 posA = oldPositions[threadNum];
	float2 velA = oldSpeeds[threadNum];
	uint cellIndex = GetHashFromPosition(posA);

	#pragma unroll
	for (int x = -1; x <= 1; x++)
	{
		#pragma unroll
		for (int y = -1; y <= 1; y++)
		{
			uint neighbourCellIndex = cellIndex + CELLCOUNTX * y + x;
			if (neighbourCellIndex >= CELLCOUNT)
				return force;

			uint neighbourCellStart = cellStarts[neighbourCellIndex];
			uint neighbourCellEnd = cellEnds[neighbourCellIndex];

			if (neighbourCellStart == 0xffffffff)
				continue;

			if (neighbourCellEnd > PARTICLECOUNT)
				return force;

			for (uint i = neighbourCellStart; i < neighbourCellEnd; i++)
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

					float2 relVel = velB - velA;

					float2 tanVel = relVel - (dot(relVel, norm) * norm);

					force += SPRINGFORCE * (COLLISIONDIST - dist) * norm;
					force += DAMPING * relVel;
					force += SHEARFORCE * tanVel;
				}
			}
		}
	}
	return force;
}

__global__ void CalcHashesD(float2* positions, uint* hashes, uint* indices)
{
	uint index = blockIdx.x* blockDim.x + threadIdx.x;

	if (index >= PARTICLECOUNT) return;

	uint hash = GetHashFromPosition(positions[index]);

	hashes[index] = hash;
	indices[index] = index;
}

__device__ void PutCircle(int middleIndex, uchar3* pixels, int threadNum, uchar4*logo)
{
	if (middleIndex<(RADIUS + 1)*IMAGEWFULL || middleIndex>(IMAGEWFULL*(IMAGEHFULL - RADIUS - 1)))
		return;

	int logoX = (threadNum % (DPPX * LOGOW)) / DPPX;
	int logoY = (threadNum / (LOGOW*DPPX)) / DPPX;
	int logoIndex = logoY * LOGOW + logoX;
	uchar4 tempColor = logo[logoIndex];

	//uchar4 tempColor = logo[threadNum];
	uchar3 color = { 0,0,0 };

	color.x = tempColor.x;
	color.y = tempColor.y;
	color.z = tempColor.z;

	//uchar3 color = { 0,0,0 };

#pragma unroll
	for (int i = -RADIUS; i <= RADIUS; i++)
#pragma unroll
		for (int j = -RADIUS; j <= RADIUS; j++)
			if (i * i + j * j <= DRAWDISTSQR)
				pixels[middleIndex + i * IMAGEWFULL + j] = color;
}

void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
}

__global__ void reorderDataAndFindCellStartD(
	uint*cellStart, uint*cellEnd,
	float2*sortedPos, float2 *sortedVel,
	uint *gridParticleHash, uint *gridParticleIndex,
	float2 *oldPos, float2 *oldVel)
{
	uint index = blockIdx.x* blockDim.x + threadIdx.x;

	if (index >= PARTICLECOUNT)
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

	if (index == PARTICLECOUNT - 1)
	{
		cellEnd[hash] = PARTICLECOUNT;
	}

	uint sortedIndex = gridParticleIndex[index];

	sortedPos[index] = oldPos[sortedIndex];
	sortedVel[index] = oldVel[sortedIndex];
}

void InitializeParticles()
{
	size_t arraySize = PARTICLECOUNT * sizeof(float2);
	size_t indicesSize = sizeof(uint)* PARTICLECOUNT;

	float2* positions = (float2*)malloc(arraySize);
	float2* speeds = (float2*)malloc(arraySize);
	memset(positions, 0, arraySize);
	memset(speeds, 0, arraySize);

	checkCudaErrors(cudaMalloc((void **)&d_cellEnds, sizeof(uint)*CELLCOUNT));
	checkCudaErrors(cudaMalloc((void **)&d_cellStarts, sizeof(uint)*CELLCOUNT));

	checkCudaErrors(cudaMalloc((void **)&d_indices, indicesSize));
	checkCudaErrors(cudaMalloc((void **)&d_hashes, indicesSize));

	checkCudaErrors(cudaMalloc((void **)&d_positions1, arraySize));
	checkCudaErrors(cudaMalloc((void **)&d_speeds1, arraySize));
	checkCudaErrors(cudaMalloc((void **)&d_positions2, arraySize));
	checkCudaErrors(cudaMalloc((void **)&d_speeds2, arraySize));

	srand(0);

	if (HEXAGONALSTART)
	{
		for (int i = 0; i < PARTICLECOUNTY/2; i++)
		{
			for (int j = 0; j < PARTICLECOUNTX; j++)
			{
				int index = 2*i * PARTICLECOUNTX + j;

				positions[index].x = STARTINGX + STARTINGDISTX * j;
				positions[index].y = STARTINGHEIGHT + STARTINGDISTY * i;
				speeds[index].x = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - 0.5f)*SPEEDFACTORX;
				speeds[index].y = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - 0.5f)*SPEEDFACTORY;
			}
			for (int j = 0; j < PARTICLECOUNTX; j++)
			{
				int index = (2 * i + 1) * PARTICLECOUNTX + j;

				positions[index].x = STARTINGX + STARTINGDISTX * j + STARTINGDIST * XDIFFPOS;
				positions[index].y = STARTINGHEIGHT + STARTINGDISTY * i + STARTINGDISTY / 2;
				speeds[index].x = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - 0.5f)*SPEEDFACTORX;
				speeds[index].y = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - 0.5f)*SPEEDFACTORY;
			}
		}
	}
	else
	{
		for (int i = 0; i < PARTICLECOUNTY; i++)
		{
			for (int j = 0; j < PARTICLECOUNTX; j++)
			{
				int index = i * PARTICLECOUNTX + j;
				positions[index].x = STARTINGX + STARTINGDIST * j;
				positions[index].y = STARTINGHEIGHT + STARTINGDIST * i;

				speeds[index].x = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - 0.5f)*SPEEDFACTORX;
				speeds[index].y = ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - 0.5f)*SPEEDFACTORY;
			}
		}
	}
	checkCudaErrors(cudaMemcpy(d_positions1, positions, arraySize, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_speeds1, speeds, arraySize, cudaMemcpyHostToDevice));

	free(speeds);
	free(positions);
}


void RunCUDA(uchar3 *d_destinationBitmap, uchar4 *d_logo, bool putCircle, float rightBound)
{
	uint numThreads, numBlocks;

	computeGridSize(PARTICLECOUNT, BLOCKSIZE, numBlocks, numThreads);

	CalcHashesD << < numBlocks, numThreads >> > (d_positions1, d_hashes, d_indices);
	getLastCudaError("CalcHashes execution failed");

	radix_sort(d_hashes, d_indices, PARTICLECOUNT);

	//thrust::sort_by_key(thrust::device_ptr<uint>(d_hashes), thrust::device_ptr<uint>(d_hashes + PARTICLECOUNT), thrust::device_ptr<uint>(d_indices));

	checkCudaErrors(cudaMemset(d_cellStarts, 0xffffffff, CELLCOUNT * sizeof(uint)));

	reorderDataAndFindCellStartD << < numBlocks, numThreads >> > (d_cellStarts, d_cellEnds, d_positions2, d_speeds2, d_hashes, d_indices, d_positions1, d_speeds1);
	getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

	ProcessForces << <numBlocks, numThreads >> > (d_positions2, d_speeds2, d_speeds1, d_indices, d_cellStarts, d_cellEnds);
	getLastCudaError("CUDADrawPixels kernel execution failed.\n");

	if (putCircle)
	{
		checkCudaErrors(cudaMemset(d_destinationBitmap, BACKGROUND, sizeof(uchar3)* IMAGEWFULL* IMAGEHFULL));
	}

	CUDADrawPixels << <numBlocks, numThreads >> > (d_destinationBitmap, d_positions1, d_speeds1, d_logo, putCircle,rightBound);
	getLastCudaError("CUDADrawPixels kernel execution failed.\n");
}

__global__ void DownsampleImageD(uchar3 *d_imageStart, uchar3 *d_imageResult)
{
	uint index = blockIdx.x* blockDim.x + threadIdx.x;
	if (index >= IMAGEW * IMAGEH)
		return;

	uint myRow = index / IMAGEW;
	uint myColumn = index % IMAGEW;
	uint inputIndexStart = DOWNSAMPLING * (IMAGEWFULL * myRow + myColumn);

	ushort3 colorTMP = { 0,0,0 };
	uchar3 color;

	#pragma unroll
	for (int j = 0; j < DOWNSAMPLING; j++)
	{
		#pragma unroll
		for (int k = 0; k < DOWNSAMPLING; k++)
		{
			uint indexP = inputIndexStart + j * IMAGEWFULL + k;
			colorTMP.x += d_imageStart[indexP].x;
			colorTMP.y += d_imageStart[indexP].y;
			colorTMP.z += d_imageStart[indexP].z;
		}
	}

	color.x = (unsigned char)(colorTMP.x >> LOGDOWNSAMPLINGSQR);
	color.y = (unsigned char)(colorTMP.y >> LOGDOWNSAMPLINGSQR);
	color.z = (unsigned char)(colorTMP.z >> LOGDOWNSAMPLINGSQR);

	d_imageResult[index] = color;
}

void DownsampleImage(uchar3 *d_imageStart, uchar3 *d_imageResult)
{
	uint numThreads, numBlocks;

	computeGridSize(IMAGEW*IMAGEH, BLOCKSIZE, numBlocks, numThreads);

	DownsampleImageD << <numBlocks, numThreads >> > (d_imageStart, d_imageResult);
	getLastCudaError("CUDADrawPixels kernel execution failed.\n");
}

