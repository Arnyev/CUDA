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
#include <sys\timeb.h> 
#include <parameters.h> 

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

__device__ void PutCircle(int middleIndex, uchar3* pixels, int threadNum, uchar4*logo);
__device__ float2 ProcessCollisions(int threadNum, float2* oldPositions, float2* oldSpeeds, uint *cellStarts, uint *cellEnds);

__global__ void ProcessForces(float2* oldPositions, float2* oldSpeeds, float2* newSpeeds, uint * gridParticleIndices, uint *cellStarts, uint *cellEnds)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= PARTICLECOUNT)
		return;

	float2 force = ProcessCollisions(threadNum, oldPositions, oldSpeeds, cellStarts, cellEnds);
	float2 newSpeed = oldSpeeds[threadNum] + force;
	newSpeed.y += GRAVITY;

	int originalIndex = gridParticleIndices[threadNum];

	newSpeeds[originalIndex] = newSpeed;
}

__global__ void CUDADrawPixels(uchar3 *pixels, float2* positions, float2* speeds, uchar4*logo, bool putCircle)
{
	int threadNum = threadIdx.x + blockDim.x*blockIdx.x;
	if (threadNum >= PARTICLECOUNT)
		return;

	float2 position = positions[threadNum];
	float2 speed = speeds[threadNum];

	if (position.x > IMAGEWFULL - BOUND)
	{
		speed.x += -1 * BOUNDARYFORCE*(position.x - IMAGEWFULL + BOUND);
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
	return (((uint)floor(p.y / CELLSIZE)) << LOGGRIDSIZE) | ((uint)floor(p.x / CELLSIZE));
}

__device__ float2 ProcessCollisions(int threadNum, float2* oldPositions, float2* oldSpeeds, uint *cellStarts, uint *cellEnds)
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
					force += DAMPING * relVel;
					// tangential shear force
					force += SHEARFORCE * tanVel;
					// attraction
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

	float2 p = positions[index];

	// get address in grid
	uint gridPosx = floor(p.x / CELLSIZE);
	uint gridPosy = floor(p.y / CELLSIZE);
	uint hash = (gridPosy << LOGGRIDSIZE) | gridPosx;

	// store grid hash and particle index
	hashes[index] = hash;
	indices[index] = index;
}

__device__ void PutCircle(int middleIndex, uchar3* pixels, int threadNum, uchar4*logo)
{
	int logoX = (threadNum % (DPPX * LOGOW)) / DPPX;
	int logoY = (threadNum / (LOGOW*DPPX)) / DPPX;
	int logoIndex = logoY * LOGOW + logoX;
	uchar4 tempColor = logo[logoIndex];
	uchar3 color;
	color.x = tempColor.x;
	color.y = tempColor.y;
	color.z = tempColor.z;

	if (RADIUS >= 3)
	{
		pixels[middleIndex - (IMAGEWFULL * 3) - 1] = color;
		pixels[middleIndex - (IMAGEWFULL * 3)] = color;
		pixels[middleIndex - (IMAGEWFULL * 3) + 1] = color;
		pixels[middleIndex - (IMAGEWFULL * 2) - 2] = color;
	}
	if (RADIUS >= 2)
	{
		pixels[middleIndex - (IMAGEWFULL * 2) - 1] = color;
		pixels[middleIndex - (IMAGEWFULL * 2)] = color;
		pixels[middleIndex - (IMAGEWFULL * 2) + 1] = color;
	}
	if (RADIUS >= 3)
	{
		pixels[middleIndex - (IMAGEWFULL * 2) + 2] = color;
		pixels[middleIndex - IMAGEWFULL - 3] = color;
	}
	if (RADIUS >= 2)
	{
		pixels[middleIndex - IMAGEWFULL - 2] = color;
		pixels[middleIndex - IMAGEWFULL - 1] = color;
	}
	pixels[middleIndex - IMAGEWFULL] = color;
	if (RADIUS >= 2)
	{
		pixels[middleIndex - IMAGEWFULL + 1] = color;
		pixels[middleIndex - IMAGEWFULL + 2] = color;
	}
	if (RADIUS >= 3)
	{
		pixels[middleIndex - IMAGEWFULL + 3] = color;
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
		pixels[middleIndex + IMAGEWFULL - 3] = color;
	}
	if (RADIUS >= 2)
	{
		pixels[middleIndex + IMAGEWFULL - 2] = color;
		pixels[middleIndex + IMAGEWFULL - 1] = color;
	}
	pixels[middleIndex + IMAGEWFULL] = color;
	if (RADIUS >= 2)
	{
		pixels[middleIndex + IMAGEWFULL + 1] = color;
		pixels[middleIndex + IMAGEWFULL + 2] = color;
	}
	if (RADIUS >= 3)
	{
		pixels[middleIndex + IMAGEWFULL + 3] = color;
		pixels[middleIndex + (IMAGEWFULL * 2) - 2] = color;
	}
	if (RADIUS >= 2)
	{
		pixels[middleIndex + (IMAGEWFULL * 2) - 1] = color;
		pixels[middleIndex + (IMAGEWFULL * 2)] = color;
		pixels[middleIndex + (IMAGEWFULL * 2) + 1] = color;
	}
	if (RADIUS >= 3)
	{
		pixels[middleIndex + (IMAGEWFULL * 2) + 2] = color;
		pixels[middleIndex + (IMAGEWFULL * 3) - 1] = color;
		pixels[middleIndex + (IMAGEWFULL * 3)] = color;
		pixels[middleIndex + (IMAGEWFULL * 3) + 1] = color;
	}

	//int r2 = RADIUS * RADIUS + 1;

	//int x = middleIndex % imageWidth;
	//int y = middleIndex / imageWidth;
	//for (int i = x - RADIUS; i <= x + RADIUS; i++)
	//{
	//	int dx2 = (i - x)*(i - x);
	//	for (int j = y - RADIUS; j <= y + RADIUS; j++)
	//	{
	//		int dy2 = (j - y)*(j - y);
	//		if (dx2 + dy2 <= r2)
	//		{
	//			int index = j * imageWidth + i;
	//			pixels[index] = color;
	//		}
	//	}
	//}
}

void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
}

void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
{
	thrust::sort_by_key(thrust::device, dGridParticleHash, dGridParticleHash + numParticles, dGridParticleIndex);
}

void CalcHashes(float2* positions, uint* hashes, uint* indices)
{
	uint numThreads, numBlocks;
	computeGridSize(PARTICLECOUNT, 1024, numBlocks, numThreads);

	// execute the kernel
	CalcHashesD << < numBlocks, numThreads >> > (positions, hashes, indices);

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

void InitializeParticles()
{
	size_t arraySize = PARTICLECOUNT * sizeof(float2);
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

	size_t indicesSize = sizeof(uint)* PARTICLECOUNT;

	checkCudaErrors(cudaMalloc((void **)&d_positions1, arraySize));
	checkCudaErrors(cudaMalloc((void **)&d_speeds1, arraySize));
	checkCudaErrors(cudaMalloc((void **)&d_positions2, arraySize));
	checkCudaErrors(cudaMalloc((void **)&d_speeds2, arraySize));
	checkCudaErrors(cudaMalloc((void **)&d_indices, indicesSize));
	checkCudaErrors(cudaMalloc((void **)&d_hashes, indicesSize));

	srand(0);

	int startX = (IMAGEWFULL - 2 * BOUND - STARTINGDIST * PARTICLECOUNTX) / 2;
	for (int i = 0; i < PARTICLECOUNTY; i++)
	{
		for (int j = 0; j < PARTICLECOUNTX; j++)
		{
			int index = i * PARTICLECOUNTX + j;
			positions[index].x = startX + STARTINGDIST * j;
			positions[index].y = BOUND + STARTINGDIST * i;
			speeds[index].x = 0;
			speeds[index].y = 0;
			speeds[index].x = (float)rand() / SPEEDFACTOR;
		}
	}
	checkCudaErrors(cudaMemcpy(d_positions1, positions, arraySize, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_speeds1, speeds, arraySize, cudaMemcpyHostToDevice));

	free(speeds);
	free(positions);
}

void RunCUDA(uchar3 *d_destinationBitmap, uchar4 *d_logo, bool putCircle)
{
	static bool initialized = false;
	cudaEvent_t start, stop;
	float milliseconds = 0;

	uint numThreads, numBlocks;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (!initialized)
	{
		initialized = true;
		InitializeParticles();
	}

	if (!d_cellStarts)
	{
		checkCudaErrors(cudaMalloc((void **)&d_cellEnds, sizeof(uint)*CELLCOUNT));
		checkCudaErrors(cudaMalloc((void **)&d_cellStarts, sizeof(uint)*CELLCOUNT));
	}

	computeGridSize(PARTICLECOUNT, 256, numBlocks, numThreads);

	cudaEventRecord(start);
	CalcHashes(d_positions1, d_hashes, d_indices);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("hashes took %f\n", milliseconds);


	cudaEventRecord(start);
	sortParticles(d_hashes, d_indices, PARTICLECOUNT);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("sorting took %f\n", milliseconds);


	cudaEventRecord(start);
	ReorderDataAndFindCellStart(d_cellStarts, d_cellEnds, d_positions2, d_speeds2, d_hashes, d_indices, d_positions1, d_speeds1, PARTICLECOUNT, CELLCOUNT);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("reordering took %f\n", milliseconds);

	cudaEventRecord(start);
	ProcessForces << <numBlocks, numThreads >> > (d_positions2, d_speeds2, d_speeds1, d_indices, d_cellStarts, d_cellEnds);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("forces took %f\n", milliseconds);
	getLastCudaError("CUDADrawPixels kernel execution failed.\n");


	if (putCircle)
	{
		cudaEventRecord(start);
		checkCudaErrors(cudaMemset(d_destinationBitmap, 0, sizeof(uchar3)* IMAGEWFULL* IMAGEHFULL));
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("memset took %f\n", milliseconds);
	}
	cudaEventRecord(start);
	CUDADrawPixels << <numBlocks, numThreads >> > (d_destinationBitmap, d_positions1, d_speeds1, d_logo, putCircle);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("drawing took %f\n", milliseconds);

	getLastCudaError("CUDADrawPixels kernel execution failed.\n");
}

__global__ void DownsampleImageD(uchar3 *d_imageStart, uchar3 *d_imageResult)
{
	int index = blockIdx.x* blockDim.x + threadIdx.x;
	if (index >= IMAGEW * IMAGEH)
		return;

	int myRow = index / IMAGEW;
	int myColumn = index % IMAGEW;
	int inputIndexStart = DOWNSAMPLING * (IMAGEWFULL * myRow + myColumn);

	ushort3 colorTMP = { 0,0,0 };
	uchar3 color;

	for (int j = 0; j < DOWNSAMPLING; j++)
	{
		for (int k = 0; k < DOWNSAMPLING; k++)
		{
			int indexP = inputIndexStart + j * IMAGEWFULL + k;
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

	computeGridSize(IMAGEW*IMAGEH, 1024, numBlocks, numThreads);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	DownsampleImageD << <numBlocks, numThreads >> > (d_imageStart, d_imageResult);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("downsampling took %f\n", milliseconds);
	getLastCudaError("CUDADrawPixels kernel execution failed.\n");
}
