#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <sys\timeb.h> 
#include <parameters.h> 
#include <mutex>
#include <condition_variable>
#include <thread>
#include <algorithm>
#include <iterator>
#include <vector>

uchar3 *d_imageBitmap = NULL;
uchar3 *d_imageBitmapDownsampled = NULL;

uchar4 *d_logo = NULL;
uchar4* pixels = NULL;

void RunCUDA(uchar3 *d_destinationBitmap, uchar4 *d_logo, bool putCircle);
void DownsampleImage(uchar3 *d_imageStart, uchar3 *d_imageResult);


class Semaphore
{
public:
	std::mutex mutex;
	std::condition_variable condition;
	unsigned long count = 0; // Initialized as locked.

public:
	void notify() {
		std::unique_lock<decltype(mutex)> lock(mutex);
		++count;
		condition.notify_one();
	}

	void wait() {
		std::unique_lock<decltype(mutex)> lock(mutex);
		while (!count) // Handle spurious wake-ups.
			condition.wait(lock);
		--count;
	}

	bool try_wait() {
		std::unique_lock<decltype(mutex)> lock(mutex);
		if (count) {
			--count;
			return true;
		}
		return false;
	}
};

unsigned char *h_imageBitmaps[THREADCOUNT];

bool bitmapsFree[THREADCOUNT];
Semaphore renderSemaphore;
std::mutex accessMutex;

void SaveToFile(int number, int pngNumber)
{
	char fileName[100];
	if (pngNumber < 10)
		sprintf(fileName, "pngs/image-000%d.png", pngNumber);
	else if (pngNumber < 100)
		sprintf(fileName, "pngs/image-00%d.png", pngNumber);
	else if (pngNumber < 1000)
		sprintf(fileName, "pngs/image-0%d.png", pngNumber);
	else
		sprintf(fileName, "pngs/image-%d.png", pngNumber);

	stbi_write_png(fileName, IMAGEW, IMAGEH, 3, (const void *)h_imageBitmaps[number], IMAGEW * 3);
	accessMutex.lock();
	bitmapsFree[number] = true;
	accessMutex.unlock();
	renderSemaphore.notify();
}

void renderImage()
{
	static int pngNumber = 0;
	static unsigned int timesRendered = 0;

	timesRendered++;
	bool putCircle = timesRendered % FRAMEDIFF == 0;

	RunCUDA(d_imageBitmap, d_logo, putCircle);

	if (timesRendered % FRAMEDIFF != 0)
		return;

	size_t resultImageSize = IMAGEH * IMAGEW * 3;

	DownsampleImage(d_imageBitmap, d_imageBitmapDownsampled);

	pngNumber++;
	renderSemaphore.wait();
	accessMutex.lock();
	int i = 0;
	for (; i < THREADCOUNT; i++)
		if (bitmapsFree[i])
			break;
	bitmapsFree[i] = false;
	accessMutex.unlock();

	checkCudaErrors(cudaMemcpy(h_imageBitmaps[i], d_imageBitmapDownsampled, resultImageSize, cudaMemcpyDeviceToHost));

	std::thread(SaveToFile, i,pngNumber).detach();
}



void createTextureImage()
{
	size_t logoSize = LOGOH * LOGOW * sizeof(uchar4);
	uchar4 tmp;

	int texWidth, texHeight, texChannels;
	pixels = (uchar4*)stbi_load("logo.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

	for (int i = 0; i < texHeight / 2; i++)
	{
		for (int j = 0; j < texWidth; j++)
		{
			int indexL = i * texWidth + j;
			int indexR = (LOGOH - i - 1)*texWidth + j;

			tmp = pixels[indexL];
			pixels[indexL] = pixels[indexR];
			pixels[indexR] = tmp;
		}
	}

	checkCudaErrors(cudaMalloc((void **)&d_logo, logoSize));
	checkCudaErrors(cudaMemcpy(d_logo, pixels, logoSize, cudaMemcpyHostToDevice));
}

int main(int argc, char **argv)
{
	stbi_flip_vertically_on_write(1);
	stbi_write_png_compression_level = 0;
	findCudaDevice(argc, (const char **)argv);
	createTextureImage();

	for (int i = 0; i < THREADCOUNT; i++)
	{
		h_imageBitmaps[i] = (unsigned char *)malloc(IMAGEW * IMAGEH * 3);
		bitmapsFree[i] = true;
	}
	renderSemaphore.count = THREADCOUNT;

	size_t resultImageSize = IMAGEW * IMAGEH * 3;
	size_t imageFullSize = IMAGEWFULL * IMAGEHFULL * 3;

	checkCudaErrors(cudaMalloc((void **)&d_imageBitmap, imageFullSize));
	checkCudaErrors(cudaMalloc((void **)&d_imageBitmapDownsampled, resultImageSize));

	for (int i = 0; i < FRAMEDIFF*FRAMECOUNT; i++)
		renderImage();
}

