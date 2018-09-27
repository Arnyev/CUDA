#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "parameters.h"
#include <thread>
#include <mutex>
#include <iomanip>
#include <sstream>

template <typename F, typename ... Args>
std::chrono::microseconds::rep measure::execution(F&& func, Args&&... args)
{
	const auto start = std::chrono::high_resolution_clock::now();
	std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>
		(std::chrono::high_resolution_clock::now() - start);
	return duration.count();
}

template <typename F, typename ... Args>
float measure::execution_gpu(F&& func, Args&&... args)
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

class Semaphore
{
public:
	std::mutex mutex;
	std::condition_variable condition;
	unsigned long count = 0; // Initialized as locked.

	void notify()
	{
		std::unique_lock<decltype(mutex)> lock(mutex);
		++count;
		condition.notify_one();
	}

	void wait()
	{
		std::unique_lock<decltype(mutex)> lock(mutex);
		while (!count) // Handle spurious wake-ups.
			condition.wait(lock);
		--count;
	}
};

struct file_writing_data
{
	std::vector<thrust::host_vector<uchar3>> bitmaps;
	std::vector<bool> bitmaps_free;
	Semaphore render_semaphore;
	std::mutex access_mutex;

	file_writing_data()
	{
		bitmaps.resize(THREADCOUNT);
		for (auto& bitmap : bitmaps)
			bitmap.resize(IMAGEH*IMAGEW);

		bitmaps_free = std::vector<bool>(THREADCOUNT, true);
		render_semaphore.count = THREADCOUNT;
	}
};

void save_to_file(const int number, const int png_number, file_writing_data* data)
{
	std::ostringstream ss;
	ss << "pngs/image-" << std::setw(5) << std::setfill('0') << png_number << ".png";

	stbi_write_png(ss.str().data(), IMAGEW, IMAGEH, 3, static_cast<const void *>(data->bitmaps[number].data()), IMAGEW * 3);
	data->access_mutex.lock();
	data->bitmaps_free[number] = true;
	data->access_mutex.unlock();
	data->render_semaphore.notify();
}

void save(file_writing_data& files, int png_number, const thrust::device_vector<uchar3> & image)
{
	files.render_semaphore.wait();
	files.access_mutex.lock();
	int bitmap_nr = 0;
	for (; bitmap_nr < THREADCOUNT; bitmap_nr++)
		if (files.bitmaps_free[bitmap_nr])
			break;
	files.bitmaps_free[bitmap_nr] = false;
	files.access_mutex.unlock();

	files.bitmaps[bitmap_nr] = image;
	std::thread(save_to_file, bitmap_nr, png_number, &files).detach();
}

void run_sim(const thrust::host_vector<uchar3>& logo_host)
{
	particle_data particles(logo_host);
	file_writing_data files;

	for (int i = 0; i < FRAMEDIFF*FRAMECOUNT; i++)
	{
		std::cout << "Step took " << measure::execution_gpu(process_step, particles) << std::endl;
		//process_step(particles);
		if (i % FRAMEDIFF != 1)
			continue;

		//std::cout << "Circles took " << measure::execution_gpu(put_circles, particles) << std::endl;
		//std::cout << "Downsample took " << measure::execution_gpu(downsample_image, particles.image.data().get(), particles.image_downsampled.data().get()) << std::endl;
		//const int png_number = i / FRAMEDIFF + 1;
		//std::cout << "Saving took " << measure::execution(save, files, png_number, particles.image_downsampled) << std::endl;

		put_circles(particles);
		downsample_image(particles.image.data().get(), particles.image_downsampled.data().get());
		const int png_number = i / FRAMEDIFF + 1;
		save(files, png_number, particles.image_downsampled);
	}
}

void load_texture(thrust::host_vector<uchar3>& texture, const char* filename)
{
	int tex_width, tex_height, tex_channels;
	const auto pixels = reinterpret_cast<uchar4*>(stbi_load(filename, &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha));

	for (int i = 0; i < LOGOH / 2; i++)
	{
		for (int j = 0; j < LOGOW; j++)
		{
			const int indexL = i * tex_width + j;
			const int indexR = (LOGOH - i - 1)*tex_width + j;

			const uchar4 tmp = pixels[indexL];
			pixels[indexL] = pixels[indexR];
			pixels[indexR] = tmp;
		}
	}

	texture.resize(LOGOH*LOGOW);
	for (int i = 0; i < LOGOW*LOGOH; i++)
		texture[i] = { pixels[i].x,pixels[i].y,pixels[i].z };

	free(pixels);
}

int main(const int argc, char **argv)
{
	stbi_flip_vertically_on_write(1);
	stbi_write_png_compression_level = 0;
	if (cudaSetDevice(0))
		return-1;
	thrust::host_vector<uchar3> logo_host;
	load_texture(logo_host, LOGOFILE);
	run_sim(logo_host);
}
