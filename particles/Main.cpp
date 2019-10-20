#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <thread>
#include <mutex>
#include <chrono>

#include "parameters.h"
#include "particles.h"
#include "helpers.h"

class Semaphore
{
public:
	std::mutex mutex;
	std::condition_variable condition;
	unsigned long count = 0;

	void notify()
	{
		std::unique_lock<decltype(mutex)> lock(mutex);
		++count;
		condition.notify_one();
	}

	void wait()
	{
		std::unique_lock<decltype(mutex)> lock(mutex);
		while (!count)
			condition.wait(lock);
		--count;
	}
};

struct file_writing_data
{
	std::vector<vector<uchar3>> bitmaps;
	std::vector<bool> bitmaps_free;
	Semaphore render_semaphore;
	std::mutex access_mutex;

	file_writing_data()
	{
		bitmaps.resize(THREADCOUNT);
		for (auto& bitmap : bitmaps)
			bitmap.resize(IMAGEH * IMAGEW);

		bitmaps_free = std::vector<bool>(THREADCOUNT, true);
		render_semaphore.count = THREADCOUNT;
	}
};

void save_to_file(const int number, const std::string filename, file_writing_data* data)
{
	stbi_write_png(filename.c_str(), IMAGEW, IMAGEH, 3, static_cast<const void*>(data->bitmaps[number].data()), IMAGEW * 3);
	data->access_mutex.lock();
	data->bitmaps_free[number] = true;
	data->access_mutex.unlock();
	data->render_semaphore.notify();
}

void save(image_data& data, const std::string& directory, const u32 frame_number, file_writing_data& files, bool normalize_colors)
{
	downsample_image(data, normalize_colors);

	files.render_semaphore.wait();
	files.access_mutex.lock();
	int bitmap_nr = 0;
	for (; bitmap_nr < THREADCOUNT; bitmap_nr++)
		if (files.bitmaps_free[bitmap_nr])
			break;
	files.bitmaps_free[bitmap_nr] = false;
	files.access_mutex.unlock();

	files.bitmaps[bitmap_nr] = data.image_downsampled.copy_to_host();

	std::ostringstream ss;
	ss << directory << "/image-" << std::setw(5) << std::setfill('0') << frame_number << ".png";
	std::thread(save_to_file, bitmap_nr, ss.str(), &files).detach();
}

void get_frame(device_memory<ftype2>& positions, device_memory<ftype2>& speeds, const u32 frame_number, file_writing_data& files, const device_memory<uchar3>& logo)
{
	particle_data particles(std::move(positions), std::move(speeds));
	print_memory_usage();

	for (u32 j = 0; j < FRAMEDIFF; j++)
		process_step(particles);

	device_memory<ftype2> positions_stable(std::move(particles.positions_stable));
	device_memory<ftype2> speeds_stable(std::move(particles.speeds_stable));
	device_memory<ftype2> positions_sort(std::move(particles.positions_sort));
	device_memory<ftype2> speeds_sort(std::move(particles.speeds_sort));
	device_memory<u32> cell_starts(std::move(particles.cell_starts));

	particles = particle_data();
	print_memory_usage();

	image_data image_data;
	print_memory_usage();

	draw_by_logo(positions_stable, image_data, logo);
	save(image_data, "pngs", frame_number, files, false);

	draw_direction(positions_stable, speeds_stable, image_data);
	save(image_data, "pngsdir", frame_number, files, false);

	draw_force(positions_sort, speeds_sort, cell_starts, image_data);
	save(image_data, "pngsf", frame_number, files, true);

	positions = std::move(positions_stable);
	speeds = std::move(speeds_stable);
}

void run_sim(const vector<uchar3>& logo_host)
{
	srand(0);

	vector<ftype2> positions(PARTICLECOUNT);
	vector<ftype2> speeds(PARTICLECOUNT);

	for (int i = 0; i < PARTICLECOUNTY; i++)
	{
		for (int j = 0; j < PARTICLECOUNTX; j++)
		{
			const int index = i * PARTICLECOUNTX + j;
			positions[index].x = STARTINGX + STARTINGDIST * j;
			positions[index].y = STARTINGHEIGHT + STARTINGDIST * i;

			speeds[index].x = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX) - 0.5) * SPEEDFACTORX;
			speeds[index].y = (static_cast<double>(rand()) / static_cast<double>(RAND_MAX) - 0.5) * SPEEDFACTORY;
		}
	}
	device_memory<uchar3> logo(logo_host);
	print_memory_usage();
	file_writing_data files;
	device_memory<ftype2> positions_device(positions);
	device_memory<ftype2> speeds_device(speeds);
	positions.clear();
	speeds.clear();
	positions.shrink_to_fit();
	speeds.shrink_to_fit();

	for (u32 i = 1; i < FRAMECOUNT; i++)
	{
		//get_frame(positions_device, speeds_device, i, files, logo);
		std::cout << "Frame took: " << measure::execution_gpu(get_frame, positions_device, speeds_device, i, files, logo) << '\n';
	}
}

void load_texture(vector<uchar3>& texture, const char* filename)
{
	texture.resize(LOGOH * LOGOW);

	//int tex_width, tex_height, tex_channels;
	//const auto pixels = reinterpret_cast<uchar4*>(stbi_load(filename, &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha));

	//for (int i = 0; i < LOGOH / 2; i++)
	//{
	//	for (int j = 0; j < LOGOW; j++)
	//	{
	//		const int indexL = i * tex_width + j;
	//		const int indexR = (LOGOH - i - 1) * tex_width + j;

	//		const uchar4 tmp = pixels[indexL];
	//		pixels[indexL] = pixels[indexR];
	//		pixels[indexR] = tmp;
	//	}
	//}

	//texture.resize(LOGOH * LOGOW);
	//for (int i = 0; i < LOGOW * LOGOH; i++)
	//	texture[i] = { pixels[i].x,pixels[i].y,pixels[i].z };

	for (size_t i = 0; i < LOGOW * LOGOH; i++)
	{
		size_t ind_w = i * 20 / LOGOW / LOGOH;
		size_t ind_h = (i % LOGOW) * 20 / LOGOH;

		if (((ind_w + ind_h) % 2) == 0)
			texture[i] = { 255,0,0 };
		else
			texture[i] = { 0,0,255 };
	}
}

int main(const int argc, char** argv)
{
	stbi_flip_vertically_on_write(1);
	if (cudaSetDevice(0))
		return-1;

	vector<uchar3> logo_host;
	load_texture(logo_host, LOGOFILE);
	run_sim(logo_host);
}
