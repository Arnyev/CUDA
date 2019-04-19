#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "parameters.h"
#include <thread>
#include <mutex>
#include <iomanip>
#include "particles.h"
#include "helpers.h"

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
    std::vector<vector<uchar3>> bitmaps;
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

void save_to_file(const int number, const std::string& filename, file_writing_data* data)
{
    stbi_write_png(filename.c_str(), IMAGEW, IMAGEH, 3, static_cast<const void *>(data->bitmaps[number].data()), IMAGEW * 3);
    data->access_mutex.lock();
    data->bitmaps_free[number] = true;
    data->access_mutex.unlock();
    data->render_semaphore.notify();
}

void save(file_writing_data& files, const std::string& filename, const device_memory<uchar3> & image)
{
    files.render_semaphore.wait();
    files.access_mutex.lock();
    int bitmap_nr = 0;
    for (; bitmap_nr < THREADCOUNT; bitmap_nr++)
        if (files.bitmaps_free[bitmap_nr])
            break;
    files.bitmaps_free[bitmap_nr] = false;
    files.access_mutex.unlock();

    files.bitmaps[bitmap_nr] = image.copy_to_host();
    std::thread(save_to_file, bitmap_nr, filename, &files).detach();
}

void get_frame(particle_data& particles, const u32 frame_number, file_writing_data& files)
{
    for (u32 j = 0; j < FRAMEDIFF; j++)
        process_step(particles);

    draw_speed(particles);
    downsample_image(particles.image.begin(), particles.image_downsampled.begin());

    std::ostringstream ss;
    ss << "pngs/image-" << std::setw(5) << std::setfill('0') << frame_number << ".png";

    save(files, ss.str(), particles.image_downsampled);
    ss = std::ostringstream();
    ss << "pngsd/image-" << std::setw(5) << std::setfill('0') << frame_number << ".png";
    draw_density(particles);
    downsample_image(particles.image.begin(), particles.image_downsampled.begin());
    save(files, ss.str(), particles.image_downsampled);
}

void run_sim(const vector<uchar3>& logo_host)
{
    particle_data particles(logo_host);
    file_writing_data files;

    process_step(particles);

    for (u32 i = 1; i < FRAMECOUNT; i++)
        std::cout << "Frame took: " << measure::execution_gpu(get_frame, particles, i, files) << '\n';
}

void load_texture(vector<uchar3>& texture, const char* filename)
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
    vector<uchar3> logo_host;
    load_texture(logo_host, LOGOFILE);
    run_sim(logo_host);
}
