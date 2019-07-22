#pragma once

#include "device_memory.h"
#include "vector_types.h"
#include "parameters.h"

struct particle_data
{
    device_memory<ftype2> positions_stable;
    device_memory<ftype2> speeds_stable;
    device_memory<ftype2> positions_sort;
    device_memory<ftype2> speeds_sort;

    device_memory<u32> cell_starts;

    device_memory<u32> indices;
    device_memory<u32> particle_cell;

    device_memory<u32> indices_helper;
    device_memory<u32> particle_cell_helper;

    device_memory<uchar3> logo;
    device_memory<uchar3> image;
    device_memory<uchar3> image_downsampled;

    device_memory<u8> helper_storage;

	particle_data::particle_data(const vector<uchar3>& logo_host, const vector<ftype2>& positions, const vector<ftype2>& speeds);
};

void draw_by_bitmap(particle_data& data);
void downsample_image(const uchar3* input, uchar3* output, bool normalize_colors);
void process_step(particle_data& particles);
void draw_speed(particle_data& data);
void draw_density(particle_data& data);
void draw_force(particle_data& data);
void draw_direction(particle_data& data);
