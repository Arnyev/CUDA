#pragma once

#include "device_memory.h"
#include "vector_types.h"

struct particle_data
{
    device_memory<float2> positions_stable;
    device_memory<float2> speeds_stable;
    device_memory<float2> positions_sort;
    device_memory<float2> speeds_sort;

    device_memory<u32> cell_starts;

    device_memory<u32> indices;
    device_memory<u32> particle_cell;

    device_memory<u32> indices_helper;
    device_memory<u32> particle_cell_helper;

    device_memory<uchar3> logo;
    device_memory<uchar3> image;
    device_memory<uchar3> image_downsampled;

    device_memory<u8> helper_storage;

    explicit particle_data(const vector<uchar3>& logo_host);
};

void put_circles(particle_data& data);
void downsample_image(const uchar3* input, uchar3* output);
void process_step(particle_data& particles);
