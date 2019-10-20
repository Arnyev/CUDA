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

	device_memory<u8> helper_storage;

	particle_data(device_memory<ftype2>&& positions, device_memory<ftype2>&& speeds);
	particle_data() = default;
};

struct image_data
{
	device_memory<uchar3> image;
	device_memory<uchar3> image_downsampled;

	image_data();
};

void downsample_image(image_data& image_data, bool normalize_colors);
void process_step(particle_data& particles);
void draw_by_logo(const device_memory<ftype2>& positions_stable, image_data& image_data, const device_memory<uchar3>& logo);
void draw_force(const device_memory<ftype2>& positions_sort, const device_memory<ftype2>& speeds_sort, const device_memory<u32>& cell_starts, image_data& image_data);
void draw_direction(const device_memory<ftype2>& positions_stable, const device_memory<ftype2>& speeds_stable, image_data& image_data);
void draw_density(const device_memory<ftype2>& positions_stable, image_data& image_data);
void downsample_image(image_data& image_data, bool normalize_colors);
void compute_acceleration_colors(const particle_data& data, device_memory<uchar3>& colors);
