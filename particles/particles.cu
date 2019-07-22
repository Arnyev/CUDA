#include "device_launch_parameters.h"
#include "helper_math.h"
#include "parameters.h"
#include "device_reverse_iterator.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "particles.h"
#ifndef __INTELLISENSE__
#include "cub/device/device_scan.cuh"
#endif

__device__ __inline__ u32 get_cell_index(const ftype2 p)
{
	const auto grid_posx = static_cast<u32>(p.x) / CELLSIZE;
	const auto grid_posy = static_cast<u32>(p.y) / CELLSIZE;

	return grid_posx + grid_posy * CELLCOUNTX;
}

__global__ void draw_by_bitmap_d(const ftype2* __restrict__ positions, uchar3* __restrict__ image, const uchar3* __restrict__ logo)
{
	const auto thread_id = THREAD_ID();
	if (thread_id >= PARTICLECOUNT)
		return;

	const auto position = positions[thread_id];
	const u32 position_x = static_cast<u32>(position.x);
	const u32 position_y = static_cast<u32>(position.y);

	if (position_y < RADIUS || position_y >= IMAGEHFULL - RADIUS || position_x < RADIUS || position_x >= IMAGEWFULL - RADIUS)
		return;

	const int logo_x = (thread_id % (DPPX * LOGOW)) / DPPX;
	const int logo_y = (thread_id / (LOGOW * DPPX)) / DPPX;
	const auto logo_index = logo_y * LOGOW + logo_x;
	const auto color = logo[logo_index];

#pragma unroll
	for (int i = -RADIUS; i <= RADIUS; i++)
#pragma unroll
		for (int j = -RADIUS; j <= RADIUS; j++)
			if (i * i + j * j <= DRAWDISTSQR)
				image[(position_y + i) * IMAGEWFULL + position_x + j] = color;
}

__global__ void draw_speed_d(const ftype2* __restrict__ positions, uchar3* __restrict__ image, const ftype2* __restrict__ speeds)
{
	const auto thread_id = THREAD_ID();
	if (thread_id >= PARTICLECOUNT)
		return;

	const auto position = positions[thread_id];
	const u32 position_x = static_cast<u32>(position.x);
	const u32 position_y = static_cast<u32>(position.y);

	if (position_y < RADIUS || position_y >= IMAGEHFULL - RADIUS || position_x < RADIUS || position_x >= IMAGEWFULL - RADIUS)
		return;

	const auto speed = speeds[thread_id];
	uchar3 color = { 0,0,0 };

	const float speedC = length(speed) * color_speed_mult;

	if (speedC < 256)
	{
		color.x = 255;
		color.y = speedC;
	}
	else if (speedC < 512)
	{
		color.y = 255;
		color.x = 512 - speedC;
	}
	else if (speedC < 768)
	{
		color.y = 255;
		color.z = speedC - 512;
	}
	else if (speedC < 1024)
	{
		color.z = 255;
		color.y = 1024 - speedC;
	}
	else if (speedC < 1280)
	{
		color.z = 255;
		color.x = speedC - 1024;
	}
	else if (speedC < 1536)
	{
		color.z = 255;
		color.x = 255;
		color.y = speedC - 1280;
	}
	else
		color.z = color.y = color.x = 255;

#pragma unroll
	for (int i = -RADIUS; i <= RADIUS; i++)
#pragma unroll
		for (int j = -RADIUS; j <= RADIUS; j++)
			if (i * i + j * j <= DRAWDISTSQRS)
				image[(position_y + i) * IMAGEWFULL + position_x + j] = color;
}

__global__ void draw_direction_d(const ftype2* __restrict__ positions, uchar3* __restrict__ image, const ftype2* __restrict__ speeds)
{
	const auto thread_id = THREAD_ID();
	if (thread_id >= PARTICLECOUNT)
		return;

	const auto position = positions[thread_id];
	const u32 position_x = static_cast<u32>(position.x);
	const u32 position_y = static_cast<u32>(position.y);

	if (position_y < RADIUS || position_y >= IMAGEHFULL - RADIUS || position_x < RADIUS || position_x >= IMAGEWFULL - RADIUS)
		return;

	constexpr ftype2 v1 = { 0,1 };
	constexpr ftype2 v2 = { -0.86602540378,-0.5f };
	constexpr ftype2 v3 = { 0.86602540378,-0.5f };

	const auto speed = speeds[thread_id];
	const auto norm_speed = normalize(speed);
	const float cos1 = acosf(dot(norm_speed, v1));
	const float cos2 = acosf(dot(norm_speed, v2));
	const float cos3 = acosf(dot(norm_speed, v3));

	float mult = base_color_intensity + length(speed) * color_intensity_multiplier;
	if (mult > 256.0f)
		mult = 256.0f;

	const float3 colorf = mult * normalize(float3{ cos1, cos2, cos3 });
	const uchar3 color = { static_cast<u8>(colorf.x),static_cast<u8>(colorf.y),static_cast<u8>(colorf.z) };

#pragma unroll
	for (int i = -RADIUSD; i <= RADIUSD; i++)
#pragma unroll
		for (int j = -RADIUSD; j <= RADIUSD; j++)
			if (i * i + j * j <= DRAWDISTSQRDIR)
				image[(position_y + i) * IMAGEWFULL + position_x + j] = color;
}

__global__ void draw_density_d(const ftype2* __restrict__ positions, uchar3* __restrict__ image)
{
	const auto thread_id = THREAD_ID();
	if (thread_id >= PARTICLECOUNT)
		return;

	const auto position = positions[thread_id];
	const u32 position_x = static_cast<u32>(position.x);
	const u32 position_y = static_cast<u32>(position.y);

	if (position_y < RADIUS || position_y >= IMAGEHFULL - RADIUS || position_x < RADIUS || position_x >= IMAGEWFULL - RADIUS)
		return;

	constexpr uchar3 color = { 255,255,255 };

#pragma unroll
	for (int i = -RADIUS; i <= RADIUS; i++)
#pragma unroll
		for (int j = -RADIUS; j <= RADIUS; j++)
			if (i * i + j * j <= DRAWDISTSQRD)
				image[(position_y + i) * IMAGEWFULL + position_x + j] = color;
}

__device__ __inline__  double maax(double a, double b)
{
	return a > b ? a : b;
}

__global__ void downsample_image_d(const uchar3* __restrict__ input, uchar3* __restrict__ output, bool normalize_colors)
{
	const u32 thread_id = THREAD_ID();
	if (thread_id >= IMAGEW * IMAGEH)
		return;

	const u32 my_row = thread_id / IMAGEW;
	const u32 my_column = thread_id % IMAGEW;
	const u32 input_index_start = DOWNSAMPLING * (IMAGEWFULL * my_row + my_column);

	uint3 color_tmp = { 0,0,0 };

#pragma unroll
	for (int j = 0; j < DOWNSAMPLING; j++)
	{
#pragma unroll
		for (int k = 0; k < DOWNSAMPLING; k++)
		{
			const u32 index_p = input_index_start + j * IMAGEWFULL + k;
			color_tmp.x += input[index_p].x;
			color_tmp.y += input[index_p].y;
			color_tmp.z += input[index_p].z;
		}
	}

	double3 colord = { color_tmp.x ,color_tmp.y,color_tmp.z };

	colord.x /= (DOWNSAMPLING * DOWNSAMPLING);
	colord.y /= (DOWNSAMPLING * DOWNSAMPLING);
	colord.z /= (DOWNSAMPLING * DOWNSAMPLING);
	if (normalize_colors)
	{
		colord.x *= 3.0;
		colord.y *= 3.0;
		colord.z *= 3.0;

		double maxdim = maax(maax(colord.x, colord.y), colord.z);
		if (maxdim > 255)
		{
			colord.x *= 255.0 / maxdim;
			colord.y *= 255.0 / maxdim;
			colord.z *= 255.0 / maxdim;
		}
	}
	colord.x = colord.x > 255 ? 255 : colord.x;
	colord.y = colord.y > 255 ? 255 : colord.y;
	colord.z = colord.z > 255 ? 255 : colord.z;

	uchar3 color;
	color.x = static_cast<unsigned char>(colord.x);
	color.y = static_cast<unsigned char>(colord.y);
	color.z = static_cast<unsigned char>(colord.z);

	output[thread_id] = color;
}

__device__ __forceinline__ ftype2 process_bounds(const ftype2 position)
{
	ftype2 force = { 0,0 };
	if (position.x > IMAGEWFULL - BOUND)
		force.x -= BOUNDARYFORCE * (position.x - IMAGEWFULL + BOUND);

	if (position.x < BOUND)
		force.x += BOUNDARYFORCE * (BOUND - position.x);

	if (position.y > IMAGEHFULL - BOUND)
		force.y -= BOUNDARYFORCE * (position.y - IMAGEHFULL + BOUND);

	if (position.y < BOUND)
		force.y += BOUNDARYFORCE * (BOUND - position.y);

	return force;
}

__device__ __forceinline__ ftype2 process_collisions(const u32 thread_id, const ftype2 position, const ftype2 speed, const u32* cell_starts, const ftype2* positions_in, const ftype2* speeds_in)
{
	const u32 my_cell = get_cell_index(position);

	if (my_cell >= CELLCOUNT - CELLCOUNTX - 2 || my_cell <= CELLCOUNTX)
		return { 0,0 };

	ftype2 force = { 0,0 };

#pragma unroll
	for (int y = -1; y <= 1; y++)
	{
		const u32 prev_cell = my_cell + CELLCOUNTX * y - 1;

		for (u32 i = cell_starts[prev_cell]; i < cell_starts[prev_cell + 3]; i++)
		{
			if (i == thread_id)
				continue;

			const ftype2 rel_pos = positions_in[i] - position;
			const ftype dist = length(rel_pos);

			if (dist >= COLLISIONDIST)
				continue;

			const ftype2 norm = rel_pos / dist;
			const ftype2 rel_vel = speeds_in[i] - speed;

			force += SPRINGFORCE * (COLLISIONDIST - dist) * norm;
			force += DAMPING * rel_vel;
			force += SHEARFORCE * (rel_vel - dot(rel_vel, norm) * norm);
		}
	}

	return { force.x,force.y };
}

__global__ void process_particles_d(const ftype2* __restrict__ positions_in, const ftype2* __restrict__ speeds_in,
	const u32* __restrict__ cell_starts, ftype2* __restrict__ positions_out, ftype2* __restrict__ speeds_out,
	const u32* __restrict__ scatter_map)
{
	const auto thread_id = THREAD_ID();
	if (thread_id >= PARTICLECOUNT)
		return;

	const ftype2 position = positions_in[thread_id];
	const ftype2 speed = speeds_in[thread_id];

	ftype2 force = { 0, GRAVITY };

	force += process_collisions(thread_id, position, speed, cell_starts, positions_in, speeds_in);
	force += process_bounds(position);

	const u32 out_index = scatter_map[thread_id];
	const ftype2 newSpeed = speed + force;
	positions_out[out_index] = position + newSpeed;
	speeds_out[out_index] = newSpeed;
}

__global__ void draw_forces_d(const ftype2* __restrict__ positions_in, const ftype2* __restrict__ speeds_in, const u32* __restrict__ cell_starts, uchar3* __restrict__ image)
{
	const auto thread_id = THREAD_ID();
	if (thread_id >= PARTICLECOUNT)
		return;

	const ftype2 position = positions_in[thread_id];
	const u32 position_x = static_cast<u32>(position.x);
	const u32 position_y = static_cast<u32>(position.y);

	if (position_y < RADIUS || position_y >= IMAGEHFULL - RADIUS || position_x < RADIUS || position_x >= IMAGEWFULL - RADIUS)
		return;

	ftype2 force = { 0, GRAVITY };

	force += process_collisions(thread_id, position, speeds_in[thread_id], cell_starts, positions_in, speeds_in);
	force += process_bounds(position);

	constexpr ftype2 v1 = { 0,1 };
	constexpr ftype2 v2 = { -0.86602540378,-0.5f };
	constexpr ftype2 v3 = { 0.86602540378,-0.5f };

	const auto norm_force = normalize(force);
	const float cos1 = acosf(dot(norm_force, v1));
	const float cos2 = acosf(dot(norm_force, v2));
	const float cos3 = acosf(dot(norm_force, v3));

	float mult = sqrt(length(force)) * color_intensity_multiplier;
	if (mult > 256.0f)
		mult = 256.0f;

	const float3 colorf = mult * normalize(float3{ cos1, cos2, cos3 });
	const uchar3 color = { static_cast<u8>(colorf.x),static_cast<u8>(colorf.y),static_cast<u8>(colorf.z) };

#pragma unroll
	for (int i = -RADIUSD; i <= RADIUSD; i++)
#pragma unroll
		for (int j = -RADIUSD; j <= RADIUSD; j++)
			if (i * i + j * j <= DRAWDISTSQRDIR)
				image[(position_y + i) * IMAGEWFULL + position_x + j] = color;
}

__global__ void compute_cell_and_sequence_d(const ftype2* __restrict__ positions, u32* __restrict__ cells, u32* __restrict__ indices)
{
	const auto thread_id = THREAD_ID();
	if (thread_id >= PARTICLECOUNT)
		return;

	cells[thread_id] = get_cell_index(positions[thread_id]);
	indices[thread_id] = thread_id;
}

void draw_by_bitmap(particle_data& data)
{
	data.image.fill_with_zeroes();
	STARTKERNEL(draw_by_bitmap_d, PARTICLECOUNT, data.positions_stable.begin(), data.image.begin(), data.logo.begin());
}

void draw_speed(particle_data& data)
{
	data.image.fill_with_zeroes();
	STARTKERNEL(draw_speed_d, PARTICLECOUNT, data.positions_stable.begin(), data.image.begin(), data.speeds_stable.begin());
}

void draw_force(particle_data& data)
{
	data.image.fill_with_zeroes();
	STARTKERNEL(draw_forces_d, PARTICLECOUNT, data.positions_sort.begin(), data.speeds_sort.begin(), data.cell_starts.begin(), data.image.begin());
}

void draw_direction(particle_data& data)
{
	data.image.fill_with_zeroes();
	STARTKERNEL(draw_direction_d, PARTICLECOUNT, data.positions_stable.begin(), data.image.begin(), data.speeds_stable.begin());
}

void draw_density(particle_data& data)
{
	data.image.fill_with_zeroes();
	STARTKERNEL(draw_density_d, PARTICLECOUNT, data.positions_stable.begin(), data.image.begin());
}

void downsample_image(const uchar3* input, uchar3* output, bool normalize_colors)
{
	STARTKERNEL(downsample_image_d, IMAGEH * IMAGEW, input, output, normalize_colors);
}

void sort_particles(particle_data& particles)
{
	cub::DoubleBuffer<u32> keys_buffer(particles.particle_cell.begin(), particles.particle_cell_helper.begin());
	cub::DoubleBuffer<u32> items_buffer(particles.indices.begin(), particles.indices_helper.begin());

	const auto count = static_cast<int>(particles.indices.size());
	auto storage_size = particles.helper_storage.size();
	const auto status = cub::DeviceRadixSort::SortPairs(particles.helper_storage.begin(), storage_size, keys_buffer, items_buffer, count, 0, sort_bits, nullptr, false);
	check_status(status, "radix_sort: failed on sorting.");

	if (keys_buffer.selector != 0)
		particles.particle_cell.swap_with(particles.particle_cell_helper);

	if (items_buffer.selector != 0)
		particles.indices.swap_with(particles.indices_helper);
}

__global__ void gather_values_compute_cell_starts_d(const u32* __restrict__ index_map,
	const ftype2* __restrict__ positions_in, const ftype2* __restrict__ speeds_in,
	ftype2* __restrict__ positions_out, ftype2* __restrict__ speeds_out,
	const u32* __restrict__ cell_indices, u32* __restrict__ cell_starts)
{
	const auto id = THREAD_ID();
	if (id >= PARTICLECOUNT)
		return;

	const auto my_index = index_map[id];
	positions_out[id] = positions_in[my_index];
	speeds_out[id] = speeds_in[my_index];

	const auto cell_index = cell_indices[id];
	if (cell_index >= CELLCOUNT)
		return;

	if (id == 0)
	{
		cell_starts[cell_index] = 0;
		return;
	}

	const auto last_index = cell_indices[id - 1];
	if (cell_index != last_index)
		cell_starts[cell_index] = id;
}

__global__ void fill_d(u32* __restrict__ output, const u32 value, const size_t count)
{
	const auto id = THREAD_ID();
	if (id >= count)
		return;

	output[id] = value;
}

template<typename T>
struct minimum
{
	__host__ __device__ __inline__ T operator()(const T& a, const T& b) const
	{
		return a < b ? a : b;
	}
};

void scan(particle_data& particles)
{
	const auto rbegin = device_reverse_iterator<u32>(particles.cell_starts.end() - 1);

	auto storage_size = particles.helper_storage.size();
	const auto status = cub::DeviceScan::InclusiveScan(particles.helper_storage.begin(), storage_size, rbegin, rbegin, minimum<u32>(), particles.cell_starts.size());
	check_status(status, "Cub device scan actual function.");
}

void gather_values_compute_cell_starts(particle_data& particles)
{
	STARTKERNEL(gather_values_compute_cell_starts_d, particles.indices.size(), particles.indices.begin(), particles.positions_stable.begin()
		, particles.speeds_stable.begin(), particles.positions_sort.begin(), particles.speeds_sort.begin(), particles.
		particle_cell.begin(), particles.cell_starts.begin());
}

void fill_cell_starts(particle_data& particles)
{
	STARTKERNEL(fill_d, particles.cell_starts.size(), particles.cell_starts.begin(), static_cast<u32>(PARTICLECOUNT), particles.cell_starts.size());
}

void process_particles(particle_data& particles)
{
	STARTKERNEL(process_particles_d, PARTICLECOUNT, particles.positions_sort.begin(), particles.speeds_sort.begin(), particles.cell_starts.begin(), particles.positions_stable.begin(), particles.speeds_stable.begin(), particles.indices.begin());
}

void compute_cells_and_sequence(particle_data& particles)
{
	STARTKERNEL(compute_cell_and_sequence_d, particles.positions_stable.size(), particles.positions_stable.begin(), particles.particle_cell.begin(), particles.indices.begin());
}
void process_step(particle_data& particles)
{
	//std::cout << "Compute cells: " << measure::execution_gpu(compute_cells_and_sequence, particles) << '\n';
	//std::cout << "Sort: " << measure::execution_gpu(sort_particles, particles) << '\n';
	//std::cout << "Fill: " << measure::execution_gpu(fill_cell_starts, particles) << '\n';
	//std::cout << "Gather, compute cells: " << measure::execution_gpu(gather_values_compute_cell_starts, particles) << '\n';
	//std::cout << "Scan: " << measure::execution_gpu(scan, particles) << '\n';
	//std::cout << "Process particles: " << measure::execution_gpu(process_particles, particles) << '\n';

	compute_cells_and_sequence(particles);
	sort_particles(particles);
	fill_cell_starts(particles);
	gather_values_compute_cell_starts(particles);
	scan(particles);
	process_particles(particles);
}

size_t get_needed_storage()
{
	cub::DoubleBuffer<u32> keys_buffer(nullptr, nullptr);
	cub::DoubleBuffer<u32> items_buffer(nullptr, nullptr);
	size_t needed_storage_sort = 0;
	cudaError_t status = cub::DeviceRadixSort::SortPairs(nullptr, needed_storage_sort, keys_buffer, items_buffer, static_cast<int>(PARTICLECOUNT), 0, sort_bits, nullptr, false);
	check_status(status, "radix_sort: failed on getting memory size.");

	size_t needed_storage_scan = 0;
	status = cub::DeviceScan::InclusiveScan(nullptr, needed_storage_scan, device_reverse_iterator<u32>(nullptr), device_reverse_iterator<u32>(nullptr), minimum<u32>(), static_cast<int>(CELLCOUNT));
	check_status(status, "Cub device scan get memory usage.");

	return needed_storage_sort > needed_storage_scan ? needed_storage_sort : needed_storage_scan;;
}

particle_data::particle_data(const vector<uchar3>& logo_host, const vector<ftype2>& positions, const vector<ftype2>& speeds)
{
	logo = logo_host;
	positions_stable = positions;
	speeds_stable = speeds;

	indices.resize(PARTICLECOUNT);
	indices_helper.resize(PARTICLECOUNT);

	particle_cell.resize(PARTICLECOUNT);
	particle_cell_helper.resize(PARTICLECOUNT);

	positions_sort.resize(PARTICLECOUNT);
	speeds_sort.resize(PARTICLECOUNT);
	cell_starts.resize(CELLCOUNT);
	image.resize(IMAGEWFULL * IMAGEHFULL);
	image_downsampled.resize(IMAGEW * IMAGEH);

	helper_storage.resize(get_needed_storage());
}
