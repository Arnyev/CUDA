#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <helper_math.h>
#include "parameters.h"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/sort.h>

using namespace thrust;

__host__ __device__ __inline__ uint get_cell_index(const float2 p)
{
	const auto grid_posx = static_cast<uint>(p.x) / CELLSIZE;
	const auto grid_posy = static_cast<uint>(p.y) / CELLSIZE;

	return grid_posx + grid_posy * CELLCOUNTX;
}

__global__ void put_circles_d(const float2* __restrict__ positions, uchar3* __restrict__ image, const uchar3* __restrict__ logo)
{
	const auto thread_id = THREAD_ID();
	if (thread_id >= PARTICLECOUNT)
		return;

	const auto position = positions[thread_id];
	const uint position_x = static_cast<uint>(position.x);
	const uint position_y = static_cast<uint>(position.y);

	if (position_y < RADIUS || position_y >= IMAGEHFULL - RADIUS || position_x < RADIUS || position_x >= IMAGEWFULL - RADIUS)
		return;

	const int logo_x = (thread_id % (DPPX * LOGOW)) / DPPX;
	const int logo_y = (thread_id / (LOGOW*DPPX)) / DPPX;
	const auto logo_index = logo_y * LOGOW + logo_x;
	const auto color = logo[logo_index];

#pragma unroll
	for (int i = -RADIUS; i <= RADIUS; i++)
#pragma unroll
		for (int j = -RADIUS; j <= RADIUS; j++)
			if (i * i + j * j <= DRAWDISTSQR)
				image[(position_y + i)* IMAGEWFULL + position_x + j] = color;
}

__global__ void downsample_image_d(const uchar3* __restrict__ input, uchar3* __restrict__ output)
{
	const uint thread_id = THREAD_ID();
	if (thread_id >= IMAGEW * IMAGEH)
		return;

	const uint my_row = thread_id / IMAGEW;
	const uint my_column = thread_id % IMAGEW;
	const uint input_index_start = DOWNSAMPLING * (IMAGEWFULL * my_row + my_column);

	uint3 color_tmp = { 0,0,0 };

#pragma unroll
	for (int j = 0; j < DOWNSAMPLING; j++)
	{
#pragma unroll
		for (int k = 0; k < DOWNSAMPLING; k++)
		{
			const uint index_p = input_index_start + j * IMAGEWFULL + k;
			color_tmp.x += input[index_p].x;
			color_tmp.y += input[index_p].y;
			color_tmp.z += input[index_p].z;
		}
	}

	uchar3 color;
	color.x = static_cast<unsigned char>(color_tmp.x / (DOWNSAMPLING*DOWNSAMPLING));
	color.y = static_cast<unsigned char>(color_tmp.y / (DOWNSAMPLING*DOWNSAMPLING));
	color.z = static_cast<unsigned char>(color_tmp.z / (DOWNSAMPLING*DOWNSAMPLING));

	output[thread_id] = color;
}

__global__ void process_particles_d(const float2* __restrict__ positions_in, const float2* __restrict__ speeds_in,
	const uint* __restrict__ cell_starts, float2* __restrict__ positions_out, float2* __restrict__ speeds_out,
	const uint* __restrict__ scatter_map)
{
	const uint thread_id = THREAD_ID();
	if (thread_id >= PARTICLECOUNT)
		return;

	const float2 position = positions_in[thread_id];
	const float2 speed = speeds_in[thread_id];
	const uint my_cell = get_cell_index(position);

	float2 force = { 0, GRAVITY };

	if (my_cell < CELLCOUNT - CELLCOUNTX - 2 && my_cell>CELLCOUNTX)
	{
#pragma unroll
		for (int y = -1; y <= 1; y++)
		{
			const uint prev_cell = my_cell + CELLCOUNTX * y - 1;
			//if (prev_cell + 3 >= CELLCOUNT)
			//	break;

			for (uint i = cell_starts[prev_cell]; i < cell_starts[prev_cell + 3]; i++)
			{
				if (i == thread_id)
					continue;
				//if (i >= PARTICLECOUNT)
				//	break;

				const float2 rel_pos = positions_in[i] - position;
				const float dist = length(rel_pos);

				if (dist >= COLLISIONDIST)
					continue;

				const float2 norm = rel_pos / dist;
				//const float2 rel_vel = speeds_in[i] - speed;

				force += SPRINGFORCE * (COLLISIONDIST - dist) * norm;
				//force += DAMPING * rel_vel;
				//force += SHEARFORCE * (rel_vel - dot(rel_vel, norm) * norm);
			}
		}
	}

	if (position.x > IMAGEWFULL - BOUND)
		force.x -= BOUNDARYFORCE*(position.x - IMAGEWFULL + BOUND);

	if (position.x < BOUND)
		force.x += BOUNDARYFORCE * (BOUND - position.x);

	if (position.y > IMAGEHFULL - BOUND)
		force.y -= BOUNDARYFORCE*(position.y - IMAGEHFULL + BOUND);

	if (position.y < BOUND)
		force.y += BOUNDARYFORCE * (BOUND - position.y);

	const uint out_index = scatter_map[thread_id];
	//if (out_index >= PARTICLECOUNT)
	//	return;

	const float2 newSpeed = speed + force;
	positions_out[out_index] = position + newSpeed;
	speeds_out[out_index] = newSpeed;
}

__global__ void update_cell_starts_d(const uint* __restrict__ cell_indices, uint* __restrict__ cell_starts)
{
	const uint thread_id = THREAD_ID();
	if (thread_id >= PARTICLECOUNT)
		return;

	const auto cell_index = cell_indices[thread_id];
	if (cell_index >= CELLCOUNT)
		return;

	if (thread_id == 0)
	{
		cell_starts[cell_index] = 0;
		return;
	}

	const auto last_index = cell_indices[thread_id - 1];
	if (cell_index != last_index)
		cell_starts[cell_index] = thread_id;
}

particle_data::particle_data(const host_vector<uchar3>& logo_host)
{
	logo = logo_host;
	host_vector<float2> positions_host(PARTICLECOUNT);
	host_vector<float2> speeds_host(PARTICLECOUNT);
	indices.resize(PARTICLECOUNT);
	particle_cell.resize(PARTICLECOUNT);
	positions_sort.resize(PARTICLECOUNT);
	speeds_sort.resize(PARTICLECOUNT);
	cell_starts.resize(CELLCOUNT);
	image.resize(IMAGEWFULL * IMAGEHFULL);
	image_downsampled.resize(IMAGEW * IMAGEH);

	srand(0);

	for (int i = 0; i < PARTICLECOUNTY; i++)
	{
		for (int j = 0; j < PARTICLECOUNTX; j++)
		{
			const int index = i * PARTICLECOUNTX + j;
			positions_host[index].x = STARTINGX + STARTINGDIST * j;
			positions_host[index].y = STARTINGHEIGHT + STARTINGDIST * i;

			speeds_host[index].x = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * SPEEDFACTORX;
			speeds_host[index].y = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * SPEEDFACTORY;
		}
	}

	positions_stable = positions_host;
	speeds_stable = speeds_host;
}

void put_circles(particle_data& data)
{
	constexpr uchar3 zero = { 0,0,0 };
	fill(data.image.begin(), data.image.end(), zero);
	STARTKERNEL(put_circles_d, PARTICLECOUNT, data.positions_stable.data().get(), data.image.data().get(), data.logo.data().get());
}

void downsample_image(const uchar3* input, uchar3* output)
{
	STARTKERNEL(downsample_image_d, IMAGEH*IMAGEW, input, output);
}

struct cell_functor : unary_function<float2, uint>
{
	__host__ __device__ __inline__ uint operator()(const float2 position) const
	{
		return get_cell_index(position);
	}
};

void process_particles(particle_data& particles)
{
	STARTKERNEL(process_particles_d, PARTICLECOUNT, particles.positions_sort.data().get(), particles.speeds_sort.data().
		get(), particles.cell_starts.data().get(), particles.positions_stable.data().get(), particles.speeds_stable.data
		().get(), particles.indices.data().get());
}

void update_cell_starts(particle_data& particles)
{
	fill(particles.cell_starts.begin(), particles.cell_starts.end(), PARTICLECOUNT);
	STARTKERNEL(update_cell_starts_d, PARTICLECOUNT, particles.particle_cell.data().get(), particles.cell_starts.data().get());
	inclusive_scan(particles.cell_starts.rbegin(), particles.cell_starts.rend(), particles.cell_starts.rbegin(), thrust::minimum<uint>());
}

void gather_particles(particle_data& particles)
{
	const auto iter_in = make_zip_iterator(make_tuple(particles.positions_stable.begin(), particles.speeds_stable.begin()));
	const auto iter_out = make_zip_iterator(make_tuple(particles.positions_sort.begin(), particles.speeds_sort.begin()));
	gather(particles.indices.begin(), particles.indices.end(), iter_in, iter_out);
}

void sort_particles(particle_data& particles)
{
	transform(particles.positions_stable.begin(), particles.positions_stable.end(), particles.particle_cell.begin(), cell_functor());
	sequence(particles.indices.begin(), particles.indices.end());
	sort_by_key(particles.particle_cell.begin(), particles.particle_cell.end(), particles.indices.begin());
}

void process_step(particle_data& particles)
{
	//std::cout << "Sorting took: " << measure::execution_gpu(sort_particles, particles) << std::endl;
	//std::cout << "Gather took: " << measure::execution_gpu(gather_particles, particles) << std::endl;
	//std::cout << "Update took: " << measure::execution_gpu(update_cell_starts, particles) << std::endl;
	//std::cout << "Processing took: " << measure::execution_gpu(process_particles, particles) << std::endl;
	sort_particles(particles);
	gather_particles(particles);
	update_cell_starts(particles);
	process_particles(particles);
}
