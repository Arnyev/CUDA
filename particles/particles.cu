#include "device_launch_parameters.h"
#include "helper_math.h"
#include "parameters.h"
#include "device_reverse_iterator.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "particles.h"
#ifndef __INTELLISENSE__
#include "cub/device/device_scan.cuh"
#endif

__host__ __device__ __inline__ u32 get_cell_index(const float2 p)
{
    const auto grid_posx = static_cast<u32>(p.x) / CELLSIZE;
    const auto grid_posy = static_cast<u32>(p.y) / CELLSIZE;

    return grid_posx + grid_posy * CELLCOUNTX;
}

__global__ void put_circles_d(const float2* __restrict__ positions, uchar3* __restrict__ image, const uchar3* __restrict__ logo)
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

    uchar3 color;
    color.x = static_cast<unsigned char>(color_tmp.x / (DOWNSAMPLING*DOWNSAMPLING));
    color.y = static_cast<unsigned char>(color_tmp.y / (DOWNSAMPLING*DOWNSAMPLING));
    color.z = static_cast<unsigned char>(color_tmp.z / (DOWNSAMPLING*DOWNSAMPLING));

    output[thread_id] = color;
}

__global__ void process_particles_d(const float2* __restrict__ positions_in, const float2* __restrict__ speeds_in,
    const u32* __restrict__ cell_starts, float2* __restrict__ positions_out, float2* __restrict__ speeds_out,
    const u32* __restrict__ scatter_map)
{
    const u32 thread_id = THREAD_ID();
    if (thread_id >= PARTICLECOUNT)
        return;

    const float2 position = positions_in[thread_id];
    const float2 speed = speeds_in[thread_id];
    const u32 my_cell = get_cell_index(position);

    float2 force = { 0, GRAVITY };

    if (my_cell < CELLCOUNT - CELLCOUNTX - 2 && my_cell>CELLCOUNTX)
    {
#pragma unroll
        for (int y = -1; y <= 1; y++)
        {
            const u32 prev_cell = my_cell + CELLCOUNTX * y - 1;

            for (u32 i = cell_starts[prev_cell]; i < cell_starts[prev_cell + 3]; i++)
            {
                if (i == thread_id)
                    continue;

                const float2 rel_pos = positions_in[i] - position;
                const float dist = length(rel_pos);

                if (dist >= COLLISIONDIST)
                    continue;

                const float2 norm = rel_pos / dist;
                const float2 rel_vel = speeds_in[i] - speed;

                force += SPRINGFORCE * (COLLISIONDIST - dist) * norm;
                force += DAMPING * rel_vel;
                force += SHEARFORCE * (rel_vel - dot(rel_vel, norm) * norm);
            }
        }
    }

    if (position.x > IMAGEWFULL - BOUND)
        force.x -= BOUNDARYFORCE * (position.x - IMAGEWFULL + BOUND);

    if (position.x < BOUND)
        force.x += BOUNDARYFORCE * (BOUND - position.x);

    if (position.y > IMAGEHFULL - BOUND)
        force.y -= BOUNDARYFORCE * (position.y - IMAGEHFULL + BOUND);

    if (position.y < BOUND)
        force.y += BOUNDARYFORCE * (BOUND - position.y);

    const u32 out_index = scatter_map[thread_id];
    const float2 newSpeed = speed + force;
    positions_out[out_index] = position + newSpeed;
    speeds_out[out_index] = newSpeed;
}

__global__ void compute_cell_and_sequence_d(const float2* __restrict__ positions, u32* __restrict__ cells, u32* __restrict__ indices, const size_t count)
{
    const u32 id = THREAD_ID();
    if (id >= count)
        return;

    cells[id] = get_cell_index(positions[id]);
    indices[id] = id;
}

template<typename T>
struct minimum
{
    __host__ __device__ __inline__ T operator()(const T& a, const T& b) const
    {
        return a < b ? a : b;
    }
};

particle_data::particle_data(const vector<uchar3>& logo_host)
{
    logo = logo_host;
    vector<float2> positions_host(PARTICLECOUNT);
    vector<float2> speeds_host(PARTICLECOUNT);
    indices.resize(PARTICLECOUNT);
    indices_helper.resize(PARTICLECOUNT);

    particle_cell.resize(PARTICLECOUNT);
    particle_cell_helper.resize(PARTICLECOUNT);

    positions_sort.resize(PARTICLECOUNT);
    speeds_sort.resize(PARTICLECOUNT);
    cell_starts.resize(CELLCOUNT);
    image.resize(IMAGEWFULL * IMAGEHFULL);
    image_downsampled.resize(IMAGEW * IMAGEH);

    cub::DoubleBuffer<u32> keys_buffer(nullptr, nullptr);
    cub::DoubleBuffer<u32> items_buffer(nullptr, nullptr);

    size_t needed_storage_sort = 0;
    cudaError_t status = cub::DeviceRadixSort::SortPairs(nullptr, needed_storage_sort, keys_buffer, items_buffer, static_cast<int>(PARTICLECOUNT), 0, sort_bits, nullptr, false);
    check_status(status, "radix_sort: failed on getting memory size.");

    size_t needed_storage_scan = 0;
    const auto rbegin = device_reverse_iterator<u32>(cell_starts.end() - 1);
    status = cub::DeviceScan::InclusiveScan(nullptr, needed_storage_scan, rbegin, rbegin, minimum<u32>(), static_cast<int>(CELLCOUNT));
    check_status(status, "Cub device scan get memory usage.");

    const size_t needed_storage = needed_storage_sort > needed_storage_scan ? needed_storage_sort : needed_storage_scan;
    helper_storage.resize(needed_storage);

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
    data.image.fill_with_zeroes();
    STARTKERNEL(put_circles_d, PARTICLECOUNT, data.positions_stable.begin(), data.image.begin(), data.logo.begin());
}

void downsample_image(const uchar3* input, uchar3* output)
{
    STARTKERNEL(downsample_image_d, IMAGEH*IMAGEW, input, output);
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
    const float2* __restrict__ positions_in, const float2* __restrict__ speeds_in,
    float2* __restrict__ positions_out, float2* __restrict__ speeds_out,
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
    STARTKERNEL(compute_cell_and_sequence_d, particles.positions_stable.size(), particles.positions_stable.begin(), particles.particle_cell.begin(), particles.indices.begin(), particles.particle_cell.size());
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
