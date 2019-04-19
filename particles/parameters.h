#pragma once
#include <cstdint>

typedef uint32_t u32;
typedef uint8_t u8;

constexpr int get_max_bit(u32 value)
{
    int bit = 0;
    while (value > 0)
    {
        bit++;
        value >>= 1;
    }
    return bit;
}

constexpr const char* LOGOFILE = "logof.jpg";

constexpr float BOUND = 80.0f;
constexpr int RADIUS = 1;
constexpr int RADIUSD = 2;
constexpr u32 DRAWDISTSQRDIR = 4;
constexpr u32 DRAWDISTSQRS = 2;
constexpr u32 DRAWDISTSQRD = RADIUS * RADIUS;
constexpr u32 DRAWDISTSQR = RADIUS * RADIUS;
constexpr float COLLISIONDIST = 2 * RADIUS + 1.0f;
constexpr float SPRINGFORCE = -0.5f;
constexpr float GRAVITY = -0.00005f;
constexpr float DAMPING = 0.012f;
constexpr float BOUNDARYFORCE = 0.5f;
constexpr float SHEARFORCE = 0.06f;

constexpr u32 DOWNSAMPLING = 16;
constexpr u32 IMAGEW = 1920;
constexpr u32 IMAGEH = 1080;
constexpr u32 IMAGEWFULL = IMAGEW * DOWNSAMPLING;
constexpr u32 IMAGEHFULL = IMAGEH * DOWNSAMPLING;

constexpr u32 CELLSIZE = static_cast<u32>(COLLISIONDIST);
constexpr u32 CELLCOUNTX = IMAGEWFULL / CELLSIZE;
constexpr u32 CELLCOUNTY = IMAGEHFULL / CELLSIZE;
constexpr u32 CELLCOUNT = CELLCOUNTX * CELLCOUNTY;

constexpr u32 FRAMEDIFF = 30;
constexpr u32 FRAMECOUNT = 99999;

constexpr u32 LOGOW = 1016;
constexpr u32 LOGOH = 856;
constexpr u32 DPPX = 5;
constexpr u32 PARTICLECOUNTX = DPPX * LOGOW;
constexpr u32 PARTICLECOUNTY = DPPX * LOGOH;

constexpr float STARTINGDIST = COLLISIONDIST + 0.3f;
constexpr float STARTINGHEIGHT = BOUND + RADIUS;
constexpr float STARTINGX = (IMAGEWFULL - 2 * BOUND - STARTINGDIST * PARTICLECOUNTX) / 2;

constexpr u32 PARTICLECOUNT = PARTICLECOUNTX * PARTICLECOUNTY;

constexpr int sort_bits = get_max_bit(CELLCOUNT);

constexpr float SPEEDFACTORY = 0.2f;
constexpr float SPEEDFACTORX = 0.2f;
constexpr float color_speed_mult = 1000.0f;
constexpr float base_color_intensity = 96.0f;
constexpr float color_intensity_multiplier = color_speed_mult * 0.3f;

constexpr u32 THREADCOUNT = 16;
constexpr u32 BLOCKSIZE = 256;
constexpr u32 GRIDDIM = 2048;
