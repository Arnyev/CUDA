#pragma once
#include <cstdint>
#include "vector_types.h"

typedef uint32_t u32;
typedef uint8_t u8;
typedef double ftype;
typedef double2 ftype2;

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
constexpr float SPRINGFORCE = -0.005f;
constexpr float GRAVITY = -0.0004f;
constexpr float DAMPING = 0;
constexpr float BOUNDARYFORCE = 0.005f;
constexpr float SHEARFORCE = 0;

constexpr u32 DOWNSAMPLING = 8;
constexpr u32 IMAGEW = 1920;
constexpr u32 IMAGEH = 1080;
constexpr u32 IMAGEWFULL = IMAGEW * DOWNSAMPLING;
constexpr u32 IMAGEHFULL = IMAGEH * DOWNSAMPLING;

constexpr u32 CELLSIZE = static_cast<u32>(COLLISIONDIST);
constexpr u32 CELLCOUNTX = IMAGEWFULL / CELLSIZE;
constexpr u32 CELLCOUNTY = IMAGEHFULL / CELLSIZE;
constexpr u32 CELLCOUNT = CELLCOUNTX * CELLCOUNTY;

constexpr u32 FRAMEDIFF = 20;
constexpr u32 FRAMECOUNT = 99999;

constexpr u32 LOGOW = 1016;
constexpr u32 LOGOH = 856;
constexpr u32 DPPX = 2;
constexpr u32 PARTICLECOUNTX = DPPX * LOGOW;
constexpr u32 PARTICLECOUNTY = DPPX * LOGOH;

constexpr float STARTINGDIST = COLLISIONDIST + 0.3f;
constexpr float STARTINGHEIGHT = BOUND + RADIUS + 2600;
constexpr float STARTINGX = (IMAGEWFULL - 2 * BOUND - STARTINGDIST * PARTICLECOUNTX) / 2;

constexpr u32 PARTICLECOUNT = PARTICLECOUNTX * PARTICLECOUNTY;

constexpr int sort_bits = get_max_bit(CELLCOUNT);

constexpr float SPEEDFACTORY = 0.002f;
constexpr float SPEEDFACTORX = 0.002f;
constexpr float color_speed_mult = 10000.0f;
constexpr float base_color_intensity = 32.0f;
constexpr float color_intensity_multiplier = color_speed_mult * 0.3f;

constexpr u32 THREADCOUNT = 16;
constexpr u32 BLOCKSIZE = 256;
constexpr u32 GRIDDIM = 2048;
