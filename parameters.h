#pragma once

#define BOUND 50.0f
#define RADIUS 1
#define DRAWDISTSQR (RADIUS*RADIUS)
#define COLLISIONDIST (2*RADIUS+1.0f)

#define SPRINGFORCE -0.25f
#define GRAVITY -0.0002f
#define DAMPING 0.00002f
#define BOUNDARYFORCE 0.2f
#define SHEARFORCE 0.01f

#define LOGCELLSIZE 2UL
#define CELLSIZE (1UL<<LOGCELLSIZE)
#define LOGGRIDSIZE 13
#define CELLCOUNTX (1<<LOGGRIDSIZE)
#define CELLCOUNTY (1<<LOGGRIDSIZE)
#define CELLCOUNT (CELLCOUNTX * CELLCOUNTY )

#define DOWNSAMPLING 16
#define DOWNSAMPLINGSQR (DOWNSAMPLING*DOWNSAMPLING)
#define LOGDOWNSAMPLINGSQR 8
#define IMAGEW 1920
#define IMAGEH 1080
#define IMAGEWFULL (IMAGEW*DOWNSAMPLING)
#define IMAGEHFULL (IMAGEH*DOWNSAMPLING)

#define FRAMEDIFF 20
#define BIGFRAMEDIFF (FRAMEDIFF*500000)
#define PHASEFRAMES 200.0f
#define WAVEAMPLITUDE 4.0f
#define STARTINGPHASE 4.71f
#define FRAMECOUNT 10000

//#define STARTINGDIST ((float)2*RADIUS+1.3f)
#define STARTINGDIST (COLLISIONDIST*1.1f)
#define STARTINGHEIGHT BOUND+RADIUS
#define STARTINGX 	((IMAGEWFULL - 2 * BOUND - STARTINGDIST * PARTICLECOUNTX) / 2)
//#define STARTINGX BOUND+RADIUS

#define LOGOW 240
#define LOGOH 234ULL
#define DPPX 16
#define PARTICLECOUNTX (DPPX*LOGOW)
#define PARTICLECOUNTY (DPPX*LOGOH)
#define XDIFFPOS 0.85
#define STARTINGDISTX STARTINGDIST
#define STARTINGDISTY STARTINGDIST
//#define STARTINGDISTX (STARTINGDIST*XDIFFPOS*2+2)
//#define STARTINGDISTY (STARTINGDIST*1.01f+2)

//#define PARTICLECOUNTX ((int)((IMAGEWFULL-2*BOUND-STARTINGDISTX-2*RADIUS)/STARTINGDISTX))
//#define PARTICLECOUNTY ((int)(2*(IMAGEHFULL-2*BOUND)/STARTINGDISTY))
#define PARTICLECOUNT (PARTICLECOUNTX*PARTICLECOUNTY)
#define HEXAGONALSTART false

#define SPEEDFACTORY 0.01f
#define SPEEDFACTORX 0.01f

#define THREADCOUNT 16

#define BLOCKSIZE 256

#define MAXBIT (LOGGRIDSIZE*2+1)

#define DRAWJPG 1

#define BACKGROUND 0