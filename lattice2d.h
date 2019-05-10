//the default settings
// #define setLatticeDim 2048
// #define setDelta 1  //unit: um, useless for now.
// #define setNumOfCells 1000
// #define setGelConcentration 0.3
// #define setTumblingRate 0
// #define setHoppingRate 10

// static const int setLatticeDim=512*4;
// static const double setDelta=1;  //unit: um, useless for now.
// static const int setNumOfCells=1000;
// static const double setGelConcentration=0.3;
// static const double setTumblingRate=10;
// static const double setHoppingRate=10;
//
// //two constant for speed
// static const int setNumSite=setLatticeDim*setLatticeDim;
// static const double setdiagHoppingRate=setHoppingRate/1.414213562373095;

//#include <cuda_runtime.h>
#include <cuda.h>
//#include <device_launch_parameters.h>
#include <curand.h>
//#include <stdint.h>
//#include <random>
using namespace std;

typedef struct
{
  int posX;     //position
  int posY;
  int deltaX;   //displacement
  int deltaY;
  int directionX;   //direction
  int directionY;
  int ifMoving;     //bool value, true if cell is moving
  int escapeDirection;
  //int havingHopped;
  //int count_return;
  //double first_tblock;
  //double last_tblock;
  //double last_treturn;
  //double stat_treturn;
  //int whereInMove;  //the index of this cell in movingEcoli array
} ecoli;

typedef struct
{
  double totalTime;
  double GelConcentration;  //volume concentration of the gel
  double TumblingRate;
  double HoppingRate;
  int LatticeDim;           //number of sites in a row
  int NumOfCells;           //number of cells
  int NumSite;              //total number of sites
} status;  //status and parameters of the system

//generate a lattices with obstruction. 0 denotes empty site, and 1 denotes blocked site.
void generateLattice(int* lattice, const status* pstatus);

//put particles on lattices. lattice[idx]==2 means the site is occupied by a cell.
__global__ void putParticles(int* lattice, ecoli* ecoliList, const status* pstatus, const unsigned int* intRand, unsigned int* offsetListInt, const int estNumIntRand);

//modifying movingEcoliEdge and movingEcoliDiag after one event
__device__ void checkIfBlockedSingleCell(const status* pstatus, const int* lattice, ecoli* ecoliList);

//initialize pstatus with the pre-setting values
void setDefaultStatus(status* pstatus);

//uniformly sample a new direction of bacteria.
__device__ void sampleDirection(ecoli* cellList, const int cellIdx, const unsigned int* intRand, unsigned int* offsetListInt, const int estNumIntRand);

//save a binary snapshot, containing all the information of the system at current time
void snapshot(const int* lattice, const ecoli* ecoliList, const status* pstatus, const int fileIndex);


void snapshotLattices(const int* lattice, const ecoli* ecoliList, const status* pstatus, const int fileIndex);
void snapshotParticles(const int* lattice, const ecoli* ecoliList, const status* pstatus, const int fileIndex);

//read the time and parameters of the system from a binary snapshot
void readSnapshotHead(status* pstatus, const int fileIndex);

//read the data of lattices and cells from a binary snapshot
void readSnapshotData(int* lattice, ecoli* ecoliList, const status* pstatus, const int fileIndex);

//simulate one gillespie step, return denote the type of event: 0:tumble, 1:hop, 2:blocked
__global__ void simulate(int* lattice, ecoli* ecoliList, status* pstatus, const double timeLapse,
	const unsigned int* intRand, unsigned int* offsetListInt, const int estNumIntRand,
	const double* doubleRand, unsigned int* offsetListDoub, const int estNumDoubRand, const int step);

__global__ void calMeanSqrDisp(const ecoli* ecoliList, const status* pstatus, double* X, double* Y);
__global__ void calMeantblock(const ecoli* ecoliList, const status* pstatus, double* t, double* count);