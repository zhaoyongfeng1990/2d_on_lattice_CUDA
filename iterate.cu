#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "lattice2d.h"
#include "mt64.h"

#define NN 312
#define MM 156
#define MATRIX_A UINT64_C(0xB5026F5AA96619E9)
#define UM UINT64_C(0xFFFFFFFF80000000) /* Most significant 33 bits */
#define LM UINT64_C(0x7FFFFFFF) /* Least significant 31 bits */

/* The array for the state vector */
static uint64_t mt_host[NN];
/* mti_host==NN+1 means mt[NN] is not initialized */
static int mti_host = NN + 1;

/* initializes mt[NN] with a seed */
void init_genrand64_host(uint64_t seed)
{
	mt_host[0] = seed;
	for (mti_host = 1; mti_host<NN; mti_host++)
		mt_host[mti_host] = (UINT64_C(6364136223846793005) * (mt_host[mti_host - 1] ^ (mt_host[mti_host - 1] >> 62)) + mti_host);
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
void init_by_array64_host(uint64_t init_key[],
	uint64_t key_length)
{
	unsigned int i, j;
	uint64_t k;
	init_genrand64_host(UINT64_C(19650218));
	i = 1; j = 0;
	k = (NN>key_length ? NN : key_length);
	for (; k; k--) {
		mt_host[i] = (mt_host[i] ^ ((mt_host[i - 1] ^ (mt_host[i - 1] >> 62)) * UINT64_C(3935559000370003845)))
			+ init_key[j] + j; /* non linear */
		i++; j++;
		if (i >= NN) { mt_host[0] = mt_host[NN - 1]; i = 1; }
		if (j >= key_length) j = 0;
	}
	for (k = NN - 1; k; k--) {
		mt_host[i] = (mt_host[i] ^ ((mt_host[i - 1] ^ (mt_host[i - 1] >> 62)) * UINT64_C(2862933555777941757)))
			- i; /* non linear */
		i++;
		if (i >= NN) { mt_host[0] = mt_host[NN - 1]; i = 1; }
	}

	mt_host[0] = UINT64_C(1) << 63; /* MSB is 1; assuring non-zero initial array */
}

/* generates a random number on [0, 2^64-1]-interval */
uint64_t genrand64_int64_host(void)
{
	int i;
	uint64_t x;
	static uint64_t mag01[2] = { UINT64_C(0), MATRIX_A };

	if (mti_host >= NN) { /* generate NN words at one time */

						  /* if init_genrand64() has not been called, */
						  /* a default initial seed is used     */
		if (mti_host == NN + 1)
			init_genrand64_host(UINT64_C(5489));

		for (i = 0; i<NN - MM; i++) {
			x = (mt_host[i] & UM) | (mt_host[i + 1] & LM);
			mt_host[i] = mt_host[i + MM] ^ (x >> 1) ^ mag01[(int)(x&UINT64_C(1))];
		}
		for (; i<NN - 1; i++) {
			x = (mt_host[i] & UM) | (mt_host[i + 1] & LM);
			mt_host[i] = mt_host[i + (MM - NN)] ^ (x >> 1) ^ mag01[(int)(x&UINT64_C(1))];
		}
		x = (mt_host[NN - 1] & UM) | (mt_host[0] & LM);
		mt_host[NN - 1] = mt_host[MM - 1] ^ (x >> 1) ^ mag01[(int)(x&UINT64_C(1))];

		mti_host = 0;
	}

	x = mt_host[mti_host++];

	x ^= (x >> 29) & UINT64_C(0x5555555555555555);
	x ^= (x << 17) & UINT64_C(0x71D67FFFEDA60000);
	x ^= (x << 37) & UINT64_C(0xFFF7EEE000000000);
	x ^= (x >> 43);

	return x;
}

/* generates a random number on [0, 2^63-1]-interval */
int64_t genrand64_int63_host(void)
{
	return (int64_t)(genrand64_int64_host() >> 1);
}

/* generates a random number on [0,1]-real-interval */
double genrand64_real1_host(void)
{
	return (genrand64_int64_host() >> 11) * (1.0 / 9007199254740991.0);
}

/* generates a random number on [0,1)-real-interval */
double genrand64_real2_host(void)
{
	return (genrand64_int64_host() >> 11) * (1.0 / 9007199254740992.0);
}

/* generates a random number on (0,1)-real-interval */
double genrand64_real3_host(void)
{
	return ((genrand64_int64_host() >> 12) + 0.5) * (1.0 / 4503599627370496.0);
}

//generate a lattices with obstruction. 0 denotes empty site, and 1 denotes blocked site.
void generateLattice(int* lattice, const status* pstatus)
{
	init_genrand64_host((uint64_t)clock());
	double GelConcentration = pstatus->GelConcentration;
	int NumSite = pstatus->NumSite;
	for (int i = 0; i < NumSite; ++i)
	{
		if (genrand64_real1_host()<GelConcentration)   //each site has probability GelConcentration to be blocked.
		{
			lattice[i] = -1;
		}
		else
		{
			lattice[i] = 0;
		}
	}
	//for (int i = NumSite / 2; i < NumSite; ++i)
	//{
	//	lattice[i] = 0;
	//}
}

//uniformly sample a new direction of bacteria.
__device__ void sampleDirection(ecoli* cellList, const int cellIdx, const unsigned int* intRand, unsigned int* offsetListInt, const int estNumIntRand)
{
	int bIdx = blockIdx.x;
	int tIdx = threadIdx.x;
	int idx = bIdx*blockDim.x + tIdx;
	int numThread = blockDim.x*gridDim.x;

	int randomDirection = intRand[offsetListInt[idx]]%4; //a random number from 0 to 7
	if (offsetListInt[idx]<estNumIntRand)
	{
		offsetListInt[idx] += numThread;
	}
	else
	{
		printf("Running out of integer random numbers !\n");
		return;
	}

	switch (randomDirection)
	{
	case 0:
		cellList[cellIdx].directionX = 1;
		cellList[cellIdx].directionY = 0;
		break;
	case 1:
		cellList[cellIdx].directionX = -1;
		cellList[cellIdx].directionY = 0;
		break;
	case 2:
		cellList[cellIdx].directionX = 0;
		cellList[cellIdx].directionY = 1;
		break;	 
	case 3:		 
		cellList[cellIdx].directionX = 0;
		cellList[cellIdx].directionY = -1;
		break;
	}
}

//put particles on lattices.
__global__ void putParticles(int* lattice, ecoli* ecoliList, const status* pstatus, const unsigned int* intRand, unsigned int* offsetListInt, const int estNumIntRand)
{
	const int bIdx = blockIdx.x;
	const int tIdx = threadIdx.x;
	const int idx = bIdx*blockDim.x + tIdx;
	const int numThread = blockDim.x*gridDim.x;

	const int LatticeDim = pstatus->LatticeDim;
	const int NumOfCells = pstatus->NumOfCells;

	for (int i = idx; i < NumOfCells; i+=numThread)
	{
		//printf("%d\n", offsetListInt[0]);
		int randomX, randomY, idxLattice;  //random coordinates and index in lattice array
		do
		{
			randomX = intRand[offsetListInt[idx]] % LatticeDim;; //a random number from 0 to 7
			if (offsetListInt[idx]<estNumIntRand)
			{
				offsetListInt[idx] += numThread;
			}
			else
			{
				printf("Running out of integer random numbers !\n");
				return;
			}
			randomY = intRand[offsetListInt[idx]] % LatticeDim;; //a random number from 0 to 7
			if (offsetListInt[idx]<estNumIntRand)
			{
				offsetListInt[idx] += numThread;
			}
			else
			{
				printf("Running out of integer random numbers !\n");
				return;
			}
			//randomX = genrand64_int64() % LatticeDim;
			//randomY = genrand64_int64() % LatticeDim;
			idxLattice = randomY*LatticeDim + randomX;   //index in lattice array
		} while (0 != lattice[idxLattice]);

		//Non-interacting particles: comment this line
		//lattice[idx]=i;   //the site is occupied
		//////

		ecoliList[i].posX = randomX;
		ecoliList[i].posY = randomY;
		ecoliList[i].deltaX = 0;
		ecoliList[i].deltaY = 0;
		sampleDirection(ecoliList, i, intRand, offsetListInt, estNumIntRand);
		//ecoliList[i].havingHopped = 1;
		ecoliList[i].ifMoving = 1;
		//checkIfBlockedSingleCell(pstatus, lattice, ecoliList);
	}
}

//initialize pstatus with the pre-setting values
void setDefaultStatus(status* pstatus)
{
	FILE* parafile = fopen("parameters.txt", "r");

	int setLatticeDim;
	int setNumOfCells;
	double setGelConcentration;
	double setTumblingRate;
	double setHoppingRate;
	double setTotalTime;
	fscanf(parafile, "%lf\n", &setTotalTime);
	fscanf(parafile, "%d\n", &setLatticeDim);
	fscanf(parafile, "%d\n", &setNumOfCells);
	fscanf(parafile, "%lf\n", &setGelConcentration);
	fscanf(parafile, "%lf\n", &setTumblingRate);
	fscanf(parafile, "%lf\n", &setHoppingRate);

	fclose(parafile);
	int setNumSite = setLatticeDim*setLatticeDim;

	pstatus->totalTime = setTotalTime;
	pstatus->GelConcentration = setGelConcentration;
	pstatus->TumblingRate = setTumblingRate;
	pstatus->HoppingRate = setHoppingRate;
	pstatus->LatticeDim = setLatticeDim;
	pstatus->NumOfCells = setNumOfCells;
	pstatus->NumSite = setNumSite;
}

__device__ void checkIfBlockedSingleCell(const status* pstatus, const int* lattice, ecoli* ecoliList)
{
	const int bIdx = blockIdx.x;
	const int tIdx = threadIdx.x;
	const int idx = bIdx*blockDim.x + tIdx;
	const int numThread = blockDim.x*gridDim.x;
	const int NumOfCells = pstatus->NumOfCells;
	const int LatticeDim = pstatus->LatticeDim;

	for (int i = idx; i < NumOfCells; i += numThread)
	{
		const int dX = -ecoliList[i].directionX;
		const int dY = -ecoliList[i].directionY;
		const int latticeIdx = (ecoliList[i].posY - dY)*LatticeDim + ecoliList[i].posX - dX;

		ecoliList[i].ifMoving = (lattice[latticeIdx] != -1);
		if (!ecoliList[i].ifMoving)
		{
			ecoliList[i].escapeDirection = dX * 3 + dY;
		}

		//if (ecoliList[i].ifMoving)  //cell can move before the event, it must be in movingEcoli list
		//{
		//	if (lattice[newIdx]==-1)  //now cell cannot move, deleting from corresponding movingEcoli array
		//	{
		//		ecoliList[i].ifMoving = 0;
		//	}
		//}
		//else  //cell cannot move before the event, it's not in any movingEcoli array
		//{
		//	if (0 == lattice[newIdx])  //now cell can move
		//	{
		//		ecoliList[i].ifMoving = 1;
		//	}
		//}
	}
}

//simulate one gillespie step, return denote the type of event: 0:tumble, 1:hop
__global__ void simulate(int* lattice, ecoli* ecoliList, status* pstatus, const double timeLapse,
	const unsigned int* intRand, unsigned int* offsetListInt, const int estNumIntRand,
	const double* doubleRand, unsigned int* offsetListDoub, const int estNumDoubRand, const int step)
{
	const int bIdx = blockIdx.x;
	const int tIdx = threadIdx.x;
	const int idx = bIdx*blockDim.x + tIdx;
	const int numThread = blockDim.x*gridDim.x;
	const int NumOfCells = pstatus->NumOfCells;

	const int LatticeDim = pstatus->LatticeDim;
	const double TumblingRate=pstatus->TumblingRate;
	const double HoppingRate=pstatus->HoppingRate;
	for (int i = idx; i < NumOfCells; i+=numThread)
	{
		double ctime = 0;

		double totalProb = TumblingRate + HoppingRate*ecoliList[i].ifMoving;

		double r1 = doubleRand[offsetListDoub[idx]];
		if (offsetListDoub[idx]<estNumDoubRand)
		{
			offsetListDoub[idx] += numThread;
		}
		else
		{
			printf("Running out of double random numbers !\n");
			return;
		}
		//printf("%f\n", r1);
		double tau = -log(r1) / totalProb;  //time increment

		while (ctime + tau<timeLapse)
		{
			ctime += tau;

			double threshold = doubleRand[offsetListDoub[idx]]*totalProb;
			if (offsetListDoub[idx]<estNumDoubRand)
			{
				offsetListDoub[idx] += numThread;
			}
			else
			{
				printf("Running out of double random numbers !\n");
				return;
			}
			//finding the event which is going to happen
			//printf("%f %f\n", threshold, TumblingRate);
			if (threshold <= TumblingRate)  //tumbling event
			{
				sampleDirection(ecoliList, i, intRand, offsetListInt, estNumIntRand);
				//ecoliList[i].havingHopped = 0;
				//checkIfBlockedSingleCell(pstatus, lattice, ecoliList);
				const int posX = ecoliList[i].posX;
				const int posY = ecoliList[i].posY;
				//if (!ecoliList[i].ifMoving)
				if(lattice[posY*LatticeDim+posX]==-1)
				{
					const int dX = ecoliList[i].directionX;
					const int dY = ecoliList[i].directionY;
					const int directionCode = dX * 3 + dY;
					ecoliList[i].ifMoving = (directionCode != ecoliList[i].escapeDirection);
				}
			}
			else   //hopping event
			{
				const int posX = ecoliList[i].posX;
				const int posY = ecoliList[i].posY;
				//int index = posY*LatticeDim + posX;

				//the mod operation is to make periodic boundary condition
				//plus with LatticeDim helps eliminate the possible of getting a negative value
				const int dX = ecoliList[i].directionX;
				const int dY = ecoliList[i].directionY;
				const int newX = (posX + dX + LatticeDim) % LatticeDim;
				const int newY = (posY + dY + LatticeDim) % LatticeDim;
				const int newIndex = newY*LatticeDim + newX;

				ecoliList[i].posX = newX;
				ecoliList[i].posY = newY;
				ecoliList[i].deltaX += dX;
				ecoliList[i].deltaY += dY;
				//lattice[index]=0;     //now the previous site is empty
				//lattice[newIndex]=i;  //and the new site is occupied
				//ecoliList[i].havingHopped = 1;
				//checkIfBlockedSingleCell(pstatus, lattice, ecoliList);
				ecoliList[i].ifMoving = (lattice[newIndex] != -1);
				if (!ecoliList[i].ifMoving)
				{
					ecoliList[i].escapeDirection = dX * 3 + dY;
				}
			}

			totalProb = TumblingRate + HoppingRate*ecoliList[i].ifMoving;
			r1 = doubleRand[offsetListDoub[idx]];
			if (offsetListDoub[idx]<estNumDoubRand)
			{
				offsetListDoub[idx] += numThread;
			}
			else
			{
				printf("Running out of double random numbers !\n");
				return;
			}
			//r1 = genrand64_real3();
			tau = -log(r1) / totalProb;  //time increment
		}

	}
	//printf("Thread idx: %d, simulating %d steps, exit at time %f, with increment %f.\n", bIdx, times, ctime, tau);
}

__global__ void calMeanSqrDisp(const ecoli* ecoliList, const status* pstatus, double* X, double* Y)
{
	int bIdx = blockIdx.x;
	int tIdx = threadIdx.x;
	int idx = bIdx*blockDim.x + tIdx;
	int numThread = blockDim.x*gridDim.x;
	int NumOfCells = pstatus->NumOfCells;

	__shared__ double dispX[256];
	__shared__ double dispY[256];
	dispX[tIdx] = 0;
	dispY[tIdx] = 0;

	for (int i = idx; i < NumOfCells; i += numThread)
	{
		int dx = ecoliList[i].deltaX;
		int dy = ecoliList[i].deltaY;
		dispX[tIdx] += dx*dx;
		dispY[tIdx] += dy*dy;
	}
	__syncthreads();
	if (tIdx<128)
	{
		dispX[tIdx] += dispX[tIdx + 128];
		dispY[tIdx] += dispY[tIdx + 128];
	}
	__syncthreads();
	if (tIdx<64)
	{
		dispX[tIdx] += dispX[tIdx + 64];
		dispY[tIdx] += dispY[tIdx + 64];
	}
	__syncthreads();
	if (tIdx<32)
	{
		dispX[tIdx] += dispX[tIdx + 32];
		dispY[tIdx] += dispY[tIdx + 32];
	}
	__syncthreads();
	if (tIdx<16)
	{
		dispX[tIdx] += dispX[tIdx + 16];
		dispY[tIdx] += dispY[tIdx + 16];
	}
	__syncthreads();
	if (tIdx<8)
	{
		dispX[tIdx] += dispX[tIdx + 8];
		dispY[tIdx] += dispY[tIdx + 8];
	}
	__syncthreads();
	if (tIdx<4)
	{
		dispX[tIdx] += dispX[tIdx + 4];
		dispY[tIdx] += dispY[tIdx + 4];
	}
	__syncthreads();
	if (tIdx<2)
	{
		dispX[tIdx] += dispX[tIdx + 2];
		dispY[tIdx] += dispY[tIdx + 2];
	}
	__syncthreads();
	if (tIdx<1)
	{
		dispX[tIdx] += dispX[tIdx + 1];
		dispY[tIdx] += dispY[tIdx + 1];
	}
	__syncthreads();
	if (tIdx==0)
	{
		X[bIdx] = dispX[0];
		Y[bIdx] = dispY[0];
	}
}
/*
__global__ void calMeantblock(const ecoli* ecoliList, const status* pstatus, double* t, double* count)
{
	int bIdx = blockIdx.x;
	int tIdx = threadIdx.x;
	int idx = bIdx*blockDim.x + tIdx;
	int numThread = blockDim.x*gridDim.x;
	int NumOfCells = pstatus->NumOfCells;

	__shared__ double totalt[256];
	__shared__ double totalcount[256];
	totalt[tIdx] = 0;
	totalcount[tIdx] = 0;

	for (int i = idx; i < NumOfCells; i += numThread)
	{
		double one_tblock = ecoliList[i].last_tblock - ecoliList[i].first_tblock;
		if (ecoliList[i].first_tblock > 0 )
		{
			totalt[tIdx] += one_tblock;
			totalcount[tIdx] += ecoliList[i].count_block;
		}
	}
	__syncthreads();
	if (tIdx<128)
	{
		totalt[tIdx] += totalt[tIdx + 128];
		totalcount[tIdx] += totalcount[tIdx + 128];
	}
	__syncthreads();
	if (tIdx<64)
	{
		totalt[tIdx] += totalt[tIdx + 64];
		totalcount[tIdx] += totalcount[tIdx + 64];
	}
	__syncthreads();
	if (tIdx<32)
	{
		totalt[tIdx] += totalt[tIdx + 32];
		totalcount[tIdx] += totalcount[tIdx + 32];
	}
	__syncthreads();
	if (tIdx<16)
	{
		totalt[tIdx] += totalt[tIdx + 16];
		totalcount[tIdx] += totalcount[tIdx + 16];
	}
	__syncthreads();
	if (tIdx<8)
	{
		totalt[tIdx] += totalt[tIdx + 8];
		totalcount[tIdx] += totalcount[tIdx + 8];
	}
	__syncthreads();
	if (tIdx<4)
	{
		totalt[tIdx] += totalt[tIdx + 4];
		totalcount[tIdx] += totalcount[tIdx + 4];
	}
	__syncthreads();
	if (tIdx<2)
	{
		totalt[tIdx] += totalt[tIdx + 2];
		totalcount[tIdx] += totalcount[tIdx + 2];
	}
	__syncthreads();
	if (tIdx<1)
	{
		totalt[tIdx] += totalt[tIdx + 1];
		totalcount[tIdx] += totalcount[tIdx + 1];
	}
	__syncthreads();
	if (tIdx == 0)
	{
		t[bIdx] = totalt[0];
		count[bIdx]=totalcount[0];
	}
}*/