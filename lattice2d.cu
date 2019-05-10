#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mt64.h"
//#include <omp.h>

#include "lattice2d.h"


int main(int argc, const char * argv[])
{
	clock_t it, ft;
	it=clock();

	status cstatus;   //the structure for status of current system
	setDefaultStatus(&cstatus);
	const int NumSite = cstatus.NumSite;
	const int NumOfCells = cstatus.NumOfCells;
	const int threadSize = 256;
	//const int blockSize = 2048;
	const int blockSize = (NumOfCells / threadSize) > 16384 ? 16384 : (NumOfCells / threadSize);
	const int TotalThreadsNum = threadSize*blockSize;
	//const int TotalThreadsNum = NumOfCells;
	const int cellsPerThread = NumOfCells / TotalThreadsNum;

	double totalTime = cstatus.totalTime;
	const double timeStep = 1;
	int NumStep = totalTime / timeStep;
	int fileIdx = 1;
	const double TotalRate = cstatus.HoppingRate + cstatus.TumblingRate;
	const int estNumIntRand = (cstatus.TumblingRate*timeStep > 1 ? cstatus.TumblingRate*timeStep : 1)*NumOfCells * 15;
	const int estNumDoubleRand = (timeStep>1?timeStep:1)*TotalRate*NumOfCells * 15;
	
	curandGenerator_t randGen;
	double* doubleRand=0;
	unsigned int* intRand=0;
	unsigned int* offsetListInt = new unsigned int[TotalThreadsNum];
	unsigned int* offsetListDoub = new unsigned int[TotalThreadsNum];
	for (int i = 0; i < TotalThreadsNum; ++i)
	{
		offsetListDoub[i] = i;
		offsetListInt[i] = i;
	}
	unsigned int* offsetListInt_dev=0;
	unsigned int* offsetListDoub_dev=0;

	cudaMalloc((void**)&doubleRand, estNumDoubleRand * sizeof(double));
	cudaMalloc((void**)&intRand, estNumIntRand * sizeof(unsigned int));

	cudaMalloc((void**)&offsetListInt_dev, TotalThreadsNum * sizeof(unsigned int));
	cudaMalloc((void**)&offsetListDoub_dev, TotalThreadsNum * sizeof(unsigned int));
	
	cudaMemcpy(offsetListInt_dev, offsetListInt, TotalThreadsNum * sizeof(unsigned int), cudaMemcpyHostToDevice);
	
	curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_MT19937);
	curandSetPseudoRandomGeneratorSeed(randGen, 4624ULL);
	curandGenerate(randGen, intRand, estNumIntRand);

	int* lattice=(int*) malloc(NumSite*sizeof(int));  //0: empty; -1: blocked; n: occupied by n-th cell
	ecoli* ecoliList=(ecoli*) malloc(NumOfCells*sizeof(ecoli));
	
	//printf("test\n");
	generateLattice(lattice, &cstatus);

	int* lattice_GPU = 0;
	ecoli* ecoliList_GPU = 0;
	status* status_GPU = 0;
	cudaMalloc((void**)&lattice_GPU, NumSite * sizeof(int));
	cudaMalloc((void**)&ecoliList_GPU, NumOfCells * sizeof(ecoli));
	cudaMalloc((void**)&status_GPU, sizeof(status));

	cudaMemcpy(lattice_GPU, lattice, NumSite * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(status_GPU, &cstatus, sizeof(status), cudaMemcpyHostToDevice);

	putParticles<<<blockSize, threadSize>>>(lattice_GPU, ecoliList_GPU, status_GPU, intRand, offsetListInt_dev, estNumIntRand);

	cudaMemcpy(ecoliList, ecoliList_GPU, NumOfCells * sizeof(ecoli), cudaMemcpyDeviceToHost);
	//snapshot(lattice, ecoliList, &cstatus, 0);

	double* disX_dev;
	double* disY_dev;
	cudaMalloc((void**)&disX_dev, blockSize * sizeof(double));
	cudaMalloc((void**)&disY_dev, blockSize * sizeof(double));

	double* disX = (double*) malloc(blockSize*sizeof(double));
	double* disY = (double*)malloc(blockSize * sizeof(double));


	double* latticep = (double*)malloc(NumSite * sizeof(double));  //0: empty; -1: blocked; n: occupied by n-th cell
	FILE* outfile = fopen("result.txt", "w");
	//for (int i = 0; i < 100000; ++i)
	//{
	//	curandGenerate(randGen, intRand, estNumIntRand);
	//	curandGenerateUniformDouble(randGen, doubleRand, estNumDoubleRand);
	//	cudaMemcpy(offsetListInt_dev, offsetListInt, TotalThreadsNum * sizeof(unsigned int), cudaMemcpyHostToDevice);
	//	cudaMemcpy(offsetListDoub_dev, offsetListDoub, TotalThreadsNum * sizeof(unsigned int), cudaMemcpyHostToDevice);
	//	simulate<<<blockSize, threadSize>>>(lattice_GPU, ecoliList_GPU, status_GPU, timeStep, intRand, offsetListInt_dev, estNumIntRand, doubleRand, offsetListDoub_dev, estNumDoubleRand, i);
	//}
	//timeStep = 0.01;
	for (int i = 0; i < NumStep; ++i)
	{
		curandGenerate(randGen, intRand, estNumIntRand);
		curandGenerateUniformDouble(randGen, doubleRand, estNumDoubleRand);
		cudaMemcpy(offsetListInt_dev, offsetListInt, TotalThreadsNum * sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(offsetListDoub_dev, offsetListDoub, TotalThreadsNum * sizeof(unsigned int), cudaMemcpyHostToDevice);
		simulate<<<blockSize, threadSize>>>(lattice_GPU, ecoliList_GPU, status_GPU, timeStep, intRand, offsetListInt_dev, estNumIntRand, doubleRand, offsetListDoub_dev, estNumDoubleRand, i);
		//cudaMemcpy(lattice, lattice_GPU, NumSite * sizeof(int), cudaMemcpyDeviceToHost);
		//cudaMemcpy(&cstatus, status_GPU, sizeof(status), cudaMemcpyDeviceToHost);
		//cudaMemcpy(ecoliList, ecoliList_GPU, NumOfCells * sizeof(ecoli), cudaMemcpyDeviceToHost);
		//for (int i = 0; i < cstatus.NumSite; ++i)
		//{
		//	latticep[i]=0;
		//}
		//for (int i = 0; i < cstatus.NumOfCells; ++i)
		//{
		//	const int posx = ecoliList[i].posX;
		//	const int posy = ecoliList[i].posY;
		//	const int idx = posy*cstatus.LatticeDim + posx;
		//	++latticep[idx];
		//}

		//char filename[5]="";  //file name will be like s0000.bin
		//char buffer[5]="";
		//sprintf(buffer, "%d", fileIdx);
		//strcat(filename, buffer);
		//FILE* pOutputFile = fopen(filename, "wb");
		//fwrite(latticep, sizeof(double), cstatus.NumSite, pOutputFile);
		//fclose(pOutputFile);
		////printf("%d\n", ecoliList[0].posY);
		////snapshotParticles(lattice, ecoliList, &cstatus, fileIdx);
		//++fileIdx;

		calMeanSqrDisp<<<blockSize, threadSize>>>(ecoliList_GPU, status_GPU, disX_dev, disY_dev);
		cudaMemcpy(disX, disX_dev, blockSize * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(disY, disY_dev, blockSize * sizeof(double), cudaMemcpyDeviceToHost);
		fprintf(outfile, "%f ", (i+1)*timeStep);
		double sumx=0;
		double sumy=0;
		for (int j = 0; j < blockSize; ++j)
		{
			sumx += disX[j];
		}
		for (int j = 0; j < blockSize; ++j)
		{
			sumy += disY[j];
		}
		sumx = sumx / NumOfCells;
		sumy = sumy / NumOfCells;
		fprintf(outfile, "%f %f \n", sumx, sumy);
	}
/*
	double* totaltblock_dev;
	double* count_dev;
	cudaMalloc((void**)&totaltblock_dev, blockSize * sizeof(double));
	cudaMalloc((void**)&count_dev, blockSize * sizeof(double));

	double* totaltblock = new double[blockSize];
	double* count = new double[blockSize];

	calMeantblock<<<blockSize, threadSize>>>(ecoliList_GPU, status_GPU, totaltblock_dev, count_dev);
	cudaMemcpy(totaltblock, totaltblock_dev, blockSize * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(count, count_dev, blockSize * sizeof(double), cudaMemcpyDeviceToHost);
	double sumt = 0;
	double sumcount = 0;
	for (int j = 0; j < blockSize; ++j)
	{
		sumt += totaltblock[j];
		sumcount += count[j];
	}
	printf("%f \n", sumt/sumcount);*/

	//cudaMemcpy(lattice, lattice_GPU, NumSite * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&cstatus, status_GPU, sizeof(status), cudaMemcpyDeviceToHost);
	//cudaMemcpy(ecoliList, ecoliList_GPU, NumOfCells * sizeof(ecoli), cudaMemcpyDeviceToHost);

	//int* latticeX = (int*)malloc(NumSite * sizeof(int));  //0: empty; -1: blocked; n: occupied by n-th cell
	//int* latticeY = (int*)malloc(NumSite * sizeof(int));
	//int* latticeF = (int*)malloc(NumSite * sizeof(int));  //0: empty; -1: blocked; n: occupied by n-th cell
	//int* latticeB = (int*)malloc(NumSite * sizeof(int));
	//for (int i = 0; i < cstatus.NumSite; ++i)
	//{
	//	latticeX[i] = 0;
	//	latticeY[i] = 0;
	//	latticeF[i] = 0;
	//	latticeB[i] = 0;
	//}
	//for (int i = 0; i < cstatus.NumOfCells; ++i)
	//{
	//	const int posx = ecoliList[i].posX;
	//	const int posy = ecoliList[i].posY;
	//	const int idx = posy*cstatus.LatticeDim + posx;
	//	if (ecoliList[i].ifMoving)
	//	{
	//		latticeX[idx] += ecoliList[i].directionX;
	//		latticeY[idx] += ecoliList[i].directionY;
	//		++latticeF[idx];
	//	}
	//	else
	//	{
	//		++latticeB[idx];
	//	}
	//}
	//for (int i = 0; i < cstatus.NumSite; ++i)
	//{
	//	fprintf(outfile, "%d ", latticeX[i]);
	//}
	//fprintf(outfile, "\n");
	//for (int i = 0; i < cstatus.NumSite; ++i)
	//{
	//	fprintf(outfile, "%d ", latticeY[i]);
	//}
	//fprintf(outfile, "\n");
	//for (int i = 0; i < cstatus.NumSite; ++i)
	//{
	//	fprintf(outfile, "%d ", latticeF[i]);
	//}
	//fprintf(outfile, "\n");
	//for (int i = 0; i < cstatus.NumSite; ++i)
	//{
	//	fprintf(outfile, "%d ", latticeB[i]);
	//}
	//fprintf(outfile, "\n");
	//for (int i = 0; i < cstatus.NumSite; ++i)
	//{
	//	fprintf(outfile, "%d ", lattice[i]);
	//}
	//free(latticeX);
	//free(latticeY);
	//free(latticeF);
	//free(latticeB);
	free(lattice);
	free(ecoliList);
	free(offsetListDoub);
	free(offsetListInt);
	free(disX);
	free(disY);
	cudaFree(disX_dev);
	cudaFree(disY_dev);
	cudaFree(lattice_GPU);
	cudaFree(ecoliList_GPU);
	cudaFree(status_GPU);
	cudaFree(intRand);
	cudaFree(doubleRand);
	cudaFree(offsetListDoub_dev);
	cudaFree(offsetListInt_dev);
	curandDestroyGenerator(randGen);
	ft=clock();
	printf("%f\n",(double)(ft-it)/ (double)CLOCKS_PER_SEC);
	return 0;
}

void snapshot(const int* lattice, const ecoli* ecoliList, const status* pstatus, const int fileIndex)
{
	char filename[10];  //file name will be like s0000.bin
	char buffer[5];
	sprintf(buffer, "%04d", fileIndex);
	strcpy(filename, "s");
	strcat(filename, buffer);
	strcat(filename, ".bin");
	FILE* pOutputFile = fopen(filename, "wb");

	fwrite(pstatus, sizeof(status), 1, pOutputFile);

	fwrite(lattice, sizeof(int), pstatus->NumSite, pOutputFile);
	fwrite(ecoliList, sizeof(ecoli), pstatus->NumOfCells, pOutputFile);
	fclose(pOutputFile);
}

void snapshotLattices(const int* lattice, const ecoli* ecoliList, const status* pstatus, const int fileIndex)
{
	char filename[10];  //file name will be like s0000.bin
	char buffer[5];
	sprintf(buffer, "%04d", fileIndex);
	strcpy(filename, "l");
	strcat(filename, buffer);
	strcat(filename, ".bin");
	FILE* pOutputFile = fopen(filename, "wb");

	fwrite(pstatus, sizeof(status), 1, pOutputFile);

	fwrite(lattice, sizeof(int), pstatus->NumSite, pOutputFile);
	fclose(pOutputFile);
}

void snapshotParticles(const int* lattice, const ecoli* ecoliList, const status* pstatus, const int fileIndex)
{
	char filename[10];  //file name will be like s0000.bin
	char buffer[5];
	sprintf(buffer, "%04d", fileIndex);
	strcpy(filename, "p");
	strcat(filename, buffer);
	strcat(filename, ".bin");
	FILE* pOutputFile = fopen(filename, "wb");

	fwrite(pstatus, sizeof(status), 1, pOutputFile);

	fwrite(ecoliList, sizeof(ecoli), pstatus->NumOfCells, pOutputFile);
	fclose(pOutputFile);
}

//read the time and parameters of the system from a binary snapshot
void readSnapshotHead(status* pstatus, const int fileIndex)
{
	char filename[10];
	char buffer[5];
	sprintf(buffer, "%04d", fileIndex);
	strcpy(filename, "s");
	strcat(filename, buffer);
	strcat(filename, ".bin");
	FILE* pInputFile = fopen(filename, "rb");

	fread(pstatus, sizeof(status), 1, pInputFile);

	fclose(pInputFile);
}

//read the data of lattices and cells from a binary snapshot
void readSnapshotData(int* lattice, ecoli* ecoliList, const status* pstatus, const int fileIndex)
{
	char filename[10];
	char buffer[5];
	sprintf(buffer, "%04d", fileIndex);
	strcpy(filename, "s");
	strcat(filename, buffer);
	strcat(filename, ".bin");
	FILE* pInputFile = fopen(filename, "rb");

	fseek(pInputFile, sizeof(status), SEEK_SET);
	fread(lattice, sizeof(int), pstatus->NumSite, pInputFile);
	fread(ecoliList, sizeof(ecoli), pstatus->NumOfCells, pInputFile);

	fclose(pInputFile);
}
