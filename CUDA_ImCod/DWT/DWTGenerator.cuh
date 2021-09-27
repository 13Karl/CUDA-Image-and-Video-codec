#ifndef DWTGENERATOR_CUH
#define DWTGENERATOR_CUH

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include "DWTGenerator.hpp"
#include "../Image/Image.hpp"
#include <iostream>

//Weights for the coefficients in each lifting step for the 5/3 and the 9/7, and normalization weights for the 9/7. Precompiled as these may not change ever.
#define LIFTING_STEPS_I53_1 	0.5f 
#define LIFTING_STEPS_I53_2 	0.25f

#define LIFTING_STEPS_I97_1 	-1.586134342059924f 
#define LIFTING_STEPS_I97_2 	-0.052980118572961f
#define LIFTING_STEPS_I97_3  	0.882911075530934f
#define LIFTING_STEPS_I97_4 	0.443506852043971f

#define NORMALIZATION_I97_1 	1.230174104914001f
#define NORMALIZATION_I97_2		0.812893066f

//This will be removed in the future
#define SHARED_MEMORY_STRIDE 0
#define VOLATILE		

#define OVERLAP_LOSSY 8
#define OVERLAP_LOSSLESS 4

//Size of the number of columns that each thread computes (columns inside the data block).
#define NELEMENTS_THREAD_X 			2

//Size of the lenght in samples of each data block. The width is fixed to 32 threads per warp by 2 samples for each thread = 64.
#if !defined(NELEMENTS_THREAD_Y)
#define NELEMENTS_THREAD_Y 	18
#endif

//Thread block size used in the forward DWT
#if !defined(NTHREADSBLOCK_DWT_F)
#define NTHREADSBLOCK_DWT_F		128
#endif

//Thread block size used in the reverse DWT
#if !defined(NTHREADSBLOCK_DWT_R)
#define NTHREADSBLOCK_DWT_R		128
#endif

//Warp size fixed to 32 in the current CUDA architectures (Pascal)
#define WARPSIZE	32

//Aux Function#ifndef BPCENGINE_CUH
#ifndef GPU_HANDLE_ERROR
#define GPU_HANDLE_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#endif
#ifndef KERNEL_ERROR_HANDLER
#define KERNEL_ERROR_HANDLER { cudaKernelAssert( __FILE__, __LINE__);}
#endif
//REVISAR CLASS/TYPENAME/TEMPLATE
template<class T, class Y>
class DWTEngine
{
public:
	

	//Constructor
	DWTEngine(Image* image, int numberOfWaveletLevels, int olap, bool isLossy, float qSize);

	//Kernel launchers
	void DWTForward(T* DDataInitial, T* DDataFinal, cudaStream_t mainStream);
	void DWTForwardChar(unsigned char* DDataInitial, T* DDataFinal, cudaStream_t mainStream);
	void DWTReverse(int* DDataInitial, T* DDataFinal, cudaStream_t mainStream);
	

	//Initialize/correct functions
	__device__  void initializeCoordinates(int DSizeCurrentX, int DSizeInitialX, int DSizeCurrentY, int* TCoordinateX, int* TCoordinateY, int* TCoordinate, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int laneID, int warpID, int nWarpsX, int nWarpsY, int warpWorkY, int LLOffset, int specialLevel, int overlap);
	__device__  void incorrectBorderValues(int laneID, int warpID, int nWarpsX, int nWarpsY, int* incorrectHorizontal, int* incorrectVerticalTop, int* incorrectVerticalBottom, int overlap);
	
	//Read block
	__device__  void readBlock2(Y* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate);
	__device__  void readBlock2Char(uchar2* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate);
	//Vertical filtering
	__device__  void verticalFilterForward(T* TData, int TDSizeY, int TDSizeX);
	__device__  void verticalFilterReverse(T* TData, int TDSizeY, int TDSizeX);
	__device__  void verticalFilterForwardLossy(T* TData, int TDSizeY, int TDSizeX);
	__device__  void verticalFilterReverseLossy(T* TData, int TDSizeY, int TDSizeX);

	//Horizontal filtering - Shuffle instructions
	__device__  void horizontalFilterForwardShuffle(T* TData, int TDSizeY, int TDSizeX);
	__device__  void horizontalFilterReverseShuffle(T* TData, int TDSizeY, int TDSizeX);
	__device__  void horizontalFilterForwardLossyShuffle(T* TData, int TDSizeY, int TDSizeX);
	__device__  void horizontalFilterReverseLossyShuffle(T* TData, int TDSizeY, int TDSizeX);

	__device__  void writeBlockScheduler(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int incorrectVerticalTop, int incorrectVerticalBottom, int overlap);
	__device__  void writeBlockSchedulerLossy(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int incorrectVerticalTop, int incorrectVerticalBottom, int overlap);
	__device__  void writeSubbandsScheduler(T* data, int DSizeCurrentX, int DSizeInitialX, T* TData, int TDSizeY, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int lastLevel, int incorrectVerticalTop, int incorrectVerticalBottom, int overlap, int currentWLevel, bool isLossy, float qSize);
	__device__  void readSubbands(int* data, int DSizeCurrentX, int DSizeInitialX, T* TData, int TDSizeY, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH);
	__device__  void readSubbandsLLAux(int* data, T* dataLL, int DSizeCurrentX, int DSizeInitialX, T* TData, int TDSizeY, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH);
	__device__  void readSubbandsLossy(int* data, int DSizeCurrentX, int DSizeInitialX, T* TData, int TDSizeY, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int wLevel, float qSize, float reconstructionFactor);
	__device__  void readSubbandsLLAuxLossy(int* data, T* dataLL, int DSizeCurrentX, int DSizeInitialX, T* TData, int TDSizeY, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int wLevel, float qSize, float reconstructionFactor);
private:

	//Error handling
	inline void gpuAssert(cudaError_t err, const char *file, int line);
	void cudaKernelAssert(const char *, int);

	//Lifting steps
	__device__  void liftingStepOne53Forward(T a, VOLATILE T* b, T c);
	__device__  void liftingStepTwo53Forward(T a, VOLATILE T* b, T c);
	__device__  void liftingStepOne53Reverse(T a, VOLATILE T* b, T c);
	__device__  void liftingStepTwo53Reverse(T a, VOLATILE T* b, T c);
	__device__  void liftingStepOne97Forward(T a, VOLATILE T* b, T c);
	__device__  void liftingStepTwo97Forward(T a, VOLATILE T* b, T c);
	__device__  void liftingStepThree97Forward(T a, VOLATILE T* b, T c);
	__device__  void liftingStepFour97Forward(T a, VOLATILE T* b, T c);
	__device__  void liftingStepOne97Reverse(T a, VOLATILE T* b, T c);
	__device__  void liftingStepTwo97Reverse(T a, VOLATILE T* b, T c);
	__device__  void liftingStepThree97Reverse(T a, VOLATILE T* b, T c);
	__device__  void liftingStepFour97Reverse(T a, VOLATILE T* b, T c);

	//Update coordinates functions
	__device__  void updateSubbandsCoordinates(int DSizeCurrentX, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH);
	__device__  void updateSubbandsCoordinatesLLAux(int DSizeCurrentX, int DSizeInitialX, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH);
	__device__  void updateSubbandsCoordinatesScheduler(int DSizeCurrentX, int DSizeInitialX, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int lastLevel);

	//Read block
	__device__  void readBlock(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate);

	//Writing the subband (HH, HL, LH, LL)
	__device__  void writeSubbands(T* data, int DSizeInitialX, int DSizeCurrentX, T* TData, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int* index, int lastLevel, int currentWLevel, bool isLossy, float qSize);
	__device__  void writeSubbandsTop(T* data, int DSizeCurrentX, int DSizeInitialX, T* TData, int TDSizeY, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int lastLevel, int overlap, int currentWLevel, bool isLossy, float qSize);
	__device__  void writeSubbandsMiddle(T* data, int DSizeCurrentX, int DSizeInitialX, T* TData, int TDSizeY, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int lastLevel, int overlap, int currentWLevel, bool isLossy, float qSize);
	__device__  void writeSubbandsBottom(T* data, int DSizeCurrentX, int DSizeInitialX, T* TData, int TDSizeY, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int lastLevel, int overlap, int currentWLevel, bool isLossy, float qSize);

	//Read subbands
	__device__  void readSubbandsIteration(int* data, int DSizeCurrentX, T* TData, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int* index);
	__device__  void readSubbandsIterationLossy(int* data, int DSizeCurrentX, T* TData, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int* index, int wLevel, float qSize, float reconstructionFactor);

	//Writing the blocks
	//It is duplicated as the "asm" instruction is per data type. T and Y are not know at compilation time; due to this restriction, the functions must be duplicated.
	__device__  void writeBlockInt1(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int* index);
	__device__  void STTwoLossy(Y* a, T b, T c);
	__device__  void writeBlockInt2Lossy(Y* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int* index);
	__device__  void writeBlockLossy(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int* index);
	__device__  void writeBlockTopLossy(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int overlap);
	__device__  void writeBlockMiddleLossy(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int overlap);
	__device__  void writeBlockBottomLossy(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int overlap);
	__device__  void STTwo(Y* a, T b, T c);
	__device__  void writeBlockInt2(Y* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int* index);
	__device__  void writeBlock(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int* index);
	__device__  void writeBlockTop(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int overlap);
	__device__  void writeBlockMiddle(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int overlap);
	__device__  void writeBlockBottom(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int overlap);


	//Variables
	Image *_img;
	int _numberOfWaveletLevels;
	bool _isLossy;
	int _currentWLevel;
	float _qSize;
	int _overLap;


};

//QSteps for Lossy Encoding
__device__ float const _DWaveletQSteps[10][4] = {
	{ 1.965908f,1.0112865f,1.0112865f,0.52021784f },
	{ 4.1224113f,1.9968134f,1.9968134f,0.96721643f },
	{ 8.416739f,4.1833673f,4.1833673f,2.0792568f },
	{ 16.935543f,8.534108f,8.534108f,4.3004827f },
	{ 33.924816f,17.166693f,17.166693f,8.686718f },
	{ 67.87687f,34.385098f,34.385098f,17.41882f },
	{ 135.76744f,68.7964f,68.7964f,34.860676f },
	{ 271.5416f,137.60588f,137.60588f,69.73287f },
	{ 543.0866f,275.21814f,275.21814f,139.47136f },
	{ 1086.1624f,550.43286f,550.43286f,278.94202f }
};

//Main kernels - Due to limitations in the CUDA Compiler, __global__ functions cannot be included in C++ classes.
template<class T, class Y>
__global__ void kernelDWTForwardLossy(T* deviceOriginalImage, T* deviceResultImage, int DSizeCurrentX, int DSizeInitialX, int DSizeCurrentY, int nWarpsX, int nWarpsY, int warpWorkY, int nWarpsBlock, int writeLLOffset, int lastLevel, int write, int wLevel, float qSize, DWTEngine<T, Y>* DWTEng);
template<class T, class Y>
__global__ void kernelDWTForwardLossyChar(unsigned char* deviceOriginalImage, T* deviceResultImage, int DSizeCurrentX, int DSizeInitialX, int DSizeCurrentY, int nWarpsX, int nWarpsY, int warpWorkY, int nWarpsBlock, int writeLLOffset, int lastLevel, int write, int wLevel, float qSize, DWTEngine<T, Y>* DWTEng);
template<class T, class Y>
__global__ void kernelDWTReverseLossy(int* deviceOriginalImage, T* deviceResultImage, int DSizeCurrentX, int DSizeInitialX, int DSizeCurrentY, int nWarpsX, int nWarpsY, int warpWorkY, int nWarpsBlock, int readLLOffset, int writeOffset, int firstLevel, int write, int wLevel, float qSize, DWTEngine<T, Y>* DWTEng);
template<class T, class Y>
__global__ void kernelDWTForward(T* deviceOriginalImage, T* deviceResultImage, int DSizeCurrentX, int DSizeInitialX, int DSizeCurrentY, int nWarpsX, int nWarpsY, int warpWorkY, int nWarpsBlock, int writeLLOffset, int lastLevel, int write, DWTEngine<T, Y>* DWTEng);
template<class T, class Y>
__global__ void kernelDWTReverse(int* deviceOriginalImage, T* deviceResultImage, int DSizeCurrentX, int DSizeInitialX, int DSizeCurrentY, int nWarpsX, int nWarpsY, int warpWorkY, int nWarpsBlock, int readLLOffset, int writeOffset, int firstLevel, int write, DWTEngine<T, Y>* DWTEng);
template<class T, class Y>
__global__ void kernelDWTForwardChar(unsigned char* deviceOriginalImage, T* deviceResultImage, int DSizeCurrentX, int DSizeInitialX, int DSizeCurrentY, int nWarpsX, int nWarpsY, int warpWorkY, int nWarpsBlock, int writeLLOffset, int lastLevel, int write, DWTEngine<T, Y>* DWTEng);

//The .cu file is included here so we can have the Declarations and Definitions of the DWTEngine class splitted in two different files. Otherwise, the definition have to be implicitly declared
//inside the class declaration.

#include "DWTGenerator.cu"
#endif