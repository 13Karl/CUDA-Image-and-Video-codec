
#pragma once
#ifndef BPCENGINE_CUH
#define BPCENGINE_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../IO/IOManager.hpp"

typedef enum BPC {
	CODE,
	DECODE,
}BPC;

//It will be defined as a global parameter
#define VOLATILE

//Pascal fixed size
#define WARPSIZE 				32


//16 bits to code the words.
#define CODEWORD_SIZE	16

//Non-modifiable parameter
#define NELEMENTS_THREAD_X 			2	

#if !defined(CBLOCK_WIDTH)
#define CBLOCK_WIDTH	(NELEMENTS_THREAD_X*WARPSIZE)
#endif

//Non-modifiable parameter
#if !defined(CBLOCK_LENGTH)
#define CBLOCK_LENGTH 			64
#endif

//NELements * Length
#if !defined(THREADBLOCK_SIZE)
#define THREADBLOCK_SIZE		128
#endif
#ifndef GPU_HANDLE_ERROR
#define GPU_HANDLE_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#endif
#ifndef KERNEL_ERROR_HANDLER
#define KERNEL_ERROR_HANDLER { cudaKernelAssert( __FILE__, __LINE__);}
#endif

template<class T>
class BPCEngine
{
public:

	__host__ BPCEngine(int wLevels, int cp, float kFactor, int amountOfLUTFiles);

	void kernelLauncher(int Direction, int DSizeX, int DSizeY, T* DDataInitial, int* DDataFinal, int LUTNumberOfBitplanes, int LUTNumberOfSubbands, int LUTContextRefinement, int LUTContextSign, int LUTContextSignificance, int LUTMultPrecision, int* LUTInformation_, int* DSizeArray, cudaStream_t mainStream, double * measurementsBPC);
	void deviceMemoryAllocator(int size, int** DDataFinal, int Direction, cudaStream_t mainStream);

	__device__ void readCoefficients(T* input, int* TData, int TCoordinate, int inputXSize);
	__device__ void readCoefficients3CP(T* input, int* TData, int TCoordinate, int inputXSize);
	__device__ void writeCoefficients(int* Output, unsigned int* TData, int TCoordinate, int inputXSize);
	__device__ void initializeCoefficients(int* TData);
	__device__ void initializeCoefficients3CP(int* TData);
	__device__ void findSubband(int* CBDecompositionLevel, int* CBSubband, int TCoordinateX, int TCoordinateY, int inputXSize, int inputYSize);
	__device__ void findMSB(unsigned int* TData, int* MSB, volatile int* sharedBuffer);
	__device__ void findMSB3CP(unsigned int* TData, int* MSB, volatile int* sharedBuffer);
	__device__ void Encode(unsigned int* TData, int MSB, VOLATILE int* codeStream, int codeStreamPointer, int sharedPointer, volatile int* codeStreamShared, int CBDecompositionLevel, int CBSubband, int* sharedBuffer, int const* 	__restrict__ LUT, float kFactor);
	__device__ void Encode3CP(unsigned int* TData, int MSB, VOLATILE int* codeStream, int codeStreamPointer, int sharedPointer, volatile int* codeStreamShared, int CBDecompositionLevel, int CBSubband, int* sharedBuffer, int const* 	__restrict__ LUT);
	__device__ void Decode(unsigned int* TData, int MSB, VOLATILE int* codeStream, int codeStreamPointer, int sharedPointer, volatile int* codeStreamShared, int CBDecompositionLevel, int CBSubband, int* sharedBuffer, int const* 	__restrict__ LUT, float kFactor);
	__device__ void Decode3CP(unsigned int* TData, int MSB, VOLATILE int* codeStream, int codeStreamPointer, int sharedPointer, volatile int* codeStreamShared, int CBDecompositionLevel, int CBSubband, int* sharedBuffer, int const* 	__restrict__ LUT);
	//FIXING EXPANSION FUNCTIONS
	__device__ void expansionFix(unsigned int* TData, int* output, int WCoordinate, int inputXSize, int* sizeArray, int nWarpsBlock);
	__device__ void copyEntireCodeblock(unsigned int* TData, int* inputCodestream, int WCoordinate, int inputXSize);
	//Const Functions
	int getNumberOfCodeblocks() const;
	int getPrefixedArraySize() const;

private:

	// -------------------------------------------------------------------------
	//DEVICE FUNCTIONS ---------------------------------------------------------
	// -------------------------------------------------------------------------

	//CONTEXT SIGN FUNCTIONS		-------------------------------------------------------------------------------------------------
	__device__ int computeContext(unsigned int upperLeftCoeff, unsigned int upperCoeff, unsigned int upperRightCoeff, unsigned int leftCoeff, unsigned int rightCoeff, unsigned int bottomLeftCoeff, unsigned int bottomCoeff, unsigned int bottomRightCoeff);
	__device__ int computeContextBulk(unsigned int upperLeftCoeff, unsigned int upperCoeff, unsigned int upperRightCoeff, unsigned int leftCoeff, unsigned int rightCoeff, unsigned int bottomLeftCoeff, unsigned int bottomCoeff, unsigned int bottomRightCoeff, int currentBitPlane);
	__device__ unsigned int computeSignContext(int horizontal, int vertical);
	__device__ unsigned int computeSignContext(int upperCoeff, int leftCoeff, int rightCoeff, int bottomCoeff);
	__device__ unsigned int computeSignContextBulk(unsigned int upperCoeff, unsigned int leftCoeff, unsigned int rightCoeff, unsigned int bottomCoeff, unsigned int currentBitPlane);
	//LUT FUNCTIONS
	__device__ void initializeLUTPointers(int* LUTRefinementPointer, int* LUTSignificancePointer, int* LUTSignPointer, int CBDecompositionLevel, int CBSubband, int MSB, int s);
	__device__ void updateLUTPointers(int* LUTRefinementPointer, int* LUTSignificancePointer, int* LUTSignPointer);
	//ARITHMETIC ENCODER FUNCTIONS
	__device__ void arithmeticEncoder(unsigned int Symbol, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int probability);
	__device__ void arithmeticDecoder(unsigned int* Symbol, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int probability);
	//BORDERS FUNCTIONS
	__device__ void shareLeftBorders(unsigned int* TData, int coeffIndex, unsigned int* upAuxiliar, unsigned int* middleAuxiliar, unsigned int* bottomAuxiliar);
	__device__ void shareRightBorders(unsigned int* TData, int coeffIndex, unsigned int* upAuxiliar, unsigned int* middleAuxiliar, unsigned int* bottomAuxiliar);
	__device__ void correctCBBorders(unsigned int* TData1, unsigned int* TData4, unsigned int* TData6, unsigned int* TData3, unsigned int* TData5, unsigned int* TData8, int direction);
	//ENCODERS_DECODERS
	__device__ void SPPEncoder(unsigned int* TData, int coeffIndex, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int direction, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, unsigned int TData1, unsigned int TData2, unsigned int TData3, unsigned int TData4, unsigned int TData5, unsigned int TData6, unsigned int TData7, unsigned int TData8);
	__device__ void SPPDecoder(unsigned int* TData, int coeffIndex, int mask, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int direction, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, unsigned int TData1, unsigned int TData2, unsigned int TData3, unsigned int TData4, unsigned int TData5, unsigned int TData6, unsigned int TData7, unsigned int TData8, int bitPlane);
	__device__ void SPPEncoder3CP(unsigned int* TData, int coeffIndex, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int direction, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, unsigned int TData1, unsigned int TData2, unsigned int TData3, unsigned int TData4, unsigned int TData5, unsigned int TData6, unsigned int TData7, unsigned int TData8);
	__device__ void SPPDecoder3CP(unsigned int* TData, int coeffIndex, int mask, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int direction, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, unsigned int TData1, unsigned int TData2, unsigned int TData3, unsigned int TData4, unsigned int TData5, unsigned int TData6, unsigned int TData7, unsigned int TData8, int bitPlane);
	__device__ void CPEncoder(unsigned int* TData, int coeffIndex, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int direction, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, unsigned int TData1, unsigned int TData2, unsigned int TData3, unsigned int TData4, unsigned int TData5, unsigned int TData6, unsigned int TData7, unsigned int TData8);
	__device__ void CPDecoder(unsigned int* TData, int coeffIndex, int mask, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int direction, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, unsigned int TData1, unsigned int TData2, unsigned int TData3, unsigned int TData4, unsigned int TData5, unsigned int TData6, unsigned int TData7, unsigned int TData8, int bitPlane);
	__device__ void MRPEncoder(unsigned int* TData, int coeffIndex, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int  sharedPointer, volatile int* codeStreamShared, int probability);
	__device__ void MRPDecoder(unsigned int* TData, int coeffIndex, int mask, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int  sharedPointer, volatile int* codeStreamShared, int probability, int bitPlane);
	//ENC_DEC LAUNCHERS
	__device__ void SPPEncoderLauncher(unsigned int* TData, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT);
	__device__ void SPPDecoderLauncher(unsigned int* TData, int mask, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, int bitPlane);
	__device__ void SPPEncoderLauncher3CP(unsigned int* TData, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT);
	__device__ void SPPDecoderLauncher3CP(unsigned int* TData, int mask, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, int bitPlane);
	__device__ void CPEncoderLauncher(unsigned int* TData, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT);
	__device__ void CPDecoderLauncher(unsigned int* TData, int mask, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, int bitPlane);
	__device__ void MRPEncoderLauncher(unsigned int* TData, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int  sharedPointer, volatile int* codeStreamShared, int probability);
	__device__ void MRPDecoderLauncher(unsigned int* TData, int mask, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int  sharedPointer, volatile int* codeStreamShared, int probability, int bitPlane);
	//BULK MODE LAUNCHERS
	__device__ void encodeBulkMode(unsigned int* TData, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, int const* __restrict__ LUT, int LUTRefinementPointer);
	__device__ void decodeBulkMode(unsigned int* TData, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, int const* __restrict__ LUT, int LUTRefinementPointer, int mask);
	//BULK PROCESS
	__device__ void encodeLeftCoefficients(unsigned int* TData, int bitPlane, int i, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, int const* __restrict__ LUT, int LUTRefinementPointer);
	__device__ void encodeRightCoefficients(unsigned int* TData, int bitPlane, int i, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, int const* __restrict__ LUT, int LUTRefinementPointer);
	__device__ void encodeBulkProcessing(int bitPlane, unsigned int* TData, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, int const* __restrict__ LUT, int LUTRefinementPointer, unsigned int TData2, unsigned int TData4, unsigned int TData5, unsigned int TData7, int context, int coeffIndex);
	__device__ void decodeLeftCoefficients(unsigned int* TData, int bitPlane, int i, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, int const* __restrict__ LUT, int LUTRefinementPointer, int mask);
	__device__ void decodeRightCoefficients(unsigned int* TData, int bitPlane, int i, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, int const* __restrict__ LUT, int LUTRefinementPointer, int mask);
	__device__ void decodeBulkProcessing(int bitPlane, unsigned int* TData, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, int const* __restrict__ LUT, int LUTRefinementPointer, unsigned int TData2, unsigned int TData4, unsigned int TData5, unsigned int TData7, int context, int coeffIndex, int mask);
	//AUXILIAR FUNCTIONS
	void gpuAssert(cudaError_t, const char *, int);
	void cudaKernelAssert(const char *, int);

	//Needs to be passed to the kernel.
	int _prefixedArraySize;
	int _numberOfWaveletLevels;
	//Number of codeblocks to be processed
	int _numberOfCodeblocks;
	//Coding Passes
	int _codingPasses;
	//K Factor for multi-bitplane processing
	float _kFactor;
	//Amount of files stored in the LUT array
	int _amountOfLUTFiles;
};


//CODER KERNEL
template<class T>
__global__ void kernelBPCCoder(T*	input, int*	Output, int	inputXSize, int	inputYSize, int 	nWarpsBlock, int 	nWarpsX, int const* __restrict__	LUT, int const wLevels, int const LUTnOfBitplanes, int const LUTnOfSubbands, int const LUTCtxtRefinement, int const LUTCtxtSign, int const LUTCtxtSignificance, int const LUTMltPrecision, int *sizeArray, BPCEngine<T>* BPCE, float kFactor, int amountOfLUTFiles);
template<class T>
__global__ void kernelBPCCoder3CP(T*	input, int*	Output, int	inputXSize, int	inputYSize, int 	nWarpsBlock, int 	nWarpsX, int const* __restrict__	LUT, int const wLevels, int const LUTnOfBitplanes, int const LUTnOfSubbands, int const LUTCtxtRefinement, int const LUTCtxtSign, int const LUTCtxtSignificance, int const LUTMltPrecision, int *sizeArray, BPCEngine<T>* BPCE);
//DECODER KERNEL
template<class T>
__global__ void kernelBPCDecoder(int*	inputCodestream, int*	outputImage, int	inputXSize, int	inputYSize, int 	nWarpsBlock, int 	nWarpsX, int const* __restrict__	LUT, int const wLevels, int const LUTnOfBitplanes, int const LUTnOfSubbands, int const LUTCtxtRefinement, int const LUTCtxtSign, int const LUTCtxtSignificance, int const LUTMltPrecision, int *sizeArray, BPCEngine<T>* BPCE, float kFactor, int amountOfLUTFiles);
template<class T>
__global__ void kernelBPCDecoder3CP(int*	inputCodestream, int*	outputImage, int	inputXSize, int	inputYSize, int 	nWarpsBlock, int 	nWarpsX, int const* __restrict__	LUT, int const wLevels, int const LUTnOfBitplanes, int const LUTnOfSubbands, int const LUTCtxtRefinement, int const LUTCtxtSign, int const LUTCtxtSignificance, int const LUTMltPrecision, int *sizeArray, BPCEngine<T>* BPCE);

//QSteps for Lossy Encoding
__device__ float const L2Norm[10][4] = {
	{ 1.965908f,1.0112865f,1.0112865f,0.52021784f }, //Level 0, subbands LL, HL, LH, HH
	{ 4.1224113f,1.9968134f,1.9968134f,0.96721643f }, //Level 1, 
	{ 8.416739f,4.1833673f,4.1833673f,2.0792568f }, //Level 2, 
	{ 16.935543f,8.534108f,8.534108f,4.3004827f }, //Level 3, 
	{ 33.924816f,17.166693f,17.166693f,8.686718f }, //Level 4,
	{ 67.87687f,34.385098f,34.385098f,17.41882f }, //Level 5, 
	{ 135.76744f,68.7964f,68.7964f,34.860676f }, //Level 6,
	{ 271.5416f,137.60588f,137.60588f,69.73287f }, //Level 7,
	{ 543.0866f,275.21814f,275.21814f,139.47136f }, //Level 8,
	{ 1086.1624f,550.43286f,550.43286f,278.94202f }  //Level 9.
};


#include "BPCEngine.cu"

#endif