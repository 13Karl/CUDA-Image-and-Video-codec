
#pragma once

#include <iostream>


//To ease code complexity, global variables to this TU are declared and used through the entire algorithm.
__device__ static int _DLUTnOfBitplanes;
__device__ static int _DLUTnOfSubbands;
__device__ static int _DLUTCtxtRefinement;
__device__ static int _DLUTCtxtSign;
__device__ static int _DLUTCtxtSignificance;
__device__ static int _DLUTMltPrecision;
__device__ static int _DWaveletLevels;
__device__ static int _LUTPointerSizePerS;

template<class T>
void BPCEngine<T>::gpuAssert(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
		exit(EXIT_FAILURE);
	}
}

template<class T>
void BPCEngine<T>::cudaKernelAssert(const char *file, int line)
{
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		std::cout << "cudaKernelAssert() failed at " << file << ":" << line << ":" << cudaGetErrorString(err) << std::endl;
		exit(-1);
	}

}
// -------------------------------------------------------------------------
//DEVICE FUNCTIONS ---------------------------------------------------------
// -------------------------------------------------------------------------

template<class T>
__device__ void BPCEngine<T>::readCoefficients(T* input, int* TData, int TCoordinate, int inputXSize)
{

	int auxTCoordinate = TCoordinate;
	int negative = 0;

	for (int i = 0; i < CBLOCK_LENGTH; i++)
	{
		TData[i*NELEMENTS_THREAD_X] = (int)input[auxTCoordinate];

		negative = (TData[i*NELEMENTS_THREAD_X] < 0) ? 1 : 0;
		TData[i*NELEMENTS_THREAD_X] = (abs(TData[i*NELEMENTS_THREAD_X]) << 1) + negative;


		TData[(i*NELEMENTS_THREAD_X) + 1] = (int)input[auxTCoordinate + 1];

		negative = (TData[(i*NELEMENTS_THREAD_X) + 1] < 0) ? 1 : 0;
		TData[(i*NELEMENTS_THREAD_X) + 1] = (abs(TData[(i*NELEMENTS_THREAD_X) + 1]) << 1) + negative;

		auxTCoordinate += inputXSize;
	}

}
template<class T>
__device__ void BPCEngine<T>::readCoefficients3CP(T* input, int* TData, int TCoordinate, int inputXSize)
{

	int auxTCoordinate = TCoordinate;
	int negative = 0;

	for (int i = 0; i < CBLOCK_LENGTH; i++)
	{
		TData[i*NELEMENTS_THREAD_X] = (int)input[auxTCoordinate];

		negative = (TData[i*NELEMENTS_THREAD_X] < 0) ? 1 : 0;
		TData[i*NELEMENTS_THREAD_X] = (abs(TData[i*NELEMENTS_THREAD_X]) << 1) + negative;


		TData[(i*NELEMENTS_THREAD_X) + 1] = (int)input[auxTCoordinate + 1];

		negative = (TData[(i*NELEMENTS_THREAD_X) + 1] < 0) ? 1 : 0;
		TData[(i*NELEMENTS_THREAD_X) + 1] = (abs(TData[(i*NELEMENTS_THREAD_X) + 1]) << 1) + negative;

		//set CP flag to 1 at start
		TData[i*NELEMENTS_THREAD_X] |= (1 << 30);
		TData[(i*NELEMENTS_THREAD_X) + 1] |= (1 << 30);

		auxTCoordinate += inputXSize;
	}

}

template<class T>
__device__ void BPCEngine<T>::writeCoefficients(int* output, unsigned int* TData, int TCoordinate, int inputXSize)
{

	int auxTCoordinate = TCoordinate;

	for (int i = 0; i < CBLOCK_LENGTH; i++)
	{

		output[auxTCoordinate] = (TData[i*NELEMENTS_THREAD_X] & 0xFFFFFF) >> 1;
		if ((TData[i*NELEMENTS_THREAD_X] & 1) == 1) output[auxTCoordinate] = -output[auxTCoordinate];
		__syncwarp();
		output[auxTCoordinate + 1] = (TData[(i*NELEMENTS_THREAD_X) + 1] & 0xFFFFFF) >> 1;
		if ((TData[(i*NELEMENTS_THREAD_X) + 1] & 1) == 1) output[auxTCoordinate + 1] = -output[auxTCoordinate + 1];
		__syncwarp();
		auxTCoordinate += inputXSize;
	}

}
template<class T>
__device__ void BPCEngine<T>::initializeCoefficients(int* TData)
{

	for (int i = 0; i < CBLOCK_LENGTH; i++)
	{
		TData[i*NELEMENTS_THREAD_X] = 0;
		TData[(i*NELEMENTS_THREAD_X) + 1] = 0;
	}

}
template<class T>
__device__ void BPCEngine<T>::initializeCoefficients3CP(int* TData)
{

	for (int i = 0; i < CBLOCK_LENGTH; i++)
	{
		TData[i*NELEMENTS_THREAD_X] = 0;
		TData[(i*NELEMENTS_THREAD_X) + 1] = 0;

		//set CP flag to 1 at start
		TData[i*NELEMENTS_THREAD_X] |= (1 << 30);
		TData[(i*NELEMENTS_THREAD_X) + 1] |= (1 << 30);
	}

}

/*
Function responsible of locating in which wavelet subband is the codeblock located, and its decomposition level.
*/
template<class T>
__device__ void BPCEngine<T>::findSubband(int* CBDecompositionLevel, int* CBSubband, int TCoordinateX, int TCoordinateY, int inputXSize, int inputYSize)
{

	int inputDecompositionLevels = _DWaveletLevels;
	int aux = 1;

	while ((*CBDecompositionLevel) == -1) {

		if (aux == inputDecompositionLevels + 1) {
			*CBDecompositionLevel = inputDecompositionLevels;
			*CBSubband = 0;
		}
		else if ((TCoordinateX >= (inputXSize >> aux)) || (TCoordinateY >= (inputYSize >> aux)))
		{

			*CBDecompositionLevel = aux - 1;

			if (TCoordinateX >= (inputXSize >> aux))
				if (TCoordinateY >= (inputYSize >> aux))
					*CBSubband = 2;
				else
					*CBSubband = 0;
			else
				*CBSubband = 1;
		}
		else aux++;
	}
}

/*
Function which calculates the Most Significant Bitplane existing in a codeblock.
*/
template<class T>
__device__ void BPCEngine<T>::findMSB(unsigned int* TData, int* MSB, volatile int* sharedBuffer)
{
	int max = 0;
	for (int i = 0; i < CBLOCK_LENGTH; i++)
		max |= ((TData[(i * 2)] >> 1) | (TData[(i * 2) + 1] >> 1));

	max |= __shfl_down_sync(0xffffffff, max, 1);
	max |= __shfl_down_sync(0xffffffff, max, 2);
	max |= __shfl_down_sync(0xffffffff, max, 4);
	max |= __shfl_down_sync(0xffffffff, max, 8);
	max |= __shfl_down_sync(0xffffffff, max, 16);

	max = __shfl_sync(0xffffffff, max, 0);

	*MSB = 32 - __ffs(__brev(max));//Reverse bits and get least significant position

}

/*
Function which calculates the Most Significant Bitplane existing in a codeblock. 3 coding passes version
*/
template<class T>
__device__ void BPCEngine<T>::findMSB3CP(unsigned int* TData, int* MSB, volatile int* sharedBuffer)
{
	int max = 0;
	for (int i = 0; i < CBLOCK_LENGTH; i++)
		max |= ((TData[(i * 2)] >> 1) | (TData[(i * 2) + 1] >> 1));

	max |= __shfl_down_sync(0xffffffff, max, 1);
	max |= __shfl_down_sync(0xffffffff, max, 2);
	max |= __shfl_down_sync(0xffffffff, max, 4);
	max |= __shfl_down_sync(0xffffffff, max, 8);
	max |= __shfl_down_sync(0xffffffff, max, 16);

	max = __shfl_sync(0xffffffff, max, 0);

	//set CP flag to 0 to compute the MSB
	max &= 0xDFFFFFFF;

	*MSB = 32 - __ffs(__brev(max));//Reverse bits and get least significant position
}

/*
Function responsible of computing the context of a given bit considering all its adjacent coefficients.
*/
template<class T>
__device__ int BPCEngine<T>::computeContext(
	unsigned int upperLeftCoeff, unsigned int upperCoeff, unsigned int upperRightCoeff,
	unsigned int leftCoeff, unsigned int rightCoeff,
	unsigned int bottomLeftCoeff, unsigned int bottomCoeff, unsigned int bottomRightCoeff)
{

	return((upperRightCoeff >> 31) + (upperCoeff >> 31) + (upperLeftCoeff >> 31) + (rightCoeff >> 31) +
		(leftCoeff >> 31) + (bottomRightCoeff >> 31) + (bottomCoeff >> 31) + (bottomLeftCoeff >> 31));
}

/*
Function responsible of computing the context of a given bit considering all its adjacent coefficients. CS Massive version.
*/
template<class T>
__device__ int BPCEngine<T>::computeContextBulk(
	unsigned int upperLeftCoeff, unsigned int upperCoeff, unsigned int upperRightCoeff,
	unsigned int leftCoeff, unsigned int rightCoeff,
	unsigned int bottomLeftCoeff, unsigned int bottomCoeff, unsigned int bottomRightCoeff, int currentBitPlane)
{
	return((((upperRightCoeff >> 24) & 31) >= currentBitPlane) + (((upperCoeff >> 24) & 31) >= currentBitPlane) + (((upperLeftCoeff >> 24) & 31) >= currentBitPlane) + (((rightCoeff >> 24) & 31) >= currentBitPlane) +
		(((leftCoeff >> 24) & 31) >= currentBitPlane) + (((bottomRightCoeff >> 24) & 31) >= currentBitPlane) + (((bottomCoeff >> 24) & 31) >= currentBitPlane) + (((bottomLeftCoeff >> 24) & 31) >= currentBitPlane));
}


//CONTEXT SIGN FUNCTIONS		-------------------------------------------------------------------------------------------------
/*
Functions responsible of computing the context of the sign of a given bit considering all its adjacent coefficients.
*/

template<class T>
__device__ unsigned int BPCEngine<T>::computeSignContext(int horizontal, int vertical)
{
	int context;

	//Sign contexts are from -4..-1,0,1..4, but they are represented with the sign as first least significant bit

	if (horizontal == 0) {
		if (vertical == 0) {
			context = 0;
		}
		else if (vertical > 0) {
			context = 2;
		}
		else {
			context = 3;
		}
	}
	else if (horizontal > 0) {
		if (vertical == 0) {
			context = 4;
		}
		else if (vertical > 0) {
			context = 6;
		}
		else {
			context = 0;
		}
	}
	else {
		if (vertical == 0) {
			context = 5;
		}
		else if (vertical > 0) {
			context = 1;
		}
		else {
			context = 7;
		}
	}

	return(context);
}

template<class T>
__device__ unsigned int BPCEngine<T>::computeSignContext(
	int upperCoeff,
	int leftCoeff, int rightCoeff,
	int bottomCoeff)
{

	int left = (leftCoeff >> 31) == 0 ? 0 : ((leftCoeff & 1) == 1 ? -1 : 1);
	int right = (rightCoeff >> 31) == 0 ? 0 : ((rightCoeff & 1) == 1 ? -1 : 1);
	int upper = (upperCoeff >> 31) == 0 ? 0 : ((upperCoeff & 1) == 1 ? -1 : 1);
	int bottom = (bottomCoeff >> 31) == 0 ? 0 : ((bottomCoeff & 1) == 1 ? -1 : 1);

	return(computeSignContext(left + right, upper + bottom));
}

template<class T>
__device__ unsigned int BPCEngine<T>::computeSignContextBulk(
	unsigned int upperCoeff,
	unsigned int leftCoeff, unsigned int rightCoeff,
	unsigned int bottomCoeff, unsigned int currentBitPlane)
{
	
	int left = (leftCoeff >> 31) == 0 ? 0 : (((leftCoeff >> 24) & 31) >= currentBitPlane) == 0 ? 0 : ((leftCoeff & 1) == 1 ? -1 : 1);
	int right = (rightCoeff >> 31) == 0 ? 0 : (((rightCoeff >> 24) & 31) >= currentBitPlane) == 0 ? 0 : ((rightCoeff & 1) == 1 ? -1 : 1);
	int upper = (upperCoeff >> 31) == 0 ? 0 : (((upperCoeff >> 24) & 31) >= currentBitPlane) == 0 ? 0 : ((upperCoeff & 1) == 1 ? -1 : 1);
	int bottom = (bottomCoeff >> 31) == 0 ? 0 : (((bottomCoeff >> 24) & 31) >= currentBitPlane) == 0 ? 0 : ((bottomCoeff & 1) == 1 ? -1 : 1);

	return(computeSignContext(left + right, upper + bottom));
}

/*
Function to initialize the LUT values used in the context and probability models. These LUT values are loaded from external files at the launch of the codec.
*/
template<class T>
__device__ void BPCEngine<T>::initializeLUTPointers(int* LUTRefinementPointer, int* LUTSignificancePointer, int* LUTSignPointer, int CBDecompositionLevel, int CBSubband, int MSB, int s)
{

	int LUTPointerAux = 0;
	int LUTOffset = s * _LUTPointerSizePerS;

	*LUTRefinementPointer = (CBDecompositionLevel * _DLUTnOfSubbands * _DLUTnOfBitplanes * _DLUTCtxtRefinement) +
		(CBSubband * _DLUTnOfBitplanes *  _DLUTCtxtRefinement) +
		(MSB*_DLUTCtxtRefinement) + LUTOffset;

	LUTPointerAux = ((_DLUTnOfSubbands) * (_DLUTnOfBitplanes) * (_DLUTCtxtRefinement) * (_DWaveletLevels)) + (_DLUTnOfBitplanes*_DLUTCtxtRefinement);

	*LUTSignificancePointer = (CBDecompositionLevel * _DLUTnOfSubbands * _DLUTnOfBitplanes*  _DLUTCtxtSignificance) +
		(CBSubband * _DLUTnOfBitplanes *  _DLUTCtxtSignificance) +
		(MSB*_DLUTCtxtSignificance) + LUTPointerAux + LUTOffset;

	LUTPointerAux += ((_DLUTnOfSubbands) * (_DLUTnOfBitplanes) * (_DLUTCtxtSignificance) * (_DWaveletLevels)) + (_DLUTnOfBitplanes*_DLUTCtxtSignificance);

	*LUTSignPointer = (CBDecompositionLevel * _DLUTnOfSubbands* _DLUTnOfBitplanes *  _DLUTCtxtSign) +
		(CBSubband * _DLUTnOfBitplanes *  _DLUTCtxtSign) +
		(MSB*_DLUTCtxtSign) + LUTPointerAux + LUTOffset;
}

template<class T>
__device__ void BPCEngine<T>::updateLUTPointers(int* LUTRefinementPointer, int* LUTSignificancePointer, int* LUTSignPointer)
{
	*LUTSignificancePointer -= _DLUTCtxtSignificance;
	*LUTSignPointer -= _DLUTCtxtSign;
	*LUTRefinementPointer -= _DLUTCtxtRefinement;
}

/*
Function in charge of processing bit by bit considering the probabilities of each context to be 0 or 1. The arithmetic encoder is responsible of transforming the wavelet coefficients into codewords, forming bitstreams.
Each cuda thread is responsible of encoding the bits of the bitplane which is being encoded.

codeStream: output stream in main memory
codeStreamPointer: Warp pointer in the main memory to where the codeblock the warp is coding starts in memory.
reservedCodeword: Location in main memory where the codeword is going to be written.
CodeStreamShared: Stores the next free space for each warp of a specific threadBlock in main memory.
sharedPointer: Pointer of the code_stream_shared to the position that each warp has reserved in the threadBlock. Integer which stores the warpID inside a threadblock (from 0 to 3).
*/
template<class T>
__device__ void BPCEngine<T>::arithmeticEncoder(unsigned int Symbol, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int probability)
{

	unsigned int aux = 0;

	if ((*ACIntervalSize) == 0) {
		//Function to know which threads are in this position. Not all the threads are supposed to be here at a given moment.
		aux = __activemask();
		*ACIntervalLower = 0;
		*ACIntervalSize = (1 << CODEWORD_SIZE) - 1;
		//Reserving the codeword in main memory.
		*reservedCodeword = min(__popc(aux << (WARPSIZE - (threadIdx.x & 0x1f))) + codeStreamShared[sharedPointer], 4094) + codeStreamPointer;
		if ((aux >> (threadIdx.x & 0x1f)) == 1)	codeStreamShared[sharedPointer] = min(codeStreamShared[sharedPointer] + __popc(aux), 4095);
	}
	//Probability is calculated by taking into account that LUT values are 7 bit precision (MULT_PRECISION is 7).
	aux = (((*ACIntervalSize)*probability) >> (_DLUTMltPrecision)) + Symbol;

	if ((Symbol == 0))		*ACIntervalSize = aux;
	else {

		*ACIntervalSize -= aux;
		*ACIntervalLower += aux;
	}
		
	if ((*ACIntervalSize) == 0) 
	{
		codeStream[*reservedCodeword] = *ACIntervalLower;
	}
}

/*
Decoding version of the arithmetic coder. Making use of probabilities retrieved from the LUT tables, decodes codewords bit by bit.
*/
template<class T>
__device__ void BPCEngine<T>::arithmeticDecoder(unsigned int* Symbol, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int probability)
{
	unsigned int aux = 0;
	unsigned int aux2 = 0;


	if ((*ACIntervalSize) == 0) {

		aux = __activemask();
		*ACIntervalLower = 0;
		*ACIntervalSize = (1 << CODEWORD_SIZE) - 1;

		aux2 = min(__popc(aux << (WARPSIZE - (threadIdx.x & 0x1f))) + codeStreamShared[sharedPointer], 4094) + codeStreamPointer;

		*codeword = codeStream[aux2];

		if ((aux >> (threadIdx.x & 0x1f)) == 1) codeStreamShared[sharedPointer] = min(codeStreamShared[sharedPointer] + __popc(aux), 4095);
	}

	aux = (((*ACIntervalSize)*probability) >> (_DLUTMltPrecision)) + 1;
	aux2 = *ACIntervalLower + aux;

	if (*codeword >= aux2)
	{
		*ACIntervalSize -= aux;
		*ACIntervalLower = aux2;

		*Symbol = 1;
	}
	else {

		*ACIntervalSize = aux - 1;

		*Symbol = 0;

	}

}

//Via shuffling instructions, the information is shared between adjacent threads for context formation purposes.
template<class T>
__device__ void BPCEngine<T>::shareLeftBorders(unsigned int* TData, int coeffIndex, unsigned int* upAuxiliar, unsigned int* middleAuxiliar, unsigned int* bottomAuxiliar) {

	*upAuxiliar = __shfl_up_sync(0xffffffff, TData[coeffIndex - 1], 1);
	*middleAuxiliar = __shfl_up_sync(0xffffffff, TData[coeffIndex + 1], 1);
	*bottomAuxiliar = __shfl_up_sync(0xffffffff, TData[coeffIndex + 3], 1);

}

//Via shuffling instructions, the information is shared between adjacent threads for context formation purposes.
template<class T>
__device__ void BPCEngine<T>::shareRightBorders(unsigned int* TData, int coeffIndex, unsigned int* upAuxiliar, unsigned int* middleAuxiliar, unsigned int* bottomAuxiliar) {

	*upAuxiliar = __shfl_down_sync(0xffffffff, TData[coeffIndex - 3], 1);
	*middleAuxiliar = __shfl_down_sync(0xffffffff, TData[coeffIndex - 1], 1);
	*bottomAuxiliar = __shfl_down_sync(0xffffffff, TData[coeffIndex + 1], 1);
}

//Function which fixes the values for threads in the borders of the frames (those will not have adjacent threads to use when coding contexts).
template<class T>
__device__ void BPCEngine<T>::correctCBBorders(unsigned int* TData1, unsigned int* TData4, unsigned int* TData6, unsigned int* TData3, unsigned int* TData5, unsigned int* TData8, int direction)
{

	if (direction == 1) {
		if ((threadIdx.x % 32) == 0)
		{
			*TData1 = 0;
			*TData4 = 0;
			*TData6 = 0;
		}
	}
	else {
		if ((threadIdx.x % 32) == 31)
		{
			*TData3 = 0;
			*TData5 = 0;
			*TData8 = 0;
		}
	}
}

/*
Significance Propagation Pass - Encoding version. Manages the context calculation and when the sign of a given coefficient must be computed (typically, when the coefficient becomes significant for the first time).
*/
template<class T>
__device__ void BPCEngine<T>::SPPEncoder(unsigned int* TData, int coeffIndex, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int direction, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, unsigned int TData1, unsigned int TData2, unsigned int TData3, unsigned int TData4, unsigned int TData5, unsigned int TData6, unsigned int TData7, unsigned int TData8)
{

	unsigned int context = 0;
	// The significance status is stored in the leftmost bit
	if (!(TData[coeffIndex] >> 31))
	{

		correctCBBorders(&TData1, &TData4, &TData6, &TData3, &TData5, &TData8, direction);

		context = computeContext(TData1, TData2, TData3, TData4, TData5, TData6, TData7, TData8);

		arithmeticEncoder((TData[coeffIndex] >> (bitPlane + 1)) & 1, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTSignificancePointer + context]);

		if (((TData[coeffIndex] >> (bitPlane + 1)) & 1) == 1)
		{

			TData[coeffIndex] |= (1 << 31);

			TData[coeffIndex] |= (bitPlane << 24);

			context = computeSignContext(TData2, TData4, TData5, TData7);

			arithmeticEncoder(((TData[coeffIndex] & 1) == (context & 1)) ? 0 : 1, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTSignPointer + (context >> 1)]);
		}
	}
}

/*
Significance Propagation Pass 3 Coding passes version - Encoding. Manages the context calculation and when the sign of a given coefficient must be computed (typically, when the coefficient becomes significant for the first time).
*/
template<class T>
__device__ void BPCEngine<T>::SPPEncoder3CP(unsigned int* TData, int coeffIndex, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int direction, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, unsigned int TData1, unsigned int TData2, unsigned int TData3, unsigned int TData4, unsigned int TData5, unsigned int TData6, unsigned int TData7, unsigned int TData8)
{

	unsigned int context = 0;
	// The significance status is stored in the leftmost bit
	if (!(TData[coeffIndex] >> 31))
	{

		correctCBBorders(&TData1, &TData4, &TData6, &TData3, &TData5, &TData8, direction);

		if ((TData1 >> 31) || (TData2 >> 31) || (TData3 >> 31) || (TData4 >> 31) || (TData5 >> 31) || (TData6 >> 31) || (TData7 >> 31) || (TData8 >> 31))
		{

			context = computeContext(TData1, TData2, TData3, TData4, TData5, TData6, TData7, TData8);

			arithmeticEncoder((TData[coeffIndex] >> (bitPlane + 1)) & 1, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTSignificancePointer + context]);

			if (((TData[coeffIndex] >> (bitPlane + 1)) & 1) == 1)
			{

				TData[coeffIndex] |= (1 << 31);

				TData[coeffIndex] |= (bitPlane << 24);

				context = computeSignContext(TData2, TData4, TData5, TData7);

				arithmeticEncoder(((TData[coeffIndex] & 1) == (context & 1)) ? 0 : 1, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTSignPointer + (context >> 1)]);
			}
		}
		else TData[coeffIndex] |= (1 << 30);
	}
}

/*
Significance Propagation Pass - Decoding version. Manages the context calculation and when the sign of a given coefficient must be computed (typically, when the coefficient becomes significant for the first time).
*/
template<class T>
__device__ void BPCEngine<T>::SPPDecoder(unsigned int* TData, int coeffIndex, int mask, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int direction, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, unsigned int TData1, unsigned int TData2, unsigned int TData3, unsigned int TData4, unsigned int TData5, unsigned int TData6, unsigned int TData7, unsigned int TData8, int bitPlane)
{

	unsigned int context = 0;
	unsigned int symbol = 0;

	if (!(TData[coeffIndex] >> 31))
	{

		correctCBBorders(&TData1, &TData4, &TData6, &TData3, &TData5, &TData8, direction);

		context = computeContext(TData1, TData2, TData3, TData4, TData5, TData6, TData7, TData8);

		arithmeticDecoder(&symbol, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTSignificancePointer + context]);

		if (symbol == 1)
		{

			TData[coeffIndex] |= mask;

			TData[coeffIndex] |= (1 << 31);

			TData[coeffIndex] |= (bitPlane << 24);

			context = computeSignContext(TData2, TData4, TData5, TData7);

			arithmeticDecoder(&symbol, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTSignPointer + (context >> 1)]);

			symbol = ((symbol & 1) == (context & 1)) ? 0 : 1;

			TData[coeffIndex] |= (symbol & 1);

		}

	}
}

/*
Significance Propagation Pass 3 Coding passes version - Decoding. Manages the context calculation and when the sign of a given coefficient must be computed (typically, when the coefficient becomes significant for the first time).
*/
template<class T>
__device__ void BPCEngine<T>::SPPDecoder3CP(unsigned int* TData, int coeffIndex, int mask, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int direction, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, unsigned int TData1, unsigned int TData2, unsigned int TData3, unsigned int TData4, unsigned int TData5, unsigned int TData6, unsigned int TData7, unsigned int TData8, int bitPlane)
{

	unsigned int context = 0;
	unsigned int symbol = 0;

	if (!(TData[coeffIndex] >> 31))
	{

		correctCBBorders(&TData1, &TData4, &TData6, &TData3, &TData5, &TData8, direction);

		if ((TData1 >> 31) || (TData2 >> 31) || (TData3 >> 31) || (TData4 >> 31) || (TData5 >> 31) || (TData6 >> 31) || (TData7 >> 31) || (TData8 >> 31))
		{

			context = computeContext(TData1, TData2, TData3, TData4, TData5, TData6, TData7, TData8);

			arithmeticDecoder(&symbol, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTSignificancePointer + context]);

			if (symbol == 1)
			{

				TData[coeffIndex] |= mask;

				TData[coeffIndex] |= (1 << 31);

				TData[coeffIndex] |= (bitPlane << 24);

				context = computeSignContext(TData2, TData4, TData5, TData7);

				arithmeticDecoder(&symbol, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTSignPointer + (context >> 1)]);

				symbol = ((symbol & 1) == (context & 1)) ? 0 : 1;

				TData[coeffIndex] |= (symbol & 1);

			}
		}
		else TData[coeffIndex] |= (1 << 30);
	}
}

/*
Cleanup Pass - Encoding version. Manages the context calculation and when the sign of a given coefficient must be computed (typically, when the coefficient becomes significant for the first time).
*/
template<class T>
__device__ void BPCEngine<T>::CPEncoder(unsigned int* TData, int coeffIndex, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int direction, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, unsigned int TData1, unsigned int TData2, unsigned int TData3, unsigned int TData4, unsigned int TData5, unsigned int TData6, unsigned int TData7, unsigned int TData8)
{

	unsigned int context = 0;

	if ((TData[coeffIndex] >> 30) & 1)
	{

		correctCBBorders(&TData1, &TData4, &TData6, &TData3, &TData5, &TData8, direction);

		context = computeContext(TData1, TData2, TData3, TData4, TData5, TData6, TData7, TData8);


		arithmeticEncoder((TData[coeffIndex] >> (bitPlane + 1)) & 1, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTSignificancePointer + context]);

		TData[coeffIndex] &= 0xBFFFFFFF;

		if (((TData[coeffIndex] >> (bitPlane + 1)) & 1) == 1)
		{

			TData[coeffIndex] |= (1 << 31);
			TData[coeffIndex] |= (1 << 29);

			TData[coeffIndex] |= (bitPlane << 24);

			context = computeSignContext(TData2, TData4, TData5, TData7);

			arithmeticEncoder(((TData[coeffIndex] & 1) == (context & 1)) ? 0 : 1, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTSignPointer + (context >> 1)]);

		}
	}
}

/*
Cleanup Pass - Decoding version. Manages the context calculation and when the sign of a given coefficient must be computed (typically, when the coefficient becomes significant for the first time).
*/
template<class T>
__device__ void BPCEngine<T>::CPDecoder(unsigned int* TData, int coeffIndex, int mask, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int direction, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, unsigned int TData1, unsigned int TData2, unsigned int TData3, unsigned int TData4, unsigned int TData5, unsigned int TData6, unsigned int TData7, unsigned int TData8, int bitPlane)
{

	unsigned int context = 0;
	unsigned int symbol = 0;

	if ((TData[coeffIndex] >> 30) & 1)
	{

		correctCBBorders(&TData1, &TData4, &TData6, &TData3, &TData5, &TData8, direction);

		context = computeContext(TData1, TData2, TData3, TData4, TData5, TData6, TData7, TData8);

		arithmeticDecoder(&symbol, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTSignificancePointer + context]);

		TData[coeffIndex] &= 0xBFFFFFFF;

		if (symbol == 1)
		{

			TData[coeffIndex] |= mask;

			TData[coeffIndex] |= (1 << 31);
			TData[coeffIndex] |= (1 << 29);

			TData[coeffIndex] |= (bitPlane << 24);

			context = computeSignContext(TData2, TData4, TData5, TData7);

			arithmeticDecoder(&symbol, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTSignPointer + (context >> 1)]);

			symbol = ((symbol & 1) == (context & 1)) ? 0 : 1;

			TData[coeffIndex] |= (symbol & 1);

		}
	}
}

/*
Magnitude Refinement Pass - Encoding version. If the coefficient being scanned has been significant in previous passes, it is scanned and processed now,
making use of the arithmetic encoder and using precomputed probabilities in the LUT table.
*/
template<class T>
__device__ void BPCEngine<T>::MRPEncoder(unsigned int* TData, int coeffIndex, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int  sharedPointer, volatile int* codeStreamShared, int probability)
{

	if ((TData[coeffIndex] >> 29) & 1)
	{

		arithmeticEncoder((TData[coeffIndex] >> (bitPlane + 1)) & 1, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, probability);

	}
	else if (TData[coeffIndex] >> 31) TData[coeffIndex] |= (1 << 29);
}

/*
Magnitude Refinement Pass - Decoding version. If the coefficient being scanned has been significant in previous passes, it is scanned and processed now,
making use of the arithmetic encoder and using precomputed probabilities in the LUT table.
*/
template<class T>
__device__ void BPCEngine<T>::MRPDecoder(unsigned int* TData, int coeffIndex, int mask, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int  sharedPointer, volatile int* codeStreamShared, int probability, int bitPlane)
{

	unsigned int symbol = 0;

	if ((TData[coeffIndex] >> 29) & 1)
	{


		arithmeticDecoder(&symbol, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, probability);

		//Delete previous aproximate bit
		TData[coeffIndex] &= (~mask);
		//Write new bit and aproximation
		TData[coeffIndex] |= (mask&(((symbol << 1) + 1) << (bitPlane)));


	}
	else if (TData[coeffIndex] >> 31) TData[coeffIndex] |= (1 << 29);
}


/*
Significance Propagation Pass encoder management function. In this function, the bitplane is scanned from the first to the last bit, calling the SPPEncoder for each coefficient. In between calls,
the threads communicate among them to share information to help forming contexts.
*/
template<class T>
__device__ void BPCEngine<T>::SPPEncoderLauncher(unsigned int* TData, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT)
{

	int i = 0;

	unsigned int upAuxiliar = 0;
	unsigned int middleAuxiliar = 0;
	unsigned int bottomAuxiliar = 0;

	int coeffIndex = i * 2;

	shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	SPPEncoder(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		0, 0, 0,
		middleAuxiliar, TData[coeffIndex + 1],
		bottomAuxiliar, TData[coeffIndex + 2], TData[coeffIndex + 3]);

	coeffIndex = (i * 2) + 1;

	shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	SPPEncoder(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		0, 0, 0,
		TData[coeffIndex - 1], middleAuxiliar,
		TData[coeffIndex + 1], TData[coeffIndex + 2], bottomAuxiliar);

	for (i = 1; i < (CBLOCK_LENGTH - 1); i++)
	{

		coeffIndex = i * 2;

		shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

		SPPEncoder(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
			ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
			upAuxiliar, TData[coeffIndex - 2], TData[coeffIndex - 1],
			middleAuxiliar, TData[coeffIndex + 1],
			bottomAuxiliar, TData[coeffIndex + 2], TData[coeffIndex + 3]);

		coeffIndex = (i * 2) + 1;

		shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

		SPPEncoder(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
			ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
			TData[coeffIndex - 3], TData[coeffIndex - 2], upAuxiliar,
			TData[coeffIndex - 1], middleAuxiliar,
			TData[coeffIndex + 1], TData[coeffIndex + 2], bottomAuxiliar);
	}

	coeffIndex = i * 2;

	shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	SPPEncoder(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		upAuxiliar, TData[coeffIndex - 2], TData[coeffIndex - 1],
		middleAuxiliar, TData[coeffIndex + 1],
		0, 0, 0);

	coeffIndex = (i * 2) + 1;

	shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	SPPEncoder(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		TData[coeffIndex - 3], TData[coeffIndex - 2], upAuxiliar,
		TData[coeffIndex - 1], middleAuxiliar,
		0, 0, 0);

}

/*
Significance Propagation Pass - 3 coding passes encoder version management function. In this function, the bitplane is scanned from the first to the last bit, calling the SPPEncoder3CP for each coefficient. In between calls,
the threads communicate among them to share information to help forming contexts.
*/
template<class T>
__device__ void BPCEngine<T>::SPPEncoderLauncher3CP(unsigned int* TData, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT)
{

	int i = 0;

	unsigned int upAuxiliar = 0;
	unsigned int middleAuxiliar = 0;
	unsigned int bottomAuxiliar = 0;

	int coeffIndex = i * 2;

	shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	SPPEncoder3CP(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		0, 0, 0,
		middleAuxiliar, TData[coeffIndex + 1],
		bottomAuxiliar, TData[coeffIndex + 2], TData[coeffIndex + 3]);

	coeffIndex = (i * 2) + 1;

	shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	SPPEncoder3CP(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		0, 0, 0,
		TData[coeffIndex - 1], middleAuxiliar,
		TData[coeffIndex + 1], TData[coeffIndex + 2], bottomAuxiliar);

	for (i = 1; i < (CBLOCK_LENGTH - 1); i++)
	{

		coeffIndex = i * 2;

		shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

		SPPEncoder3CP(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
			ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
			upAuxiliar, TData[coeffIndex - 2], TData[coeffIndex - 1],
			middleAuxiliar, TData[coeffIndex + 1],
			bottomAuxiliar, TData[coeffIndex + 2], TData[coeffIndex + 3]);

		coeffIndex = (i * 2) + 1;

		shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

		SPPEncoder3CP(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
			ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
			TData[coeffIndex - 3], TData[coeffIndex - 2], upAuxiliar,
			TData[coeffIndex - 1], middleAuxiliar,
			TData[coeffIndex + 1], TData[coeffIndex + 2], bottomAuxiliar);
	}

	coeffIndex = i * 2;

	shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	SPPEncoder3CP(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		upAuxiliar, TData[coeffIndex - 2], TData[coeffIndex - 1],
		middleAuxiliar, TData[coeffIndex + 1],
		0, 0, 0);

	coeffIndex = (i * 2) + 1;

	shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	SPPEncoder3CP(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		TData[coeffIndex - 3], TData[coeffIndex - 2], upAuxiliar,
		TData[coeffIndex - 1], middleAuxiliar,
		0, 0, 0);

}

/*
Significance Propagation Pass decoder management function. In this function, the bitplane is scanned from the first to the last bit, calling the SPPDecoder for each coefficient. In between calls,
the threads communicate among them to share information to help forming contexts.
*/
template<class T>
__device__ void BPCEngine<T>::SPPDecoderLauncher(unsigned int* TData, int mask, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, int bitPlane)
{

	int i = 0;

	unsigned int upAuxiliar = 0;
	unsigned int middleAuxiliar = 0;
	unsigned int bottomAuxiliar = 0;

	int coeffIndex = i * 2;

	shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	SPPDecoder(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		0, 0, 0,
		middleAuxiliar, TData[coeffIndex + 1],
		bottomAuxiliar, TData[coeffIndex + 2], TData[coeffIndex + 3], bitPlane);

	coeffIndex = (i * 2) + 1;

	shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	SPPDecoder(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		0, 0, 0,
		TData[coeffIndex - 1], middleAuxiliar,
		TData[coeffIndex + 1], TData[coeffIndex + 2], bottomAuxiliar, bitPlane);

	for (i = 1; i < (CBLOCK_LENGTH - 1); i++)
	{

		coeffIndex = i * 2;

		shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

		SPPDecoder(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
			ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
			upAuxiliar, TData[coeffIndex - 2], TData[coeffIndex - 1],
			middleAuxiliar, TData[coeffIndex + 1],
			bottomAuxiliar, TData[coeffIndex + 2], TData[coeffIndex + 3], bitPlane);

		coeffIndex = (i * 2) + 1;

		shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

		SPPDecoder(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
			ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
			TData[coeffIndex - 3], TData[coeffIndex - 2], upAuxiliar,
			TData[coeffIndex - 1], middleAuxiliar,
			TData[coeffIndex + 1], TData[coeffIndex + 2], bottomAuxiliar, bitPlane);
	}

	coeffIndex = i * 2;

	shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	SPPDecoder(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		upAuxiliar, TData[coeffIndex - 2], TData[coeffIndex - 1],
		middleAuxiliar, TData[coeffIndex + 1],
		0, 0, 0, bitPlane);

	coeffIndex = (i * 2) + 1;

	shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	SPPDecoder(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		TData[coeffIndex - 3], TData[coeffIndex - 2], upAuxiliar,
		TData[coeffIndex - 1], middleAuxiliar,
		0, 0, 0, bitPlane);

}

/*
Significance Propagation Pass decoder management function. In this function, the bitplane is scanned from the first to the last bit, calling the SPPDecoder3CP for each coefficient. In between calls,
the threads communicate among them to share information to help forming contexts.
*/
template<class T>
__device__ void BPCEngine<T>::SPPDecoderLauncher3CP(unsigned int* TData, int mask, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, int bitPlane)
{

	int i = 0;

	unsigned int upAuxiliar = 0;
	unsigned int middleAuxiliar = 0;
	unsigned int bottomAuxiliar = 0;

	int coeffIndex = i * 2;

	shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	SPPDecoder3CP(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		0, 0, 0,
		middleAuxiliar, TData[coeffIndex + 1],
		bottomAuxiliar, TData[coeffIndex + 2], TData[coeffIndex + 3], bitPlane);

	coeffIndex = (i * 2) + 1;

	shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	SPPDecoder3CP(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		0, 0, 0,
		TData[coeffIndex - 1], middleAuxiliar,
		TData[coeffIndex + 1], TData[coeffIndex + 2], bottomAuxiliar, bitPlane);

	for (i = 1; i < (CBLOCK_LENGTH - 1); i++)
	{

		coeffIndex = i * 2;

		shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

		SPPDecoder3CP(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
			ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
			upAuxiliar, TData[coeffIndex - 2], TData[coeffIndex - 1],
			middleAuxiliar, TData[coeffIndex + 1],
			bottomAuxiliar, TData[coeffIndex + 2], TData[coeffIndex + 3], bitPlane);

		coeffIndex = (i * 2) + 1;

		shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

		SPPDecoder3CP(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
			ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
			TData[coeffIndex - 3], TData[coeffIndex - 2], upAuxiliar,
			TData[coeffIndex - 1], middleAuxiliar,
			TData[coeffIndex + 1], TData[coeffIndex + 2], bottomAuxiliar, bitPlane);
	}

	coeffIndex = i * 2;

	shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	SPPDecoder3CP(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		upAuxiliar, TData[coeffIndex - 2], TData[coeffIndex - 1],
		middleAuxiliar, TData[coeffIndex + 1],
		0, 0, 0, bitPlane);

	coeffIndex = (i * 2) + 1;

	shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	SPPDecoder3CP(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		TData[coeffIndex - 3], TData[coeffIndex - 2], upAuxiliar,
		TData[coeffIndex - 1], middleAuxiliar,
		0, 0, 0, bitPlane);

}

/*
Cleanup Pass encoder management function. In this function, the bitplane is scanned from the first to the last bit, calling the CPEncoder for each coefficient. In between calls,
the threads communicate among them to share information to help forming contexts. This function is only used when employing 3 coding passes.
*/
template<class T>
__device__ void BPCEngine<T>::CPEncoderLauncher(unsigned int* TData, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT)
{

	int i = 0;

	unsigned int upAuxiliar = 0;
	unsigned int middleAuxiliar = 0;
	unsigned int bottomAuxiliar = 0;

	int coeffIndex = i * 2;

	shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	CPEncoder(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		0, 0, 0,
		middleAuxiliar, TData[coeffIndex + 1],
		bottomAuxiliar, TData[coeffIndex + 2], TData[coeffIndex + 3]);

	coeffIndex = (i * 2) + 1;

	shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	CPEncoder(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		0, 0, 0,
		TData[coeffIndex - 1], middleAuxiliar,
		TData[coeffIndex + 1], TData[coeffIndex + 2], bottomAuxiliar);

	for (i = 1; i < (CBLOCK_LENGTH - 1); i++)
	{

		coeffIndex = i * 2;

		shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

		CPEncoder(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
			ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
			upAuxiliar, TData[coeffIndex - 2], TData[coeffIndex - 1],
			middleAuxiliar, TData[coeffIndex + 1],
			bottomAuxiliar, TData[coeffIndex + 2], TData[coeffIndex + 3]);

		coeffIndex = (i * 2) + 1;

		shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

		CPEncoder(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
			ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
			TData[coeffIndex - 3], TData[coeffIndex - 2], upAuxiliar,
			TData[coeffIndex - 1], middleAuxiliar,
			TData[coeffIndex + 1], TData[coeffIndex + 2], bottomAuxiliar);
	}

	coeffIndex = i * 2;

	shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	CPEncoder(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		upAuxiliar, TData[coeffIndex - 2], TData[coeffIndex - 1],
		middleAuxiliar, TData[coeffIndex + 1],
		0, 0, 0);

	coeffIndex = (i * 2) + 1;

	shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	CPEncoder(TData, coeffIndex, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		TData[coeffIndex - 3], TData[coeffIndex - 2], upAuxiliar,
		TData[coeffIndex - 1], middleAuxiliar,
		0, 0, 0);

}

/*
Cleanup Pass decoder management function. In this function, the bitplane is scanned from the first to the last bit, calling the CPDecoder for each coefficient. In between calls,
the threads communicate among them to share information to help forming contexts. This function is only used when employing 3 coding passes.
*/
template<class T>
__device__ void BPCEngine<T>::CPDecoderLauncher(unsigned int* TData, int mask, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, volatile int* sharedBuffer, int const* __restrict__ LUT, int bitPlane)
{

	int i = 0;

	unsigned int upAuxiliar = 0;
	unsigned int middleAuxiliar = 0;
	unsigned int bottomAuxiliar = 0;

	int coeffIndex = i * 2;

	shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	CPDecoder(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		0, 0, 0,
		middleAuxiliar, TData[coeffIndex + 1],
		bottomAuxiliar, TData[coeffIndex + 2], TData[coeffIndex + 3], bitPlane);

	coeffIndex = (i * 2) + 1;

	shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	CPDecoder(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		0, 0, 0,
		TData[coeffIndex - 1], middleAuxiliar,
		TData[coeffIndex + 1], TData[coeffIndex + 2], bottomAuxiliar, bitPlane);

	for (i = 1; i < (CBLOCK_LENGTH - 1); i++)
	{

		coeffIndex = i * 2;

		shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

		CPDecoder(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
			ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
			upAuxiliar, TData[coeffIndex - 2], TData[coeffIndex - 1],
			middleAuxiliar, TData[coeffIndex + 1],
			bottomAuxiliar, TData[coeffIndex + 2], TData[coeffIndex + 3], bitPlane);

		coeffIndex = (i * 2) + 1;

		shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

		CPDecoder(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
			ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
			TData[coeffIndex - 3], TData[coeffIndex - 2], upAuxiliar,
			TData[coeffIndex - 1], middleAuxiliar,
			TData[coeffIndex + 1], TData[coeffIndex + 2], bottomAuxiliar, bitPlane);
	}

	coeffIndex = i * 2;

	shareLeftBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	CPDecoder(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 1, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		upAuxiliar, TData[coeffIndex - 2], TData[coeffIndex - 1],
		middleAuxiliar, TData[coeffIndex + 1],
		0, 0, 0, bitPlane);

	coeffIndex = (i * 2) + 1;

	shareRightBorders(TData, coeffIndex, &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);

	CPDecoder(TData, coeffIndex, mask, codeStream, codeStreamPointer, codeword,
		ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, 2, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT,
		TData[coeffIndex - 3], TData[coeffIndex - 2], upAuxiliar,
		TData[coeffIndex - 1], middleAuxiliar,
		0, 0, 0, bitPlane);

}

/*
Magnitude Refinement Pass encoder management function. In this function, the bitplane is scanned from the first to the last bit, calling the MRPEncoder for each coefficient.
*/
template<class T>
__device__ void BPCEngine<T>::MRPEncoderLauncher(unsigned int* TData, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int  sharedPointer, volatile int* codeStreamShared, int probability) {

	for (int i = 0; i < CBLOCK_LENGTH; i++)
	{
		__syncwarp();
		MRPEncoder(TData, i * 2, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
			ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, probability);
		__syncwarp();
		MRPEncoder(TData, (i * 2) + 1, bitPlane, codeStream, codeStreamPointer, reservedCodeword,
			ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, probability);
		__syncwarp();
	}
}

/*
Magnitud Refinement Pass decoder management function. In this function, the bitplane is scanned from the first to the last bit, calling the MRPDecoder for each coefficient.
*/
template<class T>
__device__ void BPCEngine<T>::MRPDecoderLauncher(unsigned int* TData, int mask, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int  sharedPointer, volatile int* codeStreamShared, int probability, int bitPlane) {

	for (int i = 0; i < CBLOCK_LENGTH; i++)
	{
		__syncwarp();
		MRPDecoder(TData, i * 2, mask, codeStream, codeStreamPointer, codeword,
			ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, probability, bitPlane);
		__syncwarp();
		MRPDecoder(TData, (i * 2) + 1, mask, codeStream, codeStreamPointer, codeword,
			ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, probability, bitPlane);
		__syncwarp();
	}
}

/*
Massive Complexity Scalability encode management function. Responsible of calling multiple times to the arithmetic encoder per coefficient, considering its significance and its sign.
*/
template<class T>
__device__ void BPCEngine<T>::encodeBulkProcessing(int bitPlane, unsigned int* TData, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, int const* __restrict__ LUT, int LUTRefinementPointer, unsigned int TData2, unsigned int TData4, unsigned int TData5, unsigned int TData7, int context, int coeffIndex )
{
	int currentBitplane = bitPlane;
	int contextSign = 0;
	while (currentBitplane >= 0)
	{
		if (TData[(coeffIndex)] >> 31)
		{
			arithmeticEncoder((TData[(coeffIndex)] >> (currentBitplane + 1)) & 1, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTRefinementPointer - (_DLUTCtxtRefinement * (bitPlane - currentBitplane))]);
		}
		__syncwarp();
		if (!(TData[(coeffIndex)] >> 31))
		{
			arithmeticEncoder((TData[(coeffIndex)] >> (currentBitplane + 1)) & 1, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTSignificancePointer + context - (_DLUTCtxtSignificance * (bitPlane - currentBitplane))]);

			if (((TData[(coeffIndex)] >> (currentBitplane + 1)) & 1) == 1)
			{
				TData[(coeffIndex)] |= (1 << 31);

				TData[(coeffIndex)] |= (currentBitplane << 24);

				contextSign = computeSignContextBulk(TData2, TData4, TData5, TData7, currentBitplane);

				arithmeticEncoder(((TData[(coeffIndex)] & 1) == (contextSign & 1)) ? 0 : 1, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTSignPointer + (contextSign >> 1) - (_DLUTCtxtSign * (bitPlane - currentBitplane))]);
			}
		}
		__syncwarp();
		currentBitplane--;
	}
}

/*
Massive Complexity Scalability Encoding function, responsible of encoding the first of two strides of information per codeblock per thread. Responsible of creating context information per coefficient.
*/
template<class T>
__device__ void BPCEngine<T>::encodeLeftCoefficients(unsigned int* TData, int bitPlane, int i, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize,int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, int const* __restrict__ LUT, int LUTRefinementPointer)
{

	unsigned int upAuxiliar = 0;
	unsigned int middleAuxiliar = 0;
	unsigned int bottomAuxiliar = 0;

	int context = 0;
	int const coeffIndex = i * 2;
	shareLeftBorders(TData, (coeffIndex), &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);
	if (__any_sync(0xffffffff, (!(TData[(coeffIndex)] >> 31))) != 0)
	{
		//Share Borders and calculate context stuff.
		if (i == 0)
		{
			if ((threadIdx.x % 32) == 0)
			{
				middleAuxiliar = 0;
				bottomAuxiliar = 0;
			}
			//correctCBBorders(0, &middleAuxiliar, &bottomAuxiliar, 0, &TData[(coeffIndex) + 1], &TData[(coeffIndex) + 3], 1);
			if (bitPlane != 0)
				(context) = computeContextBulk(0, 0, 0, middleAuxiliar, TData[(coeffIndex) + 1], bottomAuxiliar, TData[(coeffIndex) + 2], TData[(coeffIndex) + 3], bitPlane);
			else
				(context) = computeContext(0, 0, 0, middleAuxiliar, TData[(coeffIndex) + 1], bottomAuxiliar, TData[(coeffIndex) + 2], TData[(coeffIndex) + 3]);
			encodeBulkProcessing(bitPlane, TData, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, 0, middleAuxiliar, TData[(coeffIndex)+1], TData[(coeffIndex)+2], context, coeffIndex);
		}
		else if (i == CBLOCK_LENGTH - 1)
		{
			if ((threadIdx.x % 32) == 0)
			{
				upAuxiliar = 0;
				middleAuxiliar = 0;
			}
			//correctCBBorders(&upAuxiliar, &middleAuxiliar, 0, &TData[(coeffIndex) - 1], &TData[(coeffIndex) + 1], 0, 1);
			if (bitPlane != 0)
				(context) = computeContextBulk(upAuxiliar, TData[(coeffIndex) - 2], TData[(coeffIndex) - 1], middleAuxiliar, TData[(coeffIndex) + 1], 0, 0, 0, bitPlane);
			else
				(context) = computeContext(upAuxiliar, TData[(coeffIndex) - 2], TData[(coeffIndex) - 1], middleAuxiliar, TData[(coeffIndex) + 1], 0, 0, 0);
			encodeBulkProcessing(bitPlane, TData, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, TData[(coeffIndex)-2], middleAuxiliar, TData[(coeffIndex)+1], 0, context, coeffIndex);
		}
		else
		{
			if ((threadIdx.x % 32) == 0)
			{
				upAuxiliar = 0;
				middleAuxiliar = 0;
				bottomAuxiliar = 0;
			}
			//correctCBBorders(&upAuxiliar, &middleAuxiliar, &bottomAuxiliar, &TData[(coeffIndex) - 1], &TData[(coeffIndex) + 1], &TData[(coeffIndex) + 3], 1);
			if (bitPlane != 0)
				(context) = computeContextBulk(upAuxiliar, TData[(coeffIndex) - 2], TData[(coeffIndex) - 1], middleAuxiliar, TData[(coeffIndex) + 1], bottomAuxiliar, TData[(coeffIndex) + 2], TData[(coeffIndex) + 3], bitPlane);
			else
				(context) = computeContext(upAuxiliar, TData[(coeffIndex) - 2], TData[(coeffIndex) - 1], middleAuxiliar, TData[(coeffIndex) + 1], bottomAuxiliar, TData[(coeffIndex) + 2], TData[(coeffIndex) + 3]);
			encodeBulkProcessing(bitPlane, TData, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, TData[(coeffIndex)-2], middleAuxiliar, TData[(coeffIndex)+1], TData[(coeffIndex)+2], context, coeffIndex);
		}
	}
	else
	{
		encodeBulkProcessing(bitPlane, TData, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, TData[(coeffIndex)-2], middleAuxiliar, TData[(coeffIndex)+1], TData[(coeffIndex)+2], context, coeffIndex);
	}
}

/*
Massive Complexity Scalability Encoding function, responsible of encoding the second of two strides of information per codeblock per thread. Responsible of creating context information per coefficient.
*/
template<class T>
__device__ void BPCEngine<T>::encodeRightCoefficients(unsigned int* TData, int bitPlane, int i, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, int const* __restrict__ LUT, int LUTRefinementPointer)
{

	unsigned int upAuxiliar = 0;
	unsigned int middleAuxiliar = 0;
	unsigned int bottomAuxiliar = 0;
	
	int context = 0;
	int const coeffIndex = (i * 2) + 1;
	shareRightBorders(TData, (coeffIndex), &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);
	//Share Borders and calculate context stuff.
	if (__any_sync(0xffffffff, (!(TData[(coeffIndex)] >> 31))) != 0)
	{
		if (i == 0)
		{
			if ((threadIdx.x % 32) == 31)
			{
				middleAuxiliar = 0;
				bottomAuxiliar = 0;
			}
			//correctCBBorders(0, &TData[(coeffIndex) - 1], &TData[(coeffIndex) + 1], 0, &middleAuxiliar, &bottomAuxiliar, 2);
			if (bitPlane != 0)
				(context) = computeContextBulk(0, 0, 0, TData[(coeffIndex) - 1], middleAuxiliar, TData[(coeffIndex) + 1], TData[(coeffIndex) + 2], bottomAuxiliar, bitPlane);
			else
				(context) = computeContext(0, 0, 0, TData[(coeffIndex) - 1], middleAuxiliar, TData[(coeffIndex) + 1], TData[(coeffIndex) + 2], bottomAuxiliar);
			encodeBulkProcessing(bitPlane, TData, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, 0, TData[(coeffIndex)-1], middleAuxiliar, TData[(coeffIndex)+2], context, coeffIndex);
		}
		else if (i == CBLOCK_LENGTH - 1)
		{
			if ((threadIdx.x % 32) == 31)
			{
				upAuxiliar = 0;
				middleAuxiliar = 0;
			}
			//correctCBBorders(&TData[(coeffIndex) - 3], &TData[(coeffIndex) - 1], 0, &upAuxiliar, &middleAuxiliar, 0, 2);
			if (bitPlane != 0)
				(context) = computeContextBulk(TData[(coeffIndex) - 3], TData[(coeffIndex) - 2], upAuxiliar, TData[(coeffIndex) - 1], middleAuxiliar, 0, 0, 0, bitPlane);
			else
				(context) = computeContext(TData[(coeffIndex) - 3], TData[(coeffIndex) - 2], upAuxiliar, TData[(coeffIndex) - 1], middleAuxiliar, 0, 0, 0);
			encodeBulkProcessing(bitPlane, TData, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, TData[(coeffIndex)-2], TData[(coeffIndex)-1], middleAuxiliar, 0, context, coeffIndex);
		}
		else
		{
			if ((threadIdx.x % 32) == 31)
			{
				upAuxiliar = 0;
				middleAuxiliar = 0;
				bottomAuxiliar = 0;
			}
			//correctCBBorders(&TData[(coeffIndex) - 3], &TData[(coeffIndex) - 1], &TData[(coeffIndex) + 1], &upAuxiliar, &middleAuxiliar, &bottomAuxiliar, 2);
			if (bitPlane != 0)
				(context) = computeContextBulk(TData[(coeffIndex) - 3], TData[(coeffIndex) - 2], upAuxiliar, TData[(coeffIndex) - 1], middleAuxiliar, TData[(coeffIndex) + 1], TData[(coeffIndex) + 2], bottomAuxiliar, bitPlane);
			else
				(context) = computeContext(TData[(coeffIndex) - 3], TData[(coeffIndex) - 2], upAuxiliar, TData[(coeffIndex) - 1], middleAuxiliar, TData[(coeffIndex) + 1], TData[(coeffIndex) + 2], bottomAuxiliar);
			encodeBulkProcessing(bitPlane, TData, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, TData[(coeffIndex)-2], TData[(coeffIndex)-1], middleAuxiliar, TData[(coeffIndex)+2], context, coeffIndex);
		}
	}
	else
	{
		encodeBulkProcessing(bitPlane, TData, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, TData[(coeffIndex)-2], middleAuxiliar, TData[(coeffIndex)+1], TData[(coeffIndex)+2], context, coeffIndex);
	}
}

/*
Massive Complexity Scalability decode management function. Responsible of calling multiple times to the arithmetic decoder per coefficient, considering its significance and its sign.
*/
template<class T>
__device__ void BPCEngine<T>::decodeBulkProcessing(int bitPlane, unsigned int* TData, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, int const* __restrict__ LUT, int LUTRefinementPointer, unsigned int TData2, unsigned int TData4, unsigned int TData5, unsigned int TData7, int context, int coeffIndex, int mask)
{
	int currentBitplane = bitPlane;
	int contextSign = 0;
	unsigned int symbol;
	while (currentBitplane >= 0)
	{
		symbol = 0;
		//Check if the input data is significant
		if (TData[coeffIndex] >> 31)
		{
			arithmeticDecoder(&symbol, codeStream, codeStreamPointer, codeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTRefinementPointer - (_DLUTCtxtRefinement * (bitPlane - currentBitplane))]);
			//Delete previous approximate bit
			TData[coeffIndex] &= (~mask);
			//Write new bit and approximation
			TData[coeffIndex] |= (mask & (((symbol << 1) + 1) << (currentBitplane)));
		}
		__syncwarp();
		if (!(TData[coeffIndex] >> 31))
		{
			arithmeticDecoder(&symbol, codeStream, codeStreamPointer, codeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTSignificancePointer + context - (_DLUTCtxtSignificance * (bitPlane - currentBitplane))]);

			if (symbol == 1)
			{
				TData[coeffIndex] |= mask;

				TData[coeffIndex] |= (1 << 31);

				TData[coeffIndex] |= (currentBitplane << 24);

				contextSign = computeSignContextBulk(TData2, TData4, TData5, TData7, currentBitplane);

				arithmeticDecoder(&symbol, codeStream, codeStreamPointer, codeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUT[LUTSignPointer + (contextSign >> 1) - (_DLUTCtxtSign * (bitPlane - currentBitplane))]);

				symbol = ((symbol & 1) == (contextSign & 1)) ? 0 : 1;

				TData[coeffIndex] |= (symbol & 1);
			}
		}
		__syncwarp();
		mask >>= 1;

		if (currentBitplane == 1) mask = 0x2;
		currentBitplane--;

	}
}

/*
Massive Complexity Scalability Decoding function, responsible of encoding the first of two strides of information per codeblock per thread. Responsible of creating context information per coefficient.
*/
template<class T>
__device__ void BPCEngine<T>::decodeLeftCoefficients(unsigned int* TData, int bitPlane, int i, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, int const* __restrict__ LUT, int LUTRefinementPointer, int mask)
{

	unsigned int upAuxiliar = 0;
	unsigned int middleAuxiliar = 0;
	unsigned int bottomAuxiliar = 0;

	int context = 0;
	int const coeffIndex = i * 2;
	shareLeftBorders(TData, (coeffIndex), &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);
	if (__any_sync(0xffffffff, (!(TData[(coeffIndex)] >> 31))) != 0)
	{
		//Share Borders and calculate context stuff.
		if (i == 0)
		{
			if ((threadIdx.x % 32) == 0)
			{
				middleAuxiliar = 0;
				bottomAuxiliar = 0;
			}
			//correctCBBorders(0, &middleAuxiliar, &bottomAuxiliar, 0, &TData[(coeffIndex) + 1], &TData[(coeffIndex) + 3], 1);
			if (bitPlane != 0)
				(context) = computeContextBulk(0, 0, 0, middleAuxiliar, TData[(coeffIndex)+1], bottomAuxiliar, TData[(coeffIndex)+2], TData[(coeffIndex)+3], bitPlane);
			else
				(context) = computeContext(0, 0, 0, middleAuxiliar, TData[(coeffIndex)+1], bottomAuxiliar, TData[(coeffIndex)+2], TData[(coeffIndex)+3]);
			decodeBulkProcessing(bitPlane, TData, codeStream, codeStreamPointer, codeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, 0, middleAuxiliar, TData[(coeffIndex)+1], TData[(coeffIndex)+2], context, coeffIndex, mask);
		}
		else if (i == CBLOCK_LENGTH - 1)
		{
			if ((threadIdx.x % 32) == 0)
			{
				upAuxiliar = 0;
				middleAuxiliar = 0;
			}
			//correctCBBorders(&upAuxiliar, &middleAuxiliar, 0, &TData[(coeffIndex) - 1], &TData[(coeffIndex) + 1], 0, 1);
			if (bitPlane != 0)
				(context) = computeContextBulk(upAuxiliar, TData[(coeffIndex)-2], TData[(coeffIndex)-1], middleAuxiliar, TData[(coeffIndex)+1], 0, 0, 0, bitPlane);
			else
				(context) = computeContext(upAuxiliar, TData[(coeffIndex)-2], TData[(coeffIndex)-1], middleAuxiliar, TData[(coeffIndex)+1], 0, 0, 0);
			decodeBulkProcessing(bitPlane, TData, codeStream, codeStreamPointer, codeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, TData[(coeffIndex)-2], middleAuxiliar, TData[(coeffIndex)+1], 0, context, coeffIndex, mask);
		}
		else
		{
			if ((threadIdx.x % 32) == 0)
			{
				upAuxiliar = 0;
				middleAuxiliar = 0;
				bottomAuxiliar = 0;
			}
			//correctCBBorders(&upAuxiliar, &middleAuxiliar, &bottomAuxiliar, &TData[(coeffIndex) - 1], &TData[(coeffIndex) + 1], &TData[(coeffIndex) + 3], 1);
			if (bitPlane != 0)
				(context) = computeContextBulk(upAuxiliar, TData[(coeffIndex)-2], TData[(coeffIndex)-1], middleAuxiliar, TData[(coeffIndex)+1], bottomAuxiliar, TData[(coeffIndex)+2], TData[(coeffIndex)+3], bitPlane);
			else
				(context) = computeContext(upAuxiliar, TData[(coeffIndex)-2], TData[(coeffIndex)-1], middleAuxiliar, TData[(coeffIndex)+1], bottomAuxiliar, TData[(coeffIndex)+2], TData[(coeffIndex)+3]);
			decodeBulkProcessing(bitPlane, TData, codeStream, codeStreamPointer, codeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, TData[(coeffIndex)-2], middleAuxiliar, TData[(coeffIndex)+1], TData[(coeffIndex)+2], context, coeffIndex, mask);
		}
	}
	else
	{
		decodeBulkProcessing(bitPlane, TData, codeStream, codeStreamPointer, codeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, TData[(coeffIndex)-2], middleAuxiliar, TData[(coeffIndex)+1], TData[(coeffIndex)+2], context, coeffIndex, mask);
	}
}

/*
Massive Complexity Scalability Decoding function, responsible of encoding the second of two strides of information per codeblock per thread. Responsible of creating context information per coefficient.
*/
template<class T>
__device__ void BPCEngine<T>::decodeRightCoefficients(unsigned int* TData, int bitPlane, int i, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, int const* __restrict__ LUT, int LUTRefinementPointer, int mask)
{

	unsigned int upAuxiliar = 0;
	unsigned int middleAuxiliar = 0;
	unsigned int bottomAuxiliar = 0;

	int context = 0;
	int const coeffIndex = (i * 2) + 1;
	shareRightBorders(TData, (coeffIndex), &upAuxiliar, &middleAuxiliar, &bottomAuxiliar);
	//Share Borders and calculate context stuff.
	if (__any_sync(0xffffffff, (!(TData[(coeffIndex)] >> 31))) != 0)
	{
		if (i == 0)
		{
			if ((threadIdx.x % 32) == 31)
			{
				middleAuxiliar = 0;
				bottomAuxiliar = 0;
			}
			//correctCBBorders(0, &TData[(coeffIndex) - 1], &TData[(coeffIndex) + 1], 0, &middleAuxiliar, &bottomAuxiliar, 2);
			if (bitPlane != 0)
				(context) = computeContextBulk(0, 0, 0, TData[(coeffIndex)-1], middleAuxiliar, TData[(coeffIndex)+1], TData[(coeffIndex)+2], bottomAuxiliar, bitPlane);
			else
				(context) = computeContext(0, 0, 0, TData[(coeffIndex)-1], middleAuxiliar, TData[(coeffIndex)+1], TData[(coeffIndex)+2], bottomAuxiliar);
			decodeBulkProcessing(bitPlane, TData, codeStream, codeStreamPointer, codeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, 0, TData[(coeffIndex)-1], middleAuxiliar, TData[(coeffIndex)+2], context, coeffIndex, mask);
		}
		else if (i == CBLOCK_LENGTH - 1)
		{
			if ((threadIdx.x % 32) == 31)
			{
				upAuxiliar = 0;
				middleAuxiliar = 0;
			}
			//correctCBBorders(&TData[(coeffIndex) - 3], &TData[(coeffIndex) - 1], 0, &upAuxiliar, &middleAuxiliar, 0, 2);
			if (bitPlane != 0)
				(context) = computeContextBulk(TData[(coeffIndex)-3], TData[(coeffIndex)-2], upAuxiliar, TData[(coeffIndex)-1], middleAuxiliar, 0, 0, 0, bitPlane);
			else
				(context) = computeContext(TData[(coeffIndex)-3], TData[(coeffIndex)-2], upAuxiliar, TData[(coeffIndex)-1], middleAuxiliar, 0, 0, 0);
			decodeBulkProcessing(bitPlane, TData, codeStream, codeStreamPointer, codeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, TData[(coeffIndex)-2], TData[(coeffIndex)-1], middleAuxiliar, 0, context, coeffIndex, mask);
		}
		else
		{
			if ((threadIdx.x % 32) == 31)
			{
				upAuxiliar = 0;
				middleAuxiliar = 0;
				bottomAuxiliar = 0;
			}
			//correctCBBorders(&TData[(coeffIndex) - 3], &TData[(coeffIndex) - 1], &TData[(coeffIndex) + 1], &upAuxiliar, &middleAuxiliar, &bottomAuxiliar, 2);
			if (bitPlane != 0)
				(context) = computeContextBulk(TData[(coeffIndex)-3], TData[(coeffIndex)-2], upAuxiliar, TData[(coeffIndex)-1], middleAuxiliar, TData[(coeffIndex)+1], TData[(coeffIndex)+2], bottomAuxiliar, bitPlane);
			else
				(context) = computeContext(TData[(coeffIndex)-3], TData[(coeffIndex)-2], upAuxiliar, TData[(coeffIndex)-1], middleAuxiliar, TData[(coeffIndex)+1], TData[(coeffIndex)+2], bottomAuxiliar);
			decodeBulkProcessing(bitPlane, TData, codeStream, codeStreamPointer, codeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, TData[(coeffIndex)-2], TData[(coeffIndex)-1], middleAuxiliar, TData[(coeffIndex)+2], context, coeffIndex, mask);
		}
	}
	else
	{
		decodeBulkProcessing(bitPlane, TData, codeStream, codeStreamPointer, codeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, TData[(coeffIndex)-2], middleAuxiliar, TData[(coeffIndex)+1], TData[(coeffIndex)+2], context, coeffIndex, mask);
	}
}

/*
Main Complexity Scalability encoding management function, responsible of invoking the processing, per thread, for each of the strides for the entire codeblock (64 rows). Each Warp has 32 threads, and each thread codes two strides, hence the left/right coefficients for the first/second column per thread.
*/
template<class T>
__device__ void BPCEngine<T>::encodeBulkMode(unsigned int* TData, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* reservedCodeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, int const* __restrict__ LUT, int LUTRefinementPointer)
{
	__syncwarp();
	for (int i = 0; i < CBLOCK_LENGTH; i++)
	{
		encodeLeftCoefficients(TData, bitPlane, i, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer);
		encodeRightCoefficients(TData, bitPlane, i, codeStream, codeStreamPointer, reservedCodeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer);
	}
}

/*
Main Complexity Scalability decoding management function, responsible of invoking the processing, per thread, for each of the strides for the entire codeblock (64 rows). Each Warp has 32 threads, and each thread codes two strides, hence the left/right coefficients for the first/second column per thread.
*/
template<class T>
__device__ void BPCEngine<T>::decodeBulkMode(unsigned int* TData, int bitPlane, VOLATILE int* codeStream, int codeStreamPointer, unsigned int* codeword, unsigned int* ACIntervalLower, unsigned int* ACIntervalSize, int sharedPointer, volatile int* codeStreamShared, int LUTSignificancePointer, int LUTSignPointer, int const* __restrict__ LUT, int LUTRefinementPointer, int mask)
{
	__syncwarp();
	for (int i = 0; i < CBLOCK_LENGTH; i++)
	{
		decodeLeftCoefficients(TData, bitPlane, i, codeStream, codeStreamPointer, codeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, mask);
		decodeRightCoefficients(TData, bitPlane, i, codeStream, codeStreamPointer, codeword, ACIntervalLower, ACIntervalSize, sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, mask);
	}
}

/*
Encode general function. In charge of starting the different scanning passes per codeblock and keeping the LUT values available for each function.
*/
template<class T>
__device__ void BPCEngine<T>::Encode(unsigned int* TData, int MSB, VOLATILE int* codeStream, int codeStreamPointer, int sharedPointer, volatile int* codeStreamShared, int CBDecompositionLevel, int CBSubband, int* sharedBuffer, int const* 	__restrict__ LUT, float kFactor)

{

	//---Ini
	unsigned int reservedCodeword = 0;
	unsigned int ACIntervalLower = 0;
	unsigned int ACIntervalSize = 0;

	int LUTRefinementPointer = 0;
	int LUTSignificancePointer = 0;
	int LUTSignPointer = 0;

	int bitPlane = MSB;
	
	//New CSPaCo Variables. They control the amount of bitplanes to be coded in block as well as the LUT that will be used to process the frame.
	int consecutiveBitplanes = 0;
	if (_DWaveletLevels == CBDecompositionLevel)
	{
		consecutiveBitplanes = max((int)floor(bitPlane * (kFactor / L2Norm[max(CBDecompositionLevel-1, 0)][0])),0);
	}
	else
	{
		consecutiveBitplanes = max((int)floor(bitPlane * (kFactor / L2Norm[CBDecompositionLevel][3 - CBSubband])),0);
	}

	if (kFactor > 0)
		initializeLUTPointers(&LUTRefinementPointer, &LUTSignificancePointer, &LUTSignPointer, CBDecompositionLevel, CBSubband, MSB, min(consecutiveBitplanes, MSB));
	else
		initializeLUTPointers(&LUTRefinementPointer, &LUTSignificancePointer, &LUTSignPointer, CBDecompositionLevel, CBSubband, MSB, 0);

	for (; bitPlane >= consecutiveBitplanes; bitPlane--)
	{

		SPPEncoderLauncher(TData, bitPlane, codeStream, codeStreamPointer, &reservedCodeword, &ACIntervalLower, &ACIntervalSize,
			sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT);

		MRPEncoderLauncher(TData, bitPlane, codeStream, codeStreamPointer, &reservedCodeword, &ACIntervalLower, &ACIntervalSize,
			sharedPointer, codeStreamShared, LUT[LUTRefinementPointer]);

		updateLUTPointers(&LUTRefinementPointer, &LUTSignificancePointer, &LUTSignPointer);

	}

	if (bitPlane >= 0)
	{
		encodeBulkMode(TData, bitPlane, codeStream, codeStreamPointer, &reservedCodeword, &ACIntervalLower, &ACIntervalSize,
			sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer);
	}
	
	//Flush not exhausted CW then the encoding process finish
	codeStream[reservedCodeword] = ACIntervalLower;

}

/*
Encode general function for 3 coding passes version. In charge of starting the different scanning passes per codeblock and keeping the LUT values available for each function.
*/
template<class T>
__device__ void BPCEngine<T>::Encode3CP(unsigned int* TData, int MSB, VOLATILE int* codeStream, int codeStreamPointer, int sharedPointer, volatile int* codeStreamShared, int CBDecompositionLevel, int CBSubband, int* sharedBuffer, int const* 	__restrict__ LUT)

{

	//---Ini
	unsigned int reservedCodeword = 0;
	unsigned int ACIntervalLower = 0;
	unsigned int ACIntervalSize = 0;

	int LUTRefinementPointer = 0;
	int LUTSignificancePointer = 0;
	int LUTSignPointer = 0;

	int bitPlane = MSB;

	initializeLUTPointers(&LUTRefinementPointer, &LUTSignificancePointer, &LUTSignPointer, CBDecompositionLevel, CBSubband, MSB, 0);

	int LUTPointerAux = ((_DLUTnOfSubbands) * (_DLUTnOfBitplanes) * (_DLUTCtxtSignificance) * (_DWaveletLevels)) + (_DLUTnOfBitplanes*_DLUTCtxtSignificance) +
		((_DLUTnOfSubbands) * (_DLUTnOfBitplanes) * (_DLUTCtxtSign) * (_DWaveletLevels)) + (_DLUTnOfBitplanes*_DLUTCtxtSign);

	CPEncoderLauncher(TData, bitPlane, codeStream, codeStreamPointer, &reservedCodeword, &ACIntervalLower, &ACIntervalSize,
		sharedPointer, codeStreamShared, LUTSignificancePointer + LUTPointerAux, LUTSignPointer + LUTPointerAux, sharedBuffer, LUT);
	bitPlane--;

	updateLUTPointers(&LUTRefinementPointer, &LUTSignificancePointer, &LUTSignPointer);

	for (; bitPlane >= 0; bitPlane--)
	{

		SPPEncoderLauncher3CP(TData, bitPlane, codeStream, codeStreamPointer, &reservedCodeword, &ACIntervalLower, &ACIntervalSize,
			sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT);

		MRPEncoderLauncher(TData, bitPlane, codeStream, codeStreamPointer, &reservedCodeword, &ACIntervalLower, &ACIntervalSize,
			sharedPointer, codeStreamShared, LUT[LUTRefinementPointer]);

		CPEncoderLauncher(TData, bitPlane, codeStream, codeStreamPointer, &reservedCodeword, &ACIntervalLower, &ACIntervalSize,
			sharedPointer, codeStreamShared, LUTSignificancePointer + LUTPointerAux, LUTSignPointer + LUTPointerAux, sharedBuffer, LUT);

		updateLUTPointers(&LUTRefinementPointer, &LUTSignificancePointer, &LUTSignPointer);

	}
	//Flush not exhausted CW then the encoding process finish	
	codeStream[reservedCodeword] = ACIntervalLower;
}

/*
Decode general function. In charge of starting the different scanning passes per codeblock and keeping the LUT values available for each function. The mask, initialized here, is used to approximate values when the decoded
bit is 1 and not 0.
*/
template<class T>
__device__ void BPCEngine<T>::Decode(unsigned int* TData, int MSB, VOLATILE int* codeStream, int codeStreamPointer, int sharedPointer, volatile int* codeStreamShared, int CBDecompositionLevel, int CBSubband, int* sharedBuffer, int const* 	__restrict__ LUT, float kFactor)

{

	//---Ini
	unsigned int codeword = 0;
	unsigned int ACIntervalLower = 0;
	unsigned int ACIntervalSize = 0;

	int LUTRefinementPointer = 0;
	int LUTSignificancePointer = 0;
	int LUTSignPointer = 0;

	int bitPlane = MSB;
	int mask = 0x3 << (bitPlane);

	//New CSPaCo Variables. They control the amount of bitplanes to be coded in block as well as the LUT that will be used to process the frame.
	int consecutiveBitplanes = 0;
	if (_DWaveletLevels == CBDecompositionLevel)
	{
		consecutiveBitplanes = max((int)floor(bitPlane * (kFactor / L2Norm[max(CBDecompositionLevel - 1, 0)][0])), 0);
	}
	else
	{
		consecutiveBitplanes = max((int)floor(bitPlane * (kFactor / L2Norm[CBDecompositionLevel][3 - CBSubband])), 0);
	}

	if (bitPlane == 0) mask &= 0x2;


	if (kFactor > 0)
		initializeLUTPointers(&LUTRefinementPointer, &LUTSignificancePointer, &LUTSignPointer, CBDecompositionLevel, CBSubband, MSB, min(consecutiveBitplanes, MSB));
	else
		initializeLUTPointers(&LUTRefinementPointer, &LUTSignificancePointer, &LUTSignPointer, CBDecompositionLevel, CBSubband, MSB, 0);

	for (; bitPlane >= consecutiveBitplanes; bitPlane--)	
	{

		SPPDecoderLauncher(TData, mask, codeStream, codeStreamPointer, &codeword, &ACIntervalLower, &ACIntervalSize,
			sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT, bitPlane);

		MRPDecoderLauncher(TData, mask, codeStream, codeStreamPointer, &codeword, &ACIntervalLower, &ACIntervalSize,
			sharedPointer, codeStreamShared, LUT[LUTRefinementPointer], bitPlane);


		mask >>= 1;

		if (bitPlane == 1) mask = 0x2;

		updateLUTPointers(&LUTRefinementPointer, &LUTSignificancePointer, &LUTSignPointer);


	}

	if (bitPlane >= 0)
	{
		decodeBulkMode(TData, bitPlane, codeStream, codeStreamPointer, &codeword, &ACIntervalLower, &ACIntervalSize,
			sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, LUT, LUTRefinementPointer, mask);
	}

}

/*
Decode general function for 3 coding passes. In charge of starting the different scanning passes per codeblock and keeping the LUT values available for each function. The mask, initialized here, is used to approximate values when the decoded
bit is 1 and not 0.
*/
template<class T>
__device__ void BPCEngine<T>::Decode3CP(unsigned int* TData, int MSB, VOLATILE int* codeStream, int codeStreamPointer, int sharedPointer, volatile int* codeStreamShared, int CBDecompositionLevel, int CBSubband, int* sharedBuffer, int const* 	__restrict__ LUT)

{

	//---Ini
	unsigned int codeword = 0;
	unsigned int ACIntervalLower = 0;
	unsigned int ACIntervalSize = 0;

	int LUTRefinementPointer = 0;
	int LUTSignificancePointer = 0;
	int LUTSignPointer = 0;

	int bitPlane = MSB;
	int mask = 0x3 << (bitPlane);

	if (bitPlane == 0) mask &= 0x2;

	initializeLUTPointers(&LUTRefinementPointer, &LUTSignificancePointer, &LUTSignPointer, CBDecompositionLevel, CBSubband, MSB, 0);

	int LUTPointerAux = ((_DLUTnOfSubbands) * (_DLUTnOfBitplanes) * (_DLUTCtxtSignificance) * (_DWaveletLevels)) + (_DLUTnOfBitplanes*_DLUTCtxtSignificance) +
		((_DLUTnOfSubbands) * (_DLUTnOfBitplanes) * (_DLUTCtxtSign) * (_DWaveletLevels)) + (_DLUTnOfBitplanes*_DLUTCtxtSign);

	CPDecoderLauncher(TData, mask, codeStream, codeStreamPointer, &codeword, &ACIntervalLower, &ACIntervalSize,
		sharedPointer, codeStreamShared, LUTSignificancePointer + LUTPointerAux, LUTSignPointer + LUTPointerAux, sharedBuffer, LUT, bitPlane);
	bitPlane--;

	mask >>= 1;

	if (bitPlane == 0) mask = 0x2;

	updateLUTPointers(&LUTRefinementPointer, &LUTSignificancePointer, &LUTSignPointer);

	for (; bitPlane >= 0; bitPlane--)
	{

		SPPDecoderLauncher3CP(TData, mask, codeStream, codeStreamPointer, &codeword, &ACIntervalLower, &ACIntervalSize,
			sharedPointer, codeStreamShared, LUTSignificancePointer, LUTSignPointer, sharedBuffer, LUT, bitPlane);

		MRPDecoderLauncher(TData, mask, codeStream, codeStreamPointer, &codeword, &ACIntervalLower, &ACIntervalSize,
			sharedPointer, codeStreamShared, LUT[LUTRefinementPointer], bitPlane);

		CPDecoderLauncher(TData, mask, codeStream, codeStreamPointer, &codeword, &ACIntervalLower, &ACIntervalSize,
			sharedPointer, codeStreamShared, LUTSignificancePointer + LUTPointerAux, LUTSignPointer + LUTPointerAux, sharedBuffer, LUT, bitPlane);

		mask >>= 1;

		if (bitPlane == 1) mask = 0x2;

		updateLUTPointers(&LUTRefinementPointer, &LUTSignificancePointer, &LUTSignPointer);


	}

}

/*
If the codeblock encoded is bigger in terms of data size than its original size, this function helps alleviate the issue. However, data is lost in the process and quality in general is hindered.
No lossless recovery is possible.
*/
template<class T>
__device__ void BPCEngine<T>::expansionFix(unsigned int* TData, int* output, int WCoordinate, int inputXSize, int* sizeArray, int nWarpsBlock)
{
	for (int i = 0; i < CBLOCK_LENGTH; i++)
	{
		output[WCoordinate + ((threadIdx.x % 32) * CBLOCK_LENGTH*NELEMENTS_THREAD_X) + (i * NELEMENTS_THREAD_X)] = ((int)(TData[i*NELEMENTS_THREAD_X])) & 0x0000FFFF;
		output[WCoordinate + ((threadIdx.x % 32) * CBLOCK_LENGTH*NELEMENTS_THREAD_X) + (i * NELEMENTS_THREAD_X + 1)] = ((int)(TData[(i*NELEMENTS_THREAD_X) + 1])) & 0x0000FFFF;
	}
}

template<class T>
__device__ void BPCEngine<T>::copyEntireCodeblock(unsigned int* TData, int* inputCodestream, int WCoordinate, int inputXSize)
{
	for (int i = 0; i < CBLOCK_LENGTH; i++)
	{
		TData[i * NELEMENTS_THREAD_X] = inputCodestream[WCoordinate + ((threadIdx.x % 32) * CBLOCK_LENGTH*NELEMENTS_THREAD_X) + (i * NELEMENTS_THREAD_X)];
		TData[i * NELEMENTS_THREAD_X + 1] = inputCodestream[WCoordinate + ((threadIdx.x % 32) * CBLOCK_LENGTH*NELEMENTS_THREAD_X) + (i * NELEMENTS_THREAD_X + 1)];
	}
}

//CUDA KERNELS

//ENCODER KERNEL
template<class T>
//__launch_bounds__(2048,16)
__global__ void kernelBPCCoder(

	T*	input,
	int*	output,
	int	inputXSize,
	int	inputYSize,
	int 	nWarpsBlock,
	int 	nWarpsX,
	int const* __restrict__	LUT,
	int const wLevels,
	int const LUTnOfBitplanes,
	int const LUTnOfSubbands,
	int const LUTCtxtRefinement,
	int const LUTCtxtSign,
	int const LUTCtxtSignificance,
	int const LUTMltPrecision,
	int *sizeArray,
	BPCEngine<T>* BPCE,
	float kFactor,
	int amountOfLUTFiles)
{

	_DWaveletLevels = wLevels;
	_DLUTnOfBitplanes = LUTnOfBitplanes;
	_DLUTnOfSubbands = LUTnOfSubbands;
	_DLUTCtxtRefinement = LUTCtxtRefinement;
	_DLUTCtxtSign = LUTCtxtSign;
	_DLUTCtxtSignificance = LUTCtxtSignificance;
	_DLUTMltPrecision = LUTMltPrecision;

	_LUTPointerSizePerS = (((_DLUTnOfSubbands) * (_DLUTnOfBitplanes) * (_DLUTCtxtRefinement) * (_DWaveletLevels)) + (_DLUTnOfBitplanes * _DLUTCtxtRefinement) +
		((_DLUTnOfSubbands) * (_DLUTnOfBitplanes) * (_DLUTCtxtSignificance) * (_DWaveletLevels)) + (_DLUTnOfBitplanes * _DLUTCtxtSignificance) +
		((_DLUTnOfSubbands) * (_DLUTnOfBitplanes) * (_DLUTCtxtSign) * (_DWaveletLevels)) + (_DLUTnOfBitplanes * _DLUTCtxtSign));

	__shared__ int sharedBuffer[1];

	const int sharedAmount = THREADBLOCK_SIZE / WARPSIZE;

	int sharedPerWarp = sharedAmount / nWarpsBlock;

	register unsigned int TData[CBLOCK_LENGTH * NELEMENTS_THREAD_X];
	__shared__ volatile int codeStreamShared[sharedAmount];


	int laneID = threadIdx.x & 0x1f;
	int warpID = (((threadIdx.x >> 5) + (blockIdx.x * nWarpsBlock)));
	int TCoordinateX = (((warpID % nWarpsX) * CBLOCK_WIDTH) + (laneID * NELEMENTS_THREAD_X));
	int TCoordinateY = ((warpID / nWarpsX) * CBLOCK_LENGTH);
	int TCoordinate = (inputXSize*TCoordinateY) + TCoordinateX;

	int WCoordinate = warpID * CBLOCK_LENGTH * CBLOCK_WIDTH;

	int MSB = 0;
	//CodeBlock decomposition level: from 0 to image decomposition levels
	int CBDecompositionLevel = -1;
	//CodeBlock Subband: LL = 0, HL = 0, LH = 1, HH = 2
	int CBSubband = -1;

	//if (warpID<(inputXSize / CBLOCK_WIDTH)*(inputYSize / CBLOCK_LENGTH)) {
	if (warpID < (ceilf((((float)inputXSize / (float)CBLOCK_WIDTH)*((float)inputYSize / (float)CBLOCK_LENGTH))))) {

		BPCE->findSubband(&CBDecompositionLevel, &CBSubband, TCoordinateX, TCoordinateY, inputXSize, inputYSize);

		BPCE->readCoefficients(input, (int*)TData, TCoordinate, inputXSize);

		BPCE->findMSB(TData, &MSB, sharedBuffer);

		codeStreamShared[(warpID % nWarpsBlock)*sharedPerWarp] = 0;

		output[WCoordinate] = MSB;
		if (MSB != 32)
			BPCE->Encode(TData, MSB, output, WCoordinate + 1, (warpID % nWarpsBlock), codeStreamShared, CBDecompositionLevel, CBSubband, sharedBuffer, LUT, kFactor);

		int maxElement = 0;
		int res = -1;
		
		__syncwarp();
		if (threadIdx.x % 32 == 0)
		{
			if (warpID < (ceil((((float)inputXSize / (float)CBLOCK_WIDTH)*((float)inputYSize / (float)CBLOCK_LENGTH)))))
			{
				sizeArray[warpID] = codeStreamShared[threadIdx.x >> 5] + 1;

			}
		}
		__syncwarp();
		if (sizeArray[warpID] == CBLOCK_WIDTH * CBLOCK_LENGTH)
		{
			BPCE->expansionFix(TData, output, WCoordinate, inputXSize, sizeArray, nWarpsBlock);
			if (MSB > 15)
			{
				printf("Data is being lost as the compression algorithm is unable to compress codeblock %d and the MSB is %d, which is over the limit of 15.\n", warpID, MSB);
			}
			else
				printf("Data is has not been compressed for codeblock %d, although no image quality has been lost in the process.\n", warpID);
		}
	}
}

template<class T>
__global__ void kernelBPCCoder3CP(

	T* input,
	int* output,
	int	inputXSize,
	int	inputYSize,
	int 	nWarpsBlock,
	int 	nWarpsX,
	int const* __restrict__	LUT,
	int const wLevels,
	int const LUTnOfBitplanes,
	int const LUTnOfSubbands,
	int const LUTCtxtRefinement,
	int const LUTCtxtSign,
	int const LUTCtxtSignificance,
	int const LUTMltPrecision,
	int *sizeArray,
	BPCEngine<T>* BPCE
)
{

	_DWaveletLevels = wLevels;
	_DLUTnOfBitplanes = LUTnOfBitplanes;
	_DLUTnOfSubbands = LUTnOfSubbands;
	_DLUTCtxtRefinement = LUTCtxtRefinement;
	_DLUTCtxtSign = LUTCtxtSign;
	_DLUTCtxtSignificance = LUTCtxtSignificance;
	_DLUTMltPrecision = LUTMltPrecision;



	__shared__ int sharedBuffer[1];

	const int sharedAmount = THREADBLOCK_SIZE / WARPSIZE;

	int sharedPerWarp = sharedAmount / nWarpsBlock;

	register unsigned int TData[CBLOCK_LENGTH * NELEMENTS_THREAD_X];
	__shared__ volatile int codeStreamShared[sharedAmount];


	int laneID = threadIdx.x & 0x1f;
	int warpID = (((threadIdx.x >> 5) + (blockIdx.x * nWarpsBlock)));
	int TCoordinateX = (((warpID % nWarpsX) * CBLOCK_WIDTH) + (laneID * NELEMENTS_THREAD_X));
	int TCoordinateY = ((warpID / nWarpsX) * CBLOCK_LENGTH);
	int TCoordinate = (inputXSize*TCoordinateY) + TCoordinateX;

	int WCoordinate = warpID * CBLOCK_LENGTH * CBLOCK_WIDTH;

	int MSB = 0;
	//CodeBlock decomposition level: from 0 to image decomposition levels
	int CBDecompositionLevel = -1;
	//CodeBlock Subband: LL = 0, HL = 0, LH = 1, HH = 2
	int CBSubband = -1;


	//if (warpID<(inputXSize / CBLOCK_WIDTH)*(inputYSize / CBLOCK_LENGTH)) {
	if (warpID < (ceilf((((float)inputXSize / (float)CBLOCK_WIDTH)*((float)inputYSize / (float)CBLOCK_LENGTH))))) {

		BPCE->findSubband(&CBDecompositionLevel, &CBSubband, TCoordinateX, TCoordinateY, inputXSize, inputYSize);

		BPCE->readCoefficients3CP(input, (int*)TData, TCoordinate, inputXSize);

		BPCE->findMSB3CP(TData, &MSB, sharedBuffer);

		codeStreamShared[(warpID % nWarpsBlock)*sharedPerWarp] = 0;

		output[WCoordinate] = MSB;
		if (MSB != 32)
			BPCE->Encode3CP(TData, MSB, output, WCoordinate + 1, (warpID % nWarpsBlock), codeStreamShared, CBDecompositionLevel, CBSubband, sharedBuffer, LUT);

		int maxElement = 0;
		int res = -1;
		
		__syncwarp();
		if (threadIdx.x % 32 == 0)
		{
			if (warpID < (ceil((((float)inputXSize / (float)CBLOCK_WIDTH)*((float)inputYSize / (float)CBLOCK_LENGTH)))))
				sizeArray[warpID] = codeStreamShared[threadIdx.x >> 5] + 1;
		}
		__syncwarp();
		if (sizeArray[warpID] == CBLOCK_WIDTH * CBLOCK_LENGTH)
		{
			BPCE->expansionFix(TData, output, WCoordinate, inputXSize, sizeArray, nWarpsBlock);
			if (MSB > 15)
			{
				printf("Data is being lost as the compression algorithm is unable to compress codeblock %d and the MSB is %d, which is over the limit of 15.\n", warpID, MSB);
			}
			else
				printf("Data is has not been compressed for codeblock %d, although no image quality has been lost in the process.\n", warpID);
		}
	}
}


//DECODER KERNEL
template<class T>
__global__ void kernelBPCDecoder(

	int*	inputCodestream,
	int*	outputImage,
	int	inputXSize,
	int	inputYSize,
	int 	nWarpsBlock,
	int 	nWarpsX,
	int const* __restrict__	LUT,
	int const wLevels,
	int const LUTnOfBitplanes,
	int const LUTnOfSubbands,
	int const LUTCtxtRefinement,
	int const LUTCtxtSign,
	int const LUTCtxtSignificance,
	int const LUTMltPrecision,
	int *sizeArray,
	BPCEngine<T>* BPCE,
	float kFactor,
	int amountOfLUTFiles

)
{

	_DWaveletLevels = wLevels;
	_DLUTnOfBitplanes = LUTnOfBitplanes;
	_DLUTnOfSubbands = LUTnOfSubbands;
	_DLUTCtxtRefinement = LUTCtxtRefinement;
	_DLUTCtxtSign = LUTCtxtSign;
	_DLUTCtxtSignificance = LUTCtxtSignificance;
	_DLUTMltPrecision = LUTMltPrecision;

	_LUTPointerSizePerS = (((_DLUTnOfSubbands) * (_DLUTnOfBitplanes) * (_DLUTCtxtRefinement) * (_DWaveletLevels)) + (_DLUTnOfBitplanes * _DLUTCtxtRefinement) +
		((_DLUTnOfSubbands) * (_DLUTnOfBitplanes) * (_DLUTCtxtSignificance) * (_DWaveletLevels)) + (_DLUTnOfBitplanes * _DLUTCtxtSignificance) +
		((_DLUTnOfSubbands) * (_DLUTnOfBitplanes) * (_DLUTCtxtSign) * (_DWaveletLevels)) + (_DLUTnOfBitplanes * _DLUTCtxtSign));

	__shared__ int sharedBuffer[1];

	const int sharedAmount = THREADBLOCK_SIZE / WARPSIZE;

	int sharedPerWarp = sharedAmount / nWarpsBlock;

	register unsigned int TData[CBLOCK_LENGTH * NELEMENTS_THREAD_X];

	__shared__ volatile int codeStreamShared[sharedAmount];


	int laneID = threadIdx.x & 0x1f;
	int warpID = (((threadIdx.x >> 5) + (blockIdx.x * nWarpsBlock)));
	int TCoordinateX = (((warpID % nWarpsX) * CBLOCK_WIDTH) + (laneID * NELEMENTS_THREAD_X));
	int TCoordinateY = ((warpID / nWarpsX) * CBLOCK_LENGTH);
	int TCoordinate = (inputXSize*TCoordinateY) + TCoordinateX;

	int WCoordinate = warpID * CBLOCK_LENGTH * CBLOCK_WIDTH;

	int MSB = 0;
	//CodeBlock decomposition level: from 0 to image decomposition levels
	int CBDecompositionLevel = -1;
	//CodeBlock Subband: LL = 0, HL = 0, LH = 1, HH = 2
	int CBSubband = -1;

	if (warpID < (ceilf((((float)inputXSize / (float)CBLOCK_WIDTH)*((float)inputYSize / (float)CBLOCK_LENGTH))))) {

		BPCE->findSubband(&CBDecompositionLevel, &CBSubband, TCoordinateX, TCoordinateY, inputXSize, inputYSize);

		BPCE->initializeCoefficients((int*)TData);

		codeStreamShared[(warpID % nWarpsBlock)*sharedPerWarp] = 0;

		MSB = inputCodestream[WCoordinate];

		if (sizeArray[warpID] == CBLOCK_LENGTH * CBLOCK_WIDTH)
		{
			BPCE->copyEntireCodeblock(TData, inputCodestream, WCoordinate, inputXSize);
		}
		else if (MSB != 32)
		{
			BPCE->Decode(TData, MSB, inputCodestream, WCoordinate + 1, (warpID % nWarpsBlock), codeStreamShared, CBDecompositionLevel, CBSubband, sharedBuffer, LUT, kFactor);
		}
		else
		{
			for (int i = 0; i < CBLOCK_LENGTH * NELEMENTS_THREAD_X; i++)
				TData[i] = 0;
		}
		__syncwarp();
		BPCE->writeCoefficients(outputImage, TData, TCoordinate, inputXSize);

	}

}



//DECODER KERNEL
template<class T>
__global__ void kernelBPCDecoder3CP(

	int*	inputCodestream,
	int*	outputImage,
	int	inputXSize,
	int	inputYSize,
	int 	nWarpsBlock,
	int 	nWarpsX,
	int const* __restrict__	LUT,
	int const wLevels,
	int const LUTnOfBitplanes,
	int const LUTnOfSubbands,
	int const LUTCtxtRefinement,
	int const LUTCtxtSign,
	int const LUTCtxtSignificance,
	int const LUTMltPrecision,
	int* sizeArray,
	BPCEngine<T>* BPCE

)
{

	_DWaveletLevels = wLevels;
	_DLUTnOfBitplanes = LUTnOfBitplanes;
	_DLUTnOfSubbands = LUTnOfSubbands;
	_DLUTCtxtRefinement = LUTCtxtRefinement;
	_DLUTCtxtSign = LUTCtxtSign;
	_DLUTCtxtSignificance = LUTCtxtSignificance;
	_DLUTMltPrecision = LUTMltPrecision;

	__shared__ int sharedBuffer[1];

	const int sharedAmount = THREADBLOCK_SIZE / WARPSIZE;

	int sharedPerWarp = sharedAmount / nWarpsBlock;

	register unsigned int TData[CBLOCK_LENGTH * NELEMENTS_THREAD_X];

	__shared__ volatile int codeStreamShared[sharedAmount];


	int laneID = threadIdx.x & 0x1f;
	int warpID = (((threadIdx.x >> 5) + (blockIdx.x * nWarpsBlock)));
	int TCoordinateX = (((warpID % nWarpsX) * CBLOCK_WIDTH) + (laneID * NELEMENTS_THREAD_X));
	int TCoordinateY = ((warpID / nWarpsX) * CBLOCK_LENGTH);
	int TCoordinate = (inputXSize*TCoordinateY) + TCoordinateX;

	int WCoordinate = warpID * CBLOCK_LENGTH * CBLOCK_WIDTH;

	int MSB = 0;
	//CodeBlock decomposition level: from 0 to image decomposition levels
	int CBDecompositionLevel = -1;
	//CodeBlock Subband: LL = 0, HL = 0, LH = 1, HH = 2
	int CBSubband = -1;

	if (warpID < (ceilf((((float)inputXSize / (float)CBLOCK_WIDTH)*((float)inputYSize / (float)CBLOCK_LENGTH))))) {

		BPCE->findSubband(&CBDecompositionLevel, &CBSubband, TCoordinateX, TCoordinateY, inputXSize, inputYSize);

		BPCE->initializeCoefficients3CP((int*)TData);

		codeStreamShared[(warpID % nWarpsBlock)*sharedPerWarp] = 0;

		MSB = inputCodestream[WCoordinate];
		if (sizeArray[warpID] == CBLOCK_LENGTH * CBLOCK_WIDTH)
		{
			BPCE->copyEntireCodeblock(TData, inputCodestream, WCoordinate, inputXSize);
		}
		else if (MSB != 32)
			BPCE->Decode3CP(TData, MSB, inputCodestream, WCoordinate + 1, (warpID % nWarpsBlock), codeStreamShared, CBDecompositionLevel, CBSubband, sharedBuffer, LUT);
		else
			for (int i = 0; i < CBLOCK_LENGTH * NELEMENTS_THREAD_X; i++)
				TData[i] = 0;
		__syncwarp();
		BPCE->writeCoefficients(outputImage, TData, TCoordinate, inputXSize);

	}

}


// -----------------------------------------------------------------------
//HOST FUNCTIONS ---------------------------------------------------------
// -----------------------------------------------------------------------

template<class T>
void BPCEngine<T>::kernelLauncher(int Direction, int DSizeX, int DSizeY, T* DDataInitial, int* DDataFinal, int LUTNumberOfBitplanes_, int LUTNumberOfSubbands_, int LUTContextRefinement_, int LUTContextSign_, int LUTContextSignificance_, int LUTMultPrecision_, int* LUTInformation_, int* DSizeArray, cudaStream_t mainStream, double *measurementsBPC) {

	int warpsRow = (int)ceil(DSizeX / (float)CBLOCK_WIDTH);
	int warpsColumn = (int)ceil(DSizeY / (float)CBLOCK_LENGTH);
	int CUDAWarpsNumber = warpsRow * warpsColumn;
	int CUDABlocksNumber = (int)ceil((CUDAWarpsNumber*WARPSIZE) / (float)(THREADBLOCK_SIZE));
	int *numberOfBlocksRemaining;

	_numberOfCodeblocks = (ceil((((float)DSizeX / (float)CBLOCK_WIDTH)*((float)DSizeY / (float)CBLOCK_LENGTH))));
	_prefixedArraySize = _numberOfCodeblocks;

	std::chrono::steady_clock::time_point startProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
	switch (Direction) {
	case CODE:

		if (_codingPasses == 2)
			kernelBPCCoder<T> << <CUDABlocksNumber, THREADBLOCK_SIZE, 0, mainStream >> >
			(

				DDataInitial,
				DDataFinal,
				DSizeX,
				DSizeY,
				THREADBLOCK_SIZE / WARPSIZE,
				warpsRow,
				LUTInformation_,
				_numberOfWaveletLevels,
				LUTNumberOfBitplanes_,
				LUTNumberOfSubbands_,
				LUTContextRefinement_,
				LUTContextSign_,
				LUTContextSignificance_,
				LUTMultPrecision_,
				DSizeArray,
				this,
				_kFactor,
				_amountOfLUTFiles
				);

		else
			kernelBPCCoder3CP<T> << <CUDABlocksNumber, THREADBLOCK_SIZE, 0, mainStream >> >
			(

				DDataInitial,
				DDataFinal,
				DSizeX,
				DSizeY,
				THREADBLOCK_SIZE / WARPSIZE,
				warpsRow,
				LUTInformation_,
				_numberOfWaveletLevels,
				LUTNumberOfBitplanes_,
				LUTNumberOfSubbands_,
				LUTContextRefinement_,
				LUTContextSign_,
				LUTContextSignificance_,
				LUTMultPrecision_,
				DSizeArray,
				this
				);
		break;

	case DECODE:

		if (_codingPasses == 2)
			kernelBPCDecoder<T> << <CUDABlocksNumber, THREADBLOCK_SIZE, 0, mainStream >> >
			(

			(int*)DDataInitial,
				DDataFinal,
				DSizeX,
				DSizeY,
				THREADBLOCK_SIZE / WARPSIZE,
				warpsRow,
				LUTInformation_,
				_numberOfWaveletLevels,
				LUTNumberOfBitplanes_,
				LUTNumberOfSubbands_,
				LUTContextRefinement_,
				LUTContextSign_,
				LUTContextSignificance_,
				LUTMultPrecision_,
				DSizeArray,
				this,
				_kFactor,
				_amountOfLUTFiles
				);

		else
			kernelBPCDecoder3CP<T> << <CUDABlocksNumber, THREADBLOCK_SIZE, 0, mainStream >> >
			(

			(int*)DDataInitial,
				DDataFinal,
				DSizeX,
				DSizeY,
				THREADBLOCK_SIZE / WARPSIZE,
				warpsRow,
				LUTInformation_,
				_numberOfWaveletLevels,
				LUTNumberOfBitplanes_,
				LUTNumberOfSubbands_,
				LUTContextRefinement_,
				LUTContextSign_,
				LUTContextSignificance_,
				LUTMultPrecision_,
				DSizeArray,
				this
				);
		break;
	}
	cudaStreamSynchronize(mainStream);

	auto finishProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
	auto finishMeasurement = std::chrono::duration_cast<std::chrono::duration<double>>(finishProcessDisregardAllocationTimings - startProcessDisregardAllocationTimings).count();
	*measurementsBPC = *measurementsBPC + finishMeasurement;
	KERNEL_ERROR_HANDLER;
}
/*
Function responsible of setting the output memory region to default values to avoid errors in the coding process.
*/
template<class T>
void BPCEngine<T>::deviceMemoryAllocator(int size, int** DDataFinal, int Direction, cudaStream_t mainStream)
{
	size_t 		DSize = size * sizeof(int);

	if (Direction == CODE)
	{
		GPU_HANDLE_ERROR(cudaMemsetAsync(*DDataFinal, -1, DSize, mainStream));
	}
	else
	{
		GPU_HANDLE_ERROR(cudaMemsetAsync(*DDataFinal, -1, DSize, mainStream));
	}
}

template<class T>
BPCEngine<T>::BPCEngine(int wLevels, int cp, float k, int amountOfLUTFiles)
{
	_numberOfWaveletLevels = wLevels;
	_codingPasses = cp;
	_kFactor = k;
	_amountOfLUTFiles = amountOfLUTFiles;
}

template<class T>
int BPCEngine<T>::getNumberOfCodeblocks() const
{
	return BPCEngine::_numberOfCodeblocks;
}

template<class T>
int BPCEngine<T>::getPrefixedArraySize() const
{
	return BPCEngine::_prefixedArraySize;
}
