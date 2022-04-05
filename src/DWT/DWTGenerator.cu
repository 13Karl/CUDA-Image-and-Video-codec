#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "DWTGenerator.cuh"



template<class T, class Y>
inline void DWTEngine<T, Y>::gpuAssert(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
		exit(EXIT_FAILURE);
	}
}
template<class T, class Y>
void DWTEngine<T, Y>::cudaKernelAssert(const char *file, int line)
{
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		std::cout << "cudaKernelAssert() failed at " << file << ":" << line << ":" << cudaGetErrorString(err) << std::endl;
		exit(-1);
	}

}
/*


Functions in this code are organized in 7 sections. 2 Sections corresponds to functions executed in the host and the other 5 are functions used in the device.

Host functions sections:

1. PRE/POST COMPUTE FUNCTIONS: Allocate and release device memory.
2. DWT FUNCTIONS: Precomputing needed to apply N levels of the DWT over an input, launch a kernel for each DWT level.

Device functions sections:

3. CUDA KERNELS: Each CUDA kernel computes a single DWT level (vertical + horizontal filter).
4. PRE-COMPUTE FUNCTIONS: Inline device functions used by the kernels to (mainly) compute in which input coordinates each warp has to work (asign a data block to a warp).
5. DATA MANAGEMENT FUNCTIONS: Inline device functions used by the kernels to read (or write) a data block from the device main memory to the registers, and the other way around.
6. FILTER COMPUTATION FUNCTIONS: Inline device functions used by the kernels to compute the vertical or horizontal filter over a full data block.
7. FILTER KERNEL FUNCTIONS:  Inline device functions used by the kernels to compute a lifting step operation over 3 samples.


Example of function call flow:

<HOST> PRE/POST COMPUTE FUNCTIONS
|
|__ <HOST> DWT FUNCTIONS
|
|__ <DEVICE> CUDA KERNELS
|
|__ <DEVICE> PRE-COMPUTE FUNCTIONS
<DEVICE> DATA MANAGEMENT FUNCTIONS
<DEVICE> FILTER COMPUTATION FUNCTIONS
|
|__ <DEVICE> FILTER KERNEL FUNCTIONS

*/


/**************************************************************

START - <DEVICE> FILTER KERNEL FUNCTIONS

**************************************************************/

//CDF 5/3 (1st Lifting Step) - FORWARD 
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::liftingStepOne53Forward(T a, VOLATILE T* b, T c) { *b -= ((int)(a + c) >> 1); }

//CDF 5/3 (2nd Lifting Step) - FORWARD 
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::liftingStepTwo53Forward(T a, VOLATILE T* b, T c) { *b += ((int)(a + c + 2) >> 2); }


//CDF 5/3 (1st Lifting Step) - REVERSE
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::liftingStepOne53Reverse(T a, VOLATILE T* b, T c) { *b -= ((int)((a + c + 2)) >> 2); }

//CDF 5/3 (2nd Lifting Step) - REVERSE
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::liftingStepTwo53Reverse(T a, VOLATILE T* b, T c) { *b += (((int)(a + c) >> 1)); }



//CDF 9/7 (1st Lifting Step) - FORWARD 
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::liftingStepOne97Forward(T a, VOLATILE T* b, T c) { *b += ((a + c)* LIFTING_STEPS_I97_1); 
}

//CDF 9/7 (2nd Lifting Step) - FORWARD 
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::liftingStepTwo97Forward(T a, VOLATILE T* b, T c) { *b += ((a + c)* LIFTING_STEPS_I97_2); }

//CDF 9/7 (3rd Lifting Step) - FORWARD 
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::liftingStepThree97Forward(T a, VOLATILE T* b, T c) { *b += ((a + c)* LIFTING_STEPS_I97_3); }

//CDF 9/7 (4th Lifting Step) - FORWARD + normalization
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::liftingStepFour97Forward(T a, VOLATILE T* b, T c) { *b = (*b + ((a + c)* LIFTING_STEPS_I97_4))*NORMALIZATION_I97_2; }



//CDF 9/7 (1st Lifting Step) - REVERSE + normalization
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::liftingStepOne97Reverse(T a, VOLATILE T* b, T c) { *b = (*b / NORMALIZATION_I97_2) - ((a + c)* LIFTING_STEPS_I97_4); }

//CDF 9/7 (2nd Lifting Step) - REVERSE
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::liftingStepTwo97Reverse(T a, VOLATILE T* b, T c) { *b -= ((a + c)* LIFTING_STEPS_I97_3); }

//CDF 9/7 (3rd Lifting Step) - REVERSE
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::liftingStepThree97Reverse(T a, VOLATILE T* b, T c) { *b -= ((a + c)* LIFTING_STEPS_I97_2); }

//CDF 9/7 (4th Lifting Step) - REVERSE
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::liftingStepFour97Reverse(T a, VOLATILE T* b, T c) { *b -= ((a + c)* LIFTING_STEPS_I97_1); }

//END - <DEVICE> FILTER KERNEL FUNCTIONS -----------------------------------------------------------------------

/**************************************************************
//START - <DEVICE> FILTER COMPUTATION FUNCTIONS
**************************************************************/



//VERTICAL FILTER FUNCTIONS , generic for all versions <shuffle instructions, shared memory with auxiliary buffer or full shared memory>
//Each vertical function applies ecuations to transform the image sample into wavelet coefficients.


template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::verticalFilterForward(T* TData, int TDSizeY, int TDSizeX)
{
	int TDSizeYIndex = 0;

	for (int TDSizeXIndex = 0; TDSizeXIndex < TDSizeX; TDSizeXIndex++)
	{
		for (TDSizeYIndex = 1; TDSizeYIndex < (TDSizeY - 1); TDSizeYIndex += 2)
		{
			liftingStepOne53Forward(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}

		liftingStepOne53Forward(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*threadIdx.x)], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*threadIdx.x)], TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*threadIdx.x)]);
		TDSizeYIndex = 0;
		liftingStepTwo53Forward(TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*threadIdx.x)], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*threadIdx.x)], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*threadIdx.x)]);

		for (TDSizeYIndex = 2; TDSizeYIndex < TDSizeY; TDSizeYIndex += 2)
		{
			liftingStepTwo53Forward(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*threadIdx.x)], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*threadIdx.x)], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*threadIdx.x)]);
		}
	}
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::verticalFilterReverse(T* TData, int TDSizeY, int TDSizeX)
{
	int TDSizeYIndex = 0;

	for (int TDSizeXIndex = 0; TDSizeXIndex < TDSizeX; TDSizeXIndex++)
	{
		TDSizeYIndex = 0;
		liftingStepOne53Reverse(TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);

		for (TDSizeYIndex = 2; TDSizeYIndex < TDSizeY; TDSizeYIndex += 2)
		{
			liftingStepOne53Reverse(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}

		for (TDSizeYIndex = 1; TDSizeYIndex < (TDSizeY - 1); TDSizeYIndex += 2)
		{
			liftingStepTwo53Reverse(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}

		liftingStepTwo53Reverse(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);
	}
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::verticalFilterForwardLossy(T* TData, int TDSizeY, int TDSizeX)
{

	int TDSizeYIndex = 0;

	for (int TDSizeXIndex = 0; TDSizeXIndex < TDSizeX; TDSizeXIndex++)
	{
		for (TDSizeYIndex = 1; TDSizeYIndex < (TDSizeY - 1); TDSizeYIndex += 2)
		{
			liftingStepOne97Forward(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}

		liftingStepOne97Forward(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		TDSizeYIndex = 0;
		liftingStepTwo97Forward(TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);

		for (TDSizeYIndex = 2; TDSizeYIndex < TDSizeY; TDSizeYIndex += 2)
		{
			liftingStepTwo97Forward(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}

		for (TDSizeYIndex = 1; TDSizeYIndex < (TDSizeY - 1); TDSizeYIndex += 2)
		{
			liftingStepThree97Forward(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}

		liftingStepThree97Forward(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		TDSizeYIndex = 0;
		liftingStepFour97Forward(TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);


		for (TDSizeYIndex = 2; TDSizeYIndex < TDSizeY; TDSizeYIndex += 2)
		{
			liftingStepFour97Forward(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}

		for (TDSizeYIndex = 1; TDSizeYIndex < (TDSizeY - 1); TDSizeYIndex += 2)
		{
			TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))] *= NORMALIZATION_I97_1;
		}

		TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))] *= NORMALIZATION_I97_1;
	}
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::verticalFilterReverseLossy(T* TData, int TDSizeY, int TDSizeX)
{

	int TDSizeYIndex = 0;

	for (int TDSizeXIndex = 0; TDSizeXIndex < TDSizeX; TDSizeXIndex++)
	{
		for (TDSizeYIndex = 1; TDSizeYIndex < (TDSizeY - 1); TDSizeYIndex += 2)
		{
			TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))] /= NORMALIZATION_I97_1;
		}

		TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))] /= NORMALIZATION_I97_1;
		TDSizeYIndex = 0;
		liftingStepOne97Reverse(TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);

		for (TDSizeYIndex = 2; TDSizeYIndex < TDSizeY; TDSizeYIndex += 2)
		{
			liftingStepOne97Reverse(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}

		for (TDSizeYIndex = 1; TDSizeYIndex < (TDSizeY - 1); TDSizeYIndex += 2)
		{
			liftingStepTwo97Reverse(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}

		liftingStepTwo97Reverse(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		TDSizeYIndex = 0;
		liftingStepThree97Reverse(TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);

		for (TDSizeYIndex = 2; TDSizeYIndex < TDSizeY; TDSizeYIndex += 2)
		{
			liftingStepThree97Reverse(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}

		for (TDSizeYIndex = 1; TDSizeYIndex < (TDSizeY - 1); TDSizeYIndex += 2)
		{
			liftingStepFour97Reverse(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex + 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}

		liftingStepFour97Reverse(TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSizeYIndex * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSizeYIndex - 1) * 2) + TDSizeXIndex + (SHARED_MEMORY_STRIDE*(threadIdx.x))]);
	}
}

//SHUFFLE INSTRUCTIONS - HORIZONTAL FILTER FUNCTIONS
//Each horizontal function applies ecuations to transform the image sample into wavelet coefficients.


template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::horizontalFilterForwardShuffle(T* TData, int TDSizeY, int TDSizeX)
{
	int TDSizeXIndex = TDSizeX >> 1;

	for (int TDSizeYIndex = 0; TDSizeYIndex < TDSizeY; TDSizeYIndex++)
	{
		liftingStepOne53Forward(TData[TDSizeYIndex * 2], &TData[(TDSizeYIndex * 2) + TDSizeXIndex], __shfl_down_sync(0xffffffff, TData[TDSizeYIndex * 2], 1));
	}

	for (int TDSizeYIndex = 0; TDSizeYIndex < TDSizeY; TDSizeYIndex++)
	{
		liftingStepTwo53Forward(TData[(TDSizeYIndex * 2) + TDSizeXIndex], &TData[TDSizeYIndex * 2], __shfl_up_sync(0xffffffff, TData[(TDSizeYIndex * 2) + TDSizeXIndex], 1));
	}
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::horizontalFilterReverseShuffle(T* TData, int TDSizeY, int TDSizeX)
{
	int TDSizeXIndex = TDSizeX >> 1;

	for (int TDSizeYIndex = 0; TDSizeYIndex < TDSizeY; TDSizeYIndex++)
	{
		liftingStepOne53Reverse(TData[(TDSizeYIndex * 2) + TDSizeXIndex], &TData[TDSizeYIndex * 2], __shfl_up_sync(0xffffffff, TData[(TDSizeYIndex * 2) + TDSizeXIndex], 1));
    }

	for (int TDSizeYIndex = 0; TDSizeYIndex < TDSizeY; TDSizeYIndex++)
	{
		liftingStepTwo53Reverse(TData[TDSizeYIndex * 2], &TData[(TDSizeYIndex * 2) + TDSizeXIndex], __shfl_down_sync(0xffffffff, TData[TDSizeYIndex * 2], 1));
	}
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::horizontalFilterForwardLossyShuffle(T* TData, int TDSizeY, int TDSizeX)
{
	int TDSizeXIndex = TDSizeX >> 1;

	for (int TDSizeYIndex = 0; TDSizeYIndex < TDSizeY; TDSizeYIndex++) {
		liftingStepOne97Forward(TData[TDSizeYIndex * 2], &TData[(TDSizeYIndex * 2) + TDSizeXIndex], __shfl_down_sync(0xffffffff, TData[TDSizeYIndex * 2], 1));
		liftingStepTwo97Forward(TData[(TDSizeYIndex * 2) + TDSizeXIndex], &TData[TDSizeYIndex * 2], __shfl_up_sync(0xffffffff, TData[(TDSizeYIndex * 2) + TDSizeXIndex], 1));
		liftingStepThree97Forward(TData[TDSizeYIndex * 2], &TData[(TDSizeYIndex * 2) + TDSizeXIndex], __shfl_down_sync(0xffffffff, TData[TDSizeYIndex * 2], 1));
		liftingStepFour97Forward(TData[(TDSizeYIndex * 2) + TDSizeXIndex], &TData[TDSizeYIndex * 2], __shfl_up_sync(0xffffffff, TData[(TDSizeYIndex * 2) + TDSizeXIndex], 1));

		TData[(TDSizeYIndex * 2) + TDSizeXIndex] *= NORMALIZATION_I97_1;
	}
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::horizontalFilterReverseLossyShuffle(T* TData, int TDSizeY, int TDSizeX)
{
	int TDSizeXIndex = TDSizeX >> 1;

	for (int TDSizeYIndex = 0; TDSizeYIndex < TDSizeY; TDSizeYIndex++) {

		TData[(TDSizeYIndex * 2) + TDSizeXIndex] /= NORMALIZATION_I97_1;

		liftingStepOne97Reverse(TData[(TDSizeYIndex * 2) + TDSizeXIndex], &TData[TDSizeYIndex * 2], __shfl_up_sync(0xffffffff, TData[(TDSizeYIndex * 2) + TDSizeXIndex], 1));
		liftingStepTwo97Reverse(TData[TDSizeYIndex * 2], &TData[(TDSizeYIndex * 2) + TDSizeXIndex], __shfl_down_sync(0xffffffff, TData[TDSizeYIndex * 2], 1));
		liftingStepThree97Reverse(TData[(TDSizeYIndex * 2) + TDSizeXIndex], &TData[TDSizeYIndex * 2], __shfl_up_sync(0xffffffff, TData[(TDSizeYIndex * 2) + TDSizeXIndex], 1));
		liftingStepFour97Reverse(TData[TDSizeYIndex * 2], &TData[(TDSizeYIndex * 2) + TDSizeXIndex], __shfl_down_sync(0xffffffff, TData[TDSizeYIndex * 2], 1));
	}
}

//END - <DEVICE> FILTER COMPUTATION FUNCTIONS -----------------------------------------------------------------------


//START - <DEVICE> DATA MANAGEMENT FUNCTIONS -----------------------------------------------------------------------
//These functions are in charge of updating pointers and memory regions to the proper data structures that must be coded per thread.


template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::updateSubbandsCoordinates(int DSizeCurrentX, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH)
{
	*TCoordinateLL += DSizeCurrentX;
	*TCoordinateHL += DSizeCurrentX;
	*TCoordinateLH += DSizeCurrentX;
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::updateSubbandsCoordinatesLLAux(int DSizeCurrentX, int DSizeInitialX, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH)
{
	*TCoordinateLL += (DSizeCurrentX >> 1);
	*TCoordinateHL += DSizeInitialX;
	*TCoordinateLH += DSizeInitialX;
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::updateSubbandsCoordinatesScheduler(int DSizeCurrentX, int DSizeInitialX, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int lastLevel)
{
	if (lastLevel)	updateSubbandsCoordinates(DSizeInitialX, TCoordinateLL, TCoordinateHL, TCoordinateLH);
	else			updateSubbandsCoordinatesLLAux(DSizeCurrentX, DSizeInitialX, TCoordinateLL, TCoordinateHL, TCoordinateLH);
}

//Function that reads the information from global memory to transfer it to local registers.
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::readBlock2Char(uchar2* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate)
{

	*(TCoordinate) >>= 1;
	for (int y = 0; y < TDSizeY; y++)
	{
		TData[(y * 2)] = (T)data[*TCoordinate].x;
		TData[(y * 2) + 1 + (SHARED_MEMORY_STRIDE*threadIdx.x)] = (T)data[*TCoordinate].y;
		*TCoordinate += DSizeCurrentX;
	}
	*(TCoordinate) <<= 1;
}

//Function that reads the information from global memory to transfer it to local registers.
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::readBlock2(Y* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate)
{

	*(TCoordinate) >>= 1;
	for (int y = 0; y < TDSizeY; y++) 
	{
		TData[(y * 2)] = data[*TCoordinate].x;
		TData[(y * 2) + 1 + (SHARED_MEMORY_STRIDE*threadIdx.x)] = data[*TCoordinate].y;
		*TCoordinate += DSizeCurrentX;
	}
	*(TCoordinate) <<= 1;
}

//Writes information to the output and applies quantization if the wavelet applied is 9/7.
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeSubbands(T* data, int DSizeInitialX, int DSizeCurrentX, T* TData, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int* index, int lastLevel, int currentWLevel, bool isLossy, float qSize) {

	if (isLossy)
	{
		if (lastLevel)
			data[*TCoordinateLL] = (TData[((*index) << 1) + (SHARED_MEMORY_STRIDE*threadIdx.x)]) * _DWaveletQSteps[currentWLevel][0] * qSize;
		else
			data[*TCoordinateLL] = (TData[((*index) << 1) + (SHARED_MEMORY_STRIDE*threadIdx.x)]);
		data[*TCoordinateHL] = (TData[((*index) << 1) + 1 + (SHARED_MEMORY_STRIDE*threadIdx.x)]) * _DWaveletQSteps[currentWLevel][1] * qSize;

		++(*index);

		data[*TCoordinateLH] = (TData[((*index) << 1) + (SHARED_MEMORY_STRIDE*threadIdx.x)]) * _DWaveletQSteps[currentWLevel][2] * qSize;
		data[*TCoordinateLH + (DSizeCurrentX >> 1)] = (TData[((*index) << 1) + 1 + (SHARED_MEMORY_STRIDE*threadIdx.x)]) * _DWaveletQSteps[currentWLevel][3] * qSize;

		updateSubbandsCoordinatesScheduler(DSizeCurrentX, DSizeInitialX, TCoordinateLL, TCoordinateHL, TCoordinateLH, lastLevel);
	}
	else
	{
		data[*TCoordinateLL] = (TData[((*index) << 1) + (SHARED_MEMORY_STRIDE*threadIdx.x)]);
		data[*TCoordinateHL] = (TData[((*index) << 1) + 1 + (SHARED_MEMORY_STRIDE*threadIdx.x)]);

		++(*index);

		data[*TCoordinateLH] = (TData[((*index) << 1) + (SHARED_MEMORY_STRIDE*threadIdx.x)]);
		data[*TCoordinateLH + (DSizeCurrentX >> 1)] = (TData[((*index) << 1) + 1 + (SHARED_MEMORY_STRIDE*threadIdx.x)]);

		updateSubbandsCoordinatesScheduler(DSizeCurrentX, DSizeInitialX, TCoordinateLL, TCoordinateHL, TCoordinateLH, lastLevel);
	}

}

//Following functions manages the position and placement of the information in the input/output structures.

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeSubbandsTop(T* data, int DSizeCurrentX, int DSizeInitialX, T* TData, int TDSizeY, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int lastLevel, int overlap, int currentWLevel, bool isLossy, float qSize)
{
	for (int y = 0; y < (TDSizeY - (overlap >> 1)); y++)
		writeSubbands(data, DSizeInitialX, DSizeCurrentX, TData, TCoordinateLL, TCoordinateHL, TCoordinateLH, &y, lastLevel, currentWLevel, isLossy, qSize);
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeSubbandsMiddle(T* data, int DSizeCurrentX, int DSizeInitialX, T* TData, int TDSizeY, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int lastLevel, int overlap, int currentWLevel, bool isLossy, float qSize)
{
	for (int y = 0; y < (overlap >> 1); y += 2)
		updateSubbandsCoordinatesScheduler(DSizeCurrentX, DSizeInitialX, TCoordinateLL, TCoordinateHL, TCoordinateLH, lastLevel);

	for (int y = (overlap >> 1); y < (TDSizeY - (overlap >> 1)); y++)
		writeSubbands(data, DSizeInitialX, DSizeCurrentX, TData, TCoordinateLL, TCoordinateHL, TCoordinateLH, &y, lastLevel, currentWLevel, isLossy, qSize);
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeSubbandsBottom(T* data, int DSizeCurrentX, int DSizeInitialX, T* TData, int TDSizeY, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int lastLevel, int overlap, int currentWLevel, bool isLossy, float qSize)
{
	for (int y = 0; y < (overlap >> 1); y += 2)

		updateSubbandsCoordinatesScheduler(DSizeCurrentX, DSizeInitialX, TCoordinateLL, TCoordinateHL, TCoordinateLH, lastLevel);

	for (int y = (overlap >> 1); y < TDSizeY; y++)

		writeSubbands(data, DSizeInitialX, DSizeCurrentX, TData, TCoordinateLL, TCoordinateHL, TCoordinateLH, &y, lastLevel, currentWLevel, isLossy, qSize);
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeSubbandsScheduler(T* data, int DSizeCurrentX, int DSizeInitialX, T* TData, int TDSizeY, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int lastLevel, int incorrectVerticalTop, int incorrectVerticalBottom, int overlap, int currentWLevel, bool isLossy, float qSize)
{
	if (incorrectVerticalTop == 0)		writeSubbandsTop(data, DSizeCurrentX, DSizeInitialX, TData, TDSizeY, TCoordinateLL, TCoordinateHL, TCoordinateLH, lastLevel, overlap, currentWLevel, isLossy, qSize);
	else if (incorrectVerticalBottom == 0)		writeSubbandsBottom(data, DSizeCurrentX, DSizeInitialX, TData, TDSizeY, TCoordinateLL, TCoordinateHL, TCoordinateLH, lastLevel, overlap, currentWLevel, isLossy, qSize);
	else											writeSubbandsMiddle(data, DSizeCurrentX, DSizeInitialX, TData, TDSizeY, TCoordinateLL, TCoordinateHL, TCoordinateLH, lastLevel, overlap, currentWLevel, isLossy, qSize);

}

//Reads the information from the coded wavelet coefficients to local registers
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::readSubbandsIteration(int* data, int DSizeCurrentX, T* TData, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int* index) {

	TData[((*index) << 1) + 1 + (SHARED_MEMORY_STRIDE*threadIdx.x)] = data[*TCoordinateHL];

	++(*index);

	TData[((*index) << 1) + (SHARED_MEMORY_STRIDE*threadIdx.x)] = data[*TCoordinateLH];
	TData[((*index) << 1) + 1 + (SHARED_MEMORY_STRIDE*threadIdx.x)] = data[*TCoordinateLH + (DSizeCurrentX >> 1)];
}

//Reads the information from the coded wavelet coefficients to local registers. Only LL subband.
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::readSubbands(int* data, int DSizeCurrentX, int DSizeInitialX, T* TData, int TDSizeY, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH)
{
	for (int y = 0; y < TDSizeY; y++) {
		TData[(y << 1) + (SHARED_MEMORY_STRIDE*threadIdx.x)] = data[*TCoordinateLL];

		readSubbandsIteration(data, DSizeCurrentX, TData, TCoordinateLL, TCoordinateHL, TCoordinateLH, &y);
		updateSubbandsCoordinates(DSizeInitialX, TCoordinateLL, TCoordinateHL, TCoordinateLH);
	}
}

//Reads the information from the coded wavelet coefficients to local registers. Only LL subband.
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::readSubbandsLLAux(int* data, T* dataLL, int DSizeCurrentX, int DSizeInitialX, T* TData, int TDSizeY, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH)
{
	for (int y = 0; y < TDSizeY; y++) {
		TData[(y << 1) + (SHARED_MEMORY_STRIDE*threadIdx.x)] = dataLL[*TCoordinateLL];

		readSubbandsIteration(data, DSizeCurrentX, TData, TCoordinateLL, TCoordinateHL, TCoordinateLH, &y);
		updateSubbandsCoordinatesLLAux(DSizeCurrentX, DSizeInitialX, TCoordinateLL, TCoordinateHL, TCoordinateLH);
	}
}

//Reads the information from the coded wavelet coefficients to local registers. Lossy mode, includes dequantization.
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::readSubbandsIterationLossy(int* data, int DSizeCurrentX, T* TData, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int* index, int wLevel, float qSize, float reconstructionFactor) {

	TData[((*index) << 1) + 1 + (SHARED_MEMORY_STRIDE*threadIdx.x)] = 0;
	if (data[*TCoordinateHL] != 0)
		TData[((*index) << 1) + 1 + (SHARED_MEMORY_STRIDE*threadIdx.x)] = ((fabsf(((float)(data[*TCoordinateHL]))) + reconstructionFactor) * ((~(((int)(data[*TCoordinateHL])) >> 31 & 1) + 1) | 1 ) ) / _DWaveletQSteps[wLevel][1] / qSize;

	++(*index);
	TData[((*index) << 1) + (SHARED_MEMORY_STRIDE*threadIdx.x)] = 0;
	if (data[*TCoordinateLH] != 0)
		TData[((*index) << 1) + (SHARED_MEMORY_STRIDE*threadIdx.x)] = ((fabsf(((float)(data[*TCoordinateLH]))) + reconstructionFactor) * ((~(((int)(data[*TCoordinateLH])) >> 31 & 1) + 1) | 1)) / _DWaveletQSteps[wLevel][2] / qSize;
	
	TData[((*index) << 1) + 1 + (SHARED_MEMORY_STRIDE*threadIdx.x)] = 0;
	if (data[*TCoordinateLH + (DSizeCurrentX >> 1)] != 0)
		TData[((*index) << 1) + 1 + (SHARED_MEMORY_STRIDE*threadIdx.x)] = ((fabsf(((float)(data[*TCoordinateLH + (DSizeCurrentX >> 1)]))) + reconstructionFactor) * ((~(((int)(data[*TCoordinateLH + (DSizeCurrentX >> 1)])) >> 31 & 1) + 1) | 1)) / _DWaveletQSteps[wLevel][3] / qSize;
}

//Reads the information from the coded wavelet coefficients to local registers. Lossy mode, includes dequantization, only LL subband.
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::readSubbandsLossy(int* data, int DSizeCurrentX, int DSizeInitialX, T* TData, int TDSizeY, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int wLevel, float qSize, float reconstructionFactor)
{
	int sign = 0;
	for (int y = 0; y < TDSizeY; y++) {
		TData[(y << 1) + (SHARED_MEMORY_STRIDE*threadIdx.x)] = 0;
		if (data[*TCoordinateLL] != 0)
			TData[(y << 1) + (SHARED_MEMORY_STRIDE*threadIdx.x)] = ((fabsf(((float)(data[*TCoordinateLL]))) + reconstructionFactor) * ((~(((int)(data[*TCoordinateLL])) >> 31 & 1) + 1) | 1)) / _DWaveletQSteps[wLevel][0] / qSize;

		readSubbandsIterationLossy(data, DSizeCurrentX, TData, TCoordinateLL, TCoordinateHL, TCoordinateLH, &y, wLevel, qSize, reconstructionFactor);
		updateSubbandsCoordinates(DSizeInitialX, TCoordinateLL, TCoordinateHL, TCoordinateLH);
	}
}

//Reads the information from the coded wavelet coefficients to local registers.
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::readSubbandsLLAuxLossy(int* data, T* dataLL, int DSizeCurrentX, int DSizeInitialX, T* TData, int TDSizeY, int* TCoordinateLL, int* TCoordinateHL, int* TCoordinateLH, int wLevel, float qSize, float reconstructionFactor)
{
	for (int y = 0; y < TDSizeY; y++) {
		TData[(y << 1) + (SHARED_MEMORY_STRIDE*threadIdx.x)] = (float)(dataLL[*TCoordinateLL]);
		readSubbandsIterationLossy(data, DSizeCurrentX, TData, TCoordinateLL, TCoordinateHL, TCoordinateLH, &y, wLevel, qSize, reconstructionFactor);
		updateSubbandsCoordinatesLLAux(DSizeCurrentX, DSizeInitialX, TCoordinateLL, TCoordinateHL, TCoordinateLH);
	}
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeBlockInt1(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int* index)
{
	data[*TCoordinate] = TData[((*index) << 1) + (SHARED_MEMORY_STRIDE*threadIdx.x)];
	data[*TCoordinate + 1] = TData[(((*index) << 1) + 1) + (SHARED_MEMORY_STRIDE*threadIdx.x)];

	*TCoordinate += DSizeCurrentX;
}

//Writes the information back to global memory. Lossy mode.
template<class T, class Y>
__device__ __forceinline__ void DWTEngine<T, Y>::STTwoLossy(Y* a, T b, T c)
{
	asm("st.global.wt.v2.f32 [%0], {%1,%2};" :: "l"((float2*)a), "f"((float)b), "f"((float)c));
}

//Writes the information back to global memory. Lossless mode.
template<class T, class Y>
__device__ __forceinline__ void DWTEngine<T, Y>::STTwo(Y* a, T b, T c)
{
	asm("st.global.wt.v2.u32 [%0], {%1,%2};" :: "l"((int2*)a), "r"((int)b), "r"((int)c));
}

//Sends the proper information to the function which transfers the data to the global memory. Lossy mode.
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeBlockInt2Lossy(Y* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int* index)
{
	STTwoLossy(data + (*TCoordinate), TData[((*index) * 2) + (SHARED_MEMORY_STRIDE*threadIdx.x)], TData[((*index) * 2) + 1 + (SHARED_MEMORY_STRIDE*threadIdx.x)]);
	*TCoordinate += DSizeCurrentX;
}

//Sends the proper information to the function which transfers the data to the global memory. Lossless mode.
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeBlockInt2(Y* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int* index)
{
	int Test = *TCoordinate;
	STTwo(data + (*TCoordinate), TData[((*index) * 2) + (SHARED_MEMORY_STRIDE*threadIdx.x)], TData[((*index) * 2) + 1 + (SHARED_MEMORY_STRIDE*threadIdx.x)]);
	*TCoordinate += DSizeCurrentX;

}


template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeBlockLossy(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int* index)
{
	writeBlockInt2Lossy((Y*)data, DSizeCurrentX, TData, TDSizeY, TCoordinate, index);
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeBlock(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int* index)
{
	writeBlockInt2((Y*)data, DSizeCurrentX, TData, TDSizeY, TCoordinate, index);
}

/*

The following functions are management functions to run through the coded/decoded data and pass through that information
to the functions responsible of writing the data back to the global memory.

*/

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeBlockTopLossy(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int overlap)
{
	for (int y = 0; y < (TDSizeY - (overlap >> 1)); y++)
		writeBlockLossy(data, DSizeCurrentX, TData, TDSizeY, TCoordinate, &y);
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeBlockTop(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int overlap)
{
	for (int y = 0; y < (TDSizeY - (overlap >> 1)); y++)
		writeBlock(data, DSizeCurrentX, TData, TDSizeY, TCoordinate, &y);
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeBlockMiddleLossy(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int overlap)
{
	for (int y = 0; y < (overlap >> 1); y++)
		*TCoordinate += DSizeCurrentX;

	for (int y = (overlap >> 1); y < (TDSizeY - (overlap >> 1)); y++)
		writeBlockLossy(data, DSizeCurrentX, TData, TDSizeY, TCoordinate, &y);
}
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeBlockMiddle(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int overlap)
{
	for (int y = 0; y < (overlap >> 1); y++)
		*TCoordinate += DSizeCurrentX;

	for (int y = (overlap >> 1); y < (TDSizeY - (overlap >> 1)); y++)
		writeBlock(data, DSizeCurrentX, TData, TDSizeY, TCoordinate, &y);
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeBlockBottomLossy(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int overlap)
{
	for (int y = 0; y < (overlap >> 1); y++)
		*TCoordinate += DSizeCurrentX;

	for (int y = (overlap >> 1); y < TDSizeY; y++)
		writeBlockLossy(data, DSizeCurrentX, TData, TDSizeY, TCoordinate, &y);
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeBlockBottom(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int overlap)
{
	for (int y = 0; y < (overlap >> 1); y++)
		*TCoordinate += DSizeCurrentX;

	for (int y = (overlap >> 1); y < TDSizeY; y++)
		writeBlock(data, DSizeCurrentX, TData, TDSizeY, TCoordinate, &y);
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeBlockSchedulerLossy(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int incorrectVerticalTop, int incorrectVerticalBottom, int overlap)
{
	*(TCoordinate) >>= 1;

	if (incorrectVerticalTop == 0)		writeBlockTopLossy(data, DSizeCurrentX / 2, TData, TDSizeY, TCoordinate, overlap);
	else if (incorrectVerticalBottom == 0)		writeBlockBottomLossy(data, DSizeCurrentX / 2, TData, TDSizeY, TCoordinate, overlap);
	else											writeBlockMiddleLossy(data, DSizeCurrentX / 2, TData, TDSizeY, TCoordinate, overlap);

	*(TCoordinate) <<= 1;
}

template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::writeBlockScheduler(T* data, int DSizeCurrentX, T* TData, int TDSizeY, int* TCoordinate, int incorrectVerticalTop, int incorrectVerticalBottom, int overlap)
{
	*(TCoordinate) >>= 1;

	if (incorrectVerticalTop == 0)		writeBlockTop(data, DSizeCurrentX / 2, TData, TDSizeY, TCoordinate, overlap);
	else if (incorrectVerticalBottom == 0)		writeBlockBottom(data, DSizeCurrentX / 2, TData, TDSizeY, TCoordinate, overlap);
	else											writeBlockMiddle(data, DSizeCurrentX / 2, TData, TDSizeY, TCoordinate, overlap);

	*(TCoordinate) <<= 1;
}

//END - <DEVICE> DATA MANAGEMENT FUNCTIONS -----------------------------------------------------------------------

//START - <DEVICE> PRE-COMPUTE FUNCTIONS -----------------------------------------------------------------------

//Assign a data block to a warp (compute the coordinates from where the warp will fetch its data)
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::initializeCoordinates(int DSizeCurrentX, int DSizeInitialX, int DSizeCurrentY, int* TCoordinateX, int* TCoordinateY, int* TCoordinate, int* TCoordinateLL,
	int* TCoordinateHL, int* TCoordinateLH, int laneID, int warpID, int nWarpsX, int nWarpsY, int warpWorkY, int LLOffset,
	int specialLevel, int overlap)
{
	int XEffectiveWork = ((WARPSIZE * NELEMENTS_THREAD_X) - overlap);
	int YEffectiveWork = warpWorkY;

	int XBorderCoordinateCorrection = (((warpID + 1) % nWarpsX) == 0) ? 1 : 0;
	int YBorderCoordinateCorrection = (warpID> ((nWarpsY*nWarpsX) - nWarpsX - 1)) ? 1 : 0;

	*TCoordinateX = (((warpID % nWarpsX) * XEffectiveWork) + (laneID * NELEMENTS_THREAD_X));

	if (XBorderCoordinateCorrection) 		*TCoordinateX -= (XEffectiveWork - (DSizeCurrentX % XEffectiveWork)) % XEffectiveWork + overlap;

	*TCoordinateY = ((warpID / nWarpsX) * (YEffectiveWork));

	if (YBorderCoordinateCorrection)		*TCoordinateY -= (YEffectiveWork - ((DSizeCurrentY - overlap) % YEffectiveWork)) % YEffectiveWork;

	*TCoordinate = DSizeCurrentX*(*TCoordinateY) + *TCoordinateX;

	if (specialLevel == 1) 	*TCoordinateLL = ((*TCoordinateY >> 1)*DSizeInitialX) + (*TCoordinateX >> 1);
	else					*TCoordinateLL = ((*TCoordinateY >> 1)*(DSizeCurrentX >> 1)) + (*TCoordinateX >> 1) + LLOffset;

	*TCoordinateHL = ((*TCoordinateY >> 1)*DSizeInitialX) + (*TCoordinateX >> 1) + (DSizeCurrentX >> 1);
	*TCoordinateLH = (((*TCoordinateY >> 1) + (DSizeCurrentY >> 1))*DSizeInitialX) + (*TCoordinateX >> 1);

}


//With some image and data block sizes some warps can be assigned to data blocks beyond the image borders. This function check if this happens, and its output will be used in the time to write back the results of the DWT
template<class T, class Y>
inline __device__ void DWTEngine<T, Y>::incorrectBorderValues(int laneID, int warpID, int nWarpsX, int nWarpsY, int* incorrectHorizontal, int* incorrectVerticalTop, int* incorrectVerticalBottom, int overlap)
{
	if ((((warpID % nWarpsX) != 0) && (laneID <((overlap >> 1) / NELEMENTS_THREAD_X))) ||
		(((warpID + 1) % nWarpsX) != 0) && (laneID >(WARPSIZE - 1 - ((overlap >> 1) / NELEMENTS_THREAD_X))))

		*incorrectHorizontal = 1;

	if (warpID > (nWarpsX - 1))

		*incorrectVerticalTop = 1;

	if (warpID < (nWarpsX*(nWarpsY - 1)))

		*incorrectVerticalBottom = 1;
}

//END - <DEVICE> PRE-COMPUTE FUNCTIONS -----------------------------------------------------------------------


//START - <DEVICE> CUDA KERNELS -----------------------------------------------------------------------

//CUDA KERNEL that computes the forward DWT over an input image. Lossy mode.
template<class T, class Y>
__global__ void kernelDWTForwardLossy(
	T* deviceOriginalImage,
	T* deviceResultImage,
	int DSizeCurrentX,
	int DSizeInitialX,
	int DSizeCurrentY,
	int	nWarpsX,
	int	nWarpsY,
	int warpWorkY,
	int nWarpsBlock,
	int writeLLOffset,
	int lastLevel,
	int write,
	int wLevel,
	float qSize,
	DWTEngine<T, Y>* DWTEng
)
{
	extern __shared__ int syntheticSharedMemory[];
	//Only register mode.
	register T TData[NELEMENTS_THREAD_Y*NELEMENTS_THREAD_X];


	int laneID = threadIdx.x & 0x1f;
	int warpID = (((threadIdx.x >> 5) + (blockIdx.x * nWarpsBlock)));
	int idleWarp = 0;
	int TCoordinateX = 0;
	int TCoordinateY = 0;
	int TCoordinate = 0;
	int TCoordinateLL = 0;
	int TCoordinateHL = 0;
	int TCoordinateLH = 0;

	int incorrectHorizontal = 0;
	int incorrectVerticalTop = 0;
	int incorrectVerticalBottom = 0;
	int overlap = OVERLAP_LOSSY;


	idleWarp = (warpID < (nWarpsX*nWarpsY)) ? 0 : 1;

	if (idleWarp) return;

	DWTEng->initializeCoordinates(DSizeCurrentX, DSizeInitialX, DSizeCurrentY, &TCoordinateX, &TCoordinateY, &TCoordinate, &TCoordinateLL, &TCoordinateHL, &TCoordinateLH,
		laneID, warpID, nWarpsX, nWarpsY, warpWorkY, writeLLOffset, lastLevel, overlap);

	DWTEng->incorrectBorderValues(laneID, warpID, nWarpsX, nWarpsY, &incorrectHorizontal, &incorrectVerticalTop, &incorrectVerticalBottom, overlap);

		
	DWTEng->readBlock2((Y*)deviceOriginalImage, DSizeCurrentX / 2, TData, NELEMENTS_THREAD_Y, &TCoordinate);


	DWTEng->verticalFilterForwardLossy(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
	DWTEng->horizontalFilterForwardLossyShuffle(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);

	if (write)
		if (incorrectHorizontal == 0)
			DWTEng->writeSubbandsScheduler(deviceResultImage, DSizeCurrentX, DSizeInitialX, TData, NELEMENTS_THREAD_Y, &TCoordinateLL, &TCoordinateHL, &TCoordinateLH, lastLevel, incorrectVerticalTop, incorrectVerticalBottom, overlap, wLevel, 1, qSize);

}

// DEPRECATED
//CUDA KERNEL that computes the forward DWT over an input image. Lossy mode. It is special because the first iteration the information comes coded in chars. 
template<class T, class Y>
__global__ void kernelDWTForwardLossyChar(
	unsigned char* deviceOriginalImage,
	T* deviceResultImage,
	int DSizeCurrentX,
	int DSizeInitialX,
	int DSizeCurrentY,
	int	nWarpsX,
	int	nWarpsY,
	int warpWorkY,
	int nWarpsBlock,
	int writeLLOffset,
	int lastLevel,
	int write,
	int wLevel,
	float qSize,
	DWTEngine<T, Y>* DWTEng
)
{
	extern __shared__ int syntheticSharedMemory[];
	//Only register mode.
	register T TData[NELEMENTS_THREAD_Y*NELEMENTS_THREAD_X];


	int laneID = threadIdx.x & 0x1f;
	int warpID = (((threadIdx.x >> 5) + (blockIdx.x * nWarpsBlock)));
	int idleWarp = 0;
	int TCoordinateX = 0;
	int TCoordinateY = 0;
	int TCoordinate = 0;
	int TCoordinateLL = 0;
	int TCoordinateHL = 0;
	int TCoordinateLH = 0;

	int incorrectHorizontal = 0;
	int incorrectVerticalTop = 0;
	int incorrectVerticalBottom = 0;
	int overlap = OVERLAP_LOSSY;

	idleWarp = (warpID < (nWarpsX*nWarpsY)) ? 0 : 1;

	if (idleWarp) return;

	DWTEng->initializeCoordinates(DSizeCurrentX, DSizeInitialX, DSizeCurrentY, &TCoordinateX, &TCoordinateY, &TCoordinate, &TCoordinateLL, &TCoordinateHL, &TCoordinateLH,
		laneID, warpID, nWarpsX, nWarpsY, warpWorkY, writeLLOffset, lastLevel, overlap);

	DWTEng->incorrectBorderValues(laneID, warpID, nWarpsX, nWarpsY, &incorrectHorizontal, &incorrectVerticalTop, &incorrectVerticalBottom, overlap);

	DWTEng->readBlock2Char((uchar2*)deviceOriginalImage, DSizeCurrentX / 2, TData, NELEMENTS_THREAD_Y, &TCoordinate);


	DWTEng->verticalFilterForwardLossy(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
	DWTEng->horizontalFilterForwardLossyShuffle(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);

	if (write)
		if (incorrectHorizontal == 0)
			DWTEng->writeSubbandsScheduler(deviceResultImage, DSizeCurrentX, DSizeInitialX, TData, NELEMENTS_THREAD_Y, &TCoordinateLL, &TCoordinateHL, &TCoordinateLH, lastLevel, incorrectVerticalTop, incorrectVerticalBottom, overlap, wLevel, 1, qSize);

}

//CUDA KERNEL that computes the forward DWT over an input image. Lossless mode.
template<class T, class Y>
__global__ void kernelDWTForward(
	T* deviceOriginalImage,
	T* deviceResultImage,
	int DSizeCurrentX,
	int DSizeInitialX,
	int DSizeCurrentY,
	int	nWarpsX,
	int	nWarpsY,
	int warpWorkY,
	int nWarpsBlock,
	int writeLLOffset,
	int lastLevel,
	int write,
	DWTEngine<T, Y>* DWTEng
)
{
	extern __shared__ int syntheticSharedMemory[];
	//Only register mode.
	register T TData[NELEMENTS_THREAD_Y*NELEMENTS_THREAD_X];


	int laneID = threadIdx.x & 0x1f;
	int warpID = (((threadIdx.x >> 5) + (blockIdx.x * nWarpsBlock)));
	int idleWarp = 0;
	int TCoordinateX = 0;
	int TCoordinateY = 0;
	int TCoordinate = 0;
	int TCoordinateLL = 0;
	int TCoordinateHL = 0;
	int TCoordinateLH = 0;

	int incorrectHorizontal = 0;
	int incorrectVerticalTop = 0;
	int incorrectVerticalBottom = 0;
	int overlap = OVERLAP_LOSSLESS;

	idleWarp = (warpID < (nWarpsX*nWarpsY)) ? 0 : 1;

	if (idleWarp) return;

	DWTEng->initializeCoordinates(DSizeCurrentX, DSizeInitialX, DSizeCurrentY, &TCoordinateX, &TCoordinateY, &TCoordinate, &TCoordinateLL, &TCoordinateHL, &TCoordinateLH,
		laneID, warpID, nWarpsX, nWarpsY, warpWorkY, writeLLOffset, lastLevel, overlap);

	DWTEng->incorrectBorderValues(laneID, warpID, nWarpsX, nWarpsY, &incorrectHorizontal, &incorrectVerticalTop, &incorrectVerticalBottom, overlap);


	DWTEng->readBlock2((Y*)deviceOriginalImage, DSizeCurrentX / 2, TData, NELEMENTS_THREAD_Y, &TCoordinate);


	DWTEng->verticalFilterForward(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
	DWTEng->horizontalFilterForwardShuffle(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);



	if (write)
		if (incorrectHorizontal == 0)
			DWTEng->writeSubbandsScheduler(deviceResultImage, DSizeCurrentX, DSizeInitialX, TData, NELEMENTS_THREAD_Y, &TCoordinateLL, &TCoordinateHL, &TCoordinateLH, lastLevel, incorrectVerticalTop, incorrectVerticalBottom, overlap, 0, 0, 1);

}

// DEPRECATED
//CUDA KERNEL that computes the forward DWT over an input image. Lossless mode. It is special because the first iteration the information comes coded in chars. 
template<class T, class Y>
__global__ void kernelDWTForwardChar(
	unsigned char* deviceOriginalImage,
	T* deviceResultImage,
	int DSizeCurrentX,
	int DSizeInitialX,
	int DSizeCurrentY,
	int	nWarpsX,
	int	nWarpsY,
	int warpWorkY,
	int nWarpsBlock,
	int writeLLOffset,
	int lastLevel,
	int write,
	DWTEngine<T, Y>* DWTEng
)
{
	extern __shared__ int syntheticSharedMemory[];
	//Only register mode.
	register T TData[NELEMENTS_THREAD_Y*NELEMENTS_THREAD_X];


	int laneID = threadIdx.x & 0x1f;
	int warpID = (((threadIdx.x >> 5) + (blockIdx.x * nWarpsBlock)));
	int idleWarp = 0;
	int TCoordinateX = 0;
	int TCoordinateY = 0;
	int TCoordinate = 0;
	int TCoordinateLL = 0;
	int TCoordinateHL = 0;
	int TCoordinateLH = 0;

	int incorrectHorizontal = 0;
	int incorrectVerticalTop = 0;
	int incorrectVerticalBottom = 0;
	int overlap = OVERLAP_LOSSLESS;

	idleWarp = (warpID < (nWarpsX*nWarpsY)) ? 0 : 1;

	if (idleWarp) return;

	DWTEng->initializeCoordinates(DSizeCurrentX, DSizeInitialX, DSizeCurrentY, &TCoordinateX, &TCoordinateY, &TCoordinate, &TCoordinateLL, &TCoordinateHL, &TCoordinateLH,
		laneID, warpID, nWarpsX, nWarpsY, warpWorkY, writeLLOffset, lastLevel, overlap);

	DWTEng->incorrectBorderValues(laneID, warpID, nWarpsX, nWarpsY, &incorrectHorizontal, &incorrectVerticalTop, &incorrectVerticalBottom, overlap);

	DWTEng->readBlock2Char((uchar2*)deviceOriginalImage, DSizeCurrentX / 2, TData, NELEMENTS_THREAD_Y, &TCoordinate);


	DWTEng->verticalFilterForward(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
	DWTEng->horizontalFilterForwardShuffle(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);



	if (write)
		if (incorrectHorizontal == 0)
			DWTEng->writeSubbandsScheduler(deviceResultImage, DSizeCurrentX, DSizeInitialX, TData, NELEMENTS_THREAD_Y, &TCoordinateLL, &TCoordinateHL, &TCoordinateLH, lastLevel, incorrectVerticalTop, incorrectVerticalBottom, overlap, 0, 0, 1);

}

//CUDA KERNEL that computes the reverse DWT over an input image. Lossy mode.
template<class T, class Y>
__global__ void kernelDWTReverseLossy(
	int* deviceOriginalImage,
	T* deviceResultImage,
	int DSizeCurrentX,
	int DSizeInitialX,
	int DSizeCurrentY,
	int	nWarpsX,
	int	nWarpsY,
	int warpWorkY,
	int nWarpsBlock,
	int readLLOffset,
	int writeOffset,
	int firstLevel,
	int write,
	int wLevel,
	float qSize,
	DWTEngine<T, Y>* DWTEng
)
{

	extern __shared__ int syntheticSharedMemory[];

	register T TData[NELEMENTS_THREAD_Y*NELEMENTS_THREAD_X];



	int laneID = threadIdx.x & 0x1f;
	int warpID = (((threadIdx.x >> 5) + (blockIdx.x * nWarpsBlock)));
	if ((laneID == 0) && (warpID == 0) && (threadIdx.x == 0))
		laneID = 0;
	int TCoordinateX = 0;
	int TCoordinateY = 0;
	int TCoordinate = 0;
	int TCoordinateLL = 0;
	int TCoordinateHL = 0;
	int TCoordinateLH = 0;
	int overlap = OVERLAP_LOSSY;

	int incorrectHorizontal = 0;
	int incorrectVerticalTop = 0;
	int incorrectVerticalBottom = 0;
	int idleWarp = (warpID < (nWarpsX*nWarpsY)) ? 0 : 1;

	if (idleWarp) return;

	DWTEng->initializeCoordinates(DSizeCurrentX, DSizeInitialX, DSizeCurrentY, &TCoordinateX, &TCoordinateY, &TCoordinate, &TCoordinateLL, &TCoordinateHL, &TCoordinateLH,
		laneID, warpID, nWarpsX, nWarpsY, warpWorkY, readLLOffset, firstLevel, overlap);

	DWTEng->incorrectBorderValues(laneID, warpID, nWarpsX, nWarpsY, &incorrectHorizontal, &incorrectVerticalTop, &incorrectVerticalBottom, overlap);

	if (firstLevel)		DWTEng->readSubbandsLossy(deviceOriginalImage, DSizeCurrentX, DSizeInitialX, TData, NELEMENTS_THREAD_Y, &TCoordinateLL, &TCoordinateHL, &TCoordinateLH, wLevel, qSize, 0.5f);
	else 				DWTEng->readSubbandsLLAuxLossy(deviceOriginalImage, deviceResultImage - writeOffset, DSizeCurrentX, DSizeInitialX, TData, NELEMENTS_THREAD_Y, &TCoordinateLL, &TCoordinateHL, &TCoordinateLH, wLevel, qSize, 0.5f);



	DWTEng->horizontalFilterReverseLossyShuffle(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
	DWTEng->verticalFilterReverseLossy(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);


	if (write)
		if (incorrectHorizontal == 0)
			DWTEng->writeBlockSchedulerLossy(deviceResultImage, DSizeCurrentX, TData, NELEMENTS_THREAD_Y, &TCoordinate, incorrectVerticalTop, incorrectVerticalBottom, overlap);
}

//CUDA KERNEL that computes the reverse DWT over an input image. Lossless mode.
template<class T, class Y>
__global__ void kernelDWTReverse(
	int* deviceOriginalImage,
	T* deviceResultImage,
	int DSizeCurrentX,
	int DSizeInitialX,
	int DSizeCurrentY,
	int	nWarpsX,
	int	nWarpsY,
	int warpWorkY,
	int nWarpsBlock,
	int readLLOffset,
	int writeOffset,
	int firstLevel,
	int write,
	DWTEngine<T, Y>* DWTEng
)
{

	extern __shared__ int syntheticSharedMemory[];

	register T TData[NELEMENTS_THREAD_Y*NELEMENTS_THREAD_X];

	int laneID = threadIdx.x & 0x1f;
	int warpID = (((threadIdx.x >> 5) + (blockIdx.x * nWarpsBlock)));
	int TCoordinateX = 0;
	int TCoordinateY = 0;
	int TCoordinate = 0;
	int TCoordinateLL = 0;
	int TCoordinateHL = 0;
	int TCoordinateLH = 0;
	int overlap = OVERLAP_LOSSLESS;

	int incorrectHorizontal = 0;
	int incorrectVerticalTop = 0;
	int incorrectVerticalBottom = 0;
	int idleWarp = (warpID < (nWarpsX*nWarpsY)) ? 0 : 1;

	if (idleWarp) return;

	DWTEng->initializeCoordinates(DSizeCurrentX, DSizeInitialX, DSizeCurrentY, &TCoordinateX, &TCoordinateY, &TCoordinate, &TCoordinateLL, &TCoordinateHL, &TCoordinateLH,
		laneID, warpID, nWarpsX, nWarpsY, warpWorkY, readLLOffset, firstLevel, overlap);

	DWTEng->incorrectBorderValues(laneID, warpID, nWarpsX, nWarpsY, &incorrectHorizontal, &incorrectVerticalTop, &incorrectVerticalBottom, overlap);

	if (firstLevel)		DWTEng->readSubbands(deviceOriginalImage, DSizeCurrentX, DSizeInitialX, TData, NELEMENTS_THREAD_Y, &TCoordinateLL, &TCoordinateHL, &TCoordinateLH);
	else 				DWTEng->readSubbandsLLAux(deviceOriginalImage, deviceResultImage - writeOffset, DSizeCurrentX, DSizeInitialX, TData, NELEMENTS_THREAD_Y, &TCoordinateLL, &TCoordinateHL, &TCoordinateLH);


	DWTEng->horizontalFilterReverseShuffle(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);	
	DWTEng->verticalFilterReverse(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);



	if (write)
		if (incorrectHorizontal == 0)
			DWTEng->writeBlockScheduler(deviceResultImage, DSizeCurrentX, TData, NELEMENTS_THREAD_Y, &TCoordinate, incorrectVerticalTop, incorrectVerticalBottom, overlap);
}

//END - <DEVICE> CUDA KERNELS -----------------------------------------------------------------------






//START - <HOST> DWT FUNCTIONS -----------------------------------------------------------------------

/*
DEPRECATED
Function controlled by the host which manages the amount of wavelet levels that must be executed, calling the kernel
as many times as needed, providing the proper pointers to the input/output memory structures.
*/
template<class T, class Y>
inline void DWTEngine<T, Y>::DWTForwardChar(unsigned char* DDataInitial, T* DDataFinal, cudaStream_t mainStream) 
{

	int 		warpsRow, warpsColumn, warpVerticalLengthWork, CUDABlocksNumber, CUDAWarpsAmount;
	int 		writeLLOffset = 0;
	int			lastLevel = 0;
	int 		DSizeCurrentX = _img->getAdaptedWidth();
	int 		DSizeCurrentY = _img->getAdaptedHeight();
	T*	DDataInitial_aux = (T*)DDataInitial;
	int currentLevel;

	//A CUDA kernel is launched for every DWT level
	for (int currentWLevel = _numberOfWaveletLevels; currentWLevel > 0; --currentWLevel)
	{
		warpsRow = (int)ceil(DSizeCurrentX / ((float)(WARPSIZE*NELEMENTS_THREAD_X) - _overLap));
		warpVerticalLengthWork = (NELEMENTS_THREAD_Y)-_overLap;
		warpsColumn = (int)ceil((((DSizeCurrentY - _overLap) / (float)(warpVerticalLengthWork))));
		CUDAWarpsAmount = warpsRow * warpsColumn;
		CUDABlocksNumber = (int)ceil((CUDAWarpsAmount*WARPSIZE) / (float)(NTHREADSBLOCK_DWT_F));

		lastLevel = (currentWLevel == 1) ? 1 : lastLevel;
		writeLLOffset += (DSizeCurrentX*DSizeCurrentY);
		if (_isLossy)
		{
			//Assign and copy the values of te isLossy and the wavelet level.
			currentLevel = _numberOfWaveletLevels - currentWLevel;
			if (currentWLevel == _numberOfWaveletLevels)
			{
				kernelDWTForwardLossyChar<T, Y> << <CUDABlocksNumber, NTHREADSBLOCK_DWT_F, 0, mainStream >> >
					(
						DDataInitial,
						DDataFinal,
						DSizeCurrentX,
						_img->getAdaptedWidth(),
						DSizeCurrentY,
						warpsRow,
						warpsColumn,
						warpVerticalLengthWork,
						NTHREADSBLOCK_DWT_F / WARPSIZE,
						writeLLOffset,
						lastLevel,
						1,
						currentLevel,
						_qSize,
						this
						);
			}
			else
			{
				kernelDWTForwardLossy<T, Y> << <CUDABlocksNumber, NTHREADSBLOCK_DWT_F, 0, mainStream >> >
					(
						DDataInitial_aux,
						DDataFinal,
						DSizeCurrentX,
						_img->getAdaptedWidth(),
						DSizeCurrentY,
						warpsRow,
						warpsColumn,
						warpVerticalLengthWork,
						NTHREADSBLOCK_DWT_F / WARPSIZE,
						writeLLOffset,
						lastLevel,
						1,
						currentLevel,
						_qSize,
						this
						);
			}
		}
		else
		{
			if (currentWLevel == _numberOfWaveletLevels)
			{
				kernelDWTForwardChar<T, Y> << <CUDABlocksNumber, NTHREADSBLOCK_DWT_F, 0, mainStream >> >
					(
						DDataInitial,
						DDataFinal,
						DSizeCurrentX,
						_img->getAdaptedWidth(),
						DSizeCurrentY,
						warpsRow,
						warpsColumn,
						warpVerticalLengthWork,
						NTHREADSBLOCK_DWT_F / WARPSIZE,
						writeLLOffset,
						lastLevel,
						1,
						this
						);
			}
			else
			{
				kernelDWTForward<T, Y> << <CUDABlocksNumber, NTHREADSBLOCK_DWT_F, 0, mainStream >> >
					(
						DDataInitial_aux,
						DDataFinal,
						DSizeCurrentX,
						_img->getAdaptedWidth(),
						DSizeCurrentY,
						warpsRow,
						warpsColumn,
						warpVerticalLengthWork,
						NTHREADSBLOCK_DWT_F / WARPSIZE,
						writeLLOffset,
						lastLevel,
						1,
						this
						);
			}

		}
		cudaStreamSynchronize(mainStream);
		KERNEL_ERROR_HANDLER;
		DDataInitial_aux = DDataFinal + (writeLLOffset);

		DSizeCurrentX >>= 1;
		DSizeCurrentY >>= 1;


	}
}

/*
DWT Forward function controlled by the host which manages the amount of wavelet levels that must be executed, calling the kernel
as many times as needed, providing the proper pointers to the input/output memory structures.
*/
template<class T, class Y>
inline void DWTEngine<T, Y>::DWTForward(T* DDataInitial, T* DDataFinal, cudaStream_t mainStream) {

	int 		warpsRow, warpsColumn, warpVerticalLengthWork, CUDABlocksNumber, CUDAWarpsAmount;
	int 		writeLLOffset = 0;
	int			lastLevel = 0;
	int 		DSizeCurrentX = _img->getAdaptedWidth();
	int 		DSizeCurrentY = _img->getAdaptedHeight();
	T*	DDataInitial_aux = DDataInitial;
	int currentLevel;

	//A CUDA kernel is launched for every DWT level
	for (int currentWLevel = _numberOfWaveletLevels; currentWLevel>0; --currentWLevel)
	{
		warpsRow = (int)ceil(DSizeCurrentX / ((float)(WARPSIZE*NELEMENTS_THREAD_X) - _overLap));
		warpVerticalLengthWork = (NELEMENTS_THREAD_Y)-_overLap;
		warpsColumn = (int)ceil((((DSizeCurrentY - _overLap) / (float)(warpVerticalLengthWork))));
		CUDAWarpsAmount = warpsRow *	warpsColumn;
		CUDABlocksNumber = (int)ceil((CUDAWarpsAmount*WARPSIZE) / (float)(NTHREADSBLOCK_DWT_F));

		lastLevel = (currentWLevel == 1) ? 1 : lastLevel;
		writeLLOffset += (DSizeCurrentX*DSizeCurrentY);
		if (_isLossy)
		{
			//Assign and copy the values of te isLossy and the wavelet level.
			currentLevel = _numberOfWaveletLevels - currentWLevel;
				
			kernelDWTForwardLossy<T, Y> << <CUDABlocksNumber, NTHREADSBLOCK_DWT_F, 0, mainStream >> >
				(
					DDataInitial_aux,
					DDataFinal,
					DSizeCurrentX,
					_img->getAdaptedWidth(),
					DSizeCurrentY,
					warpsRow,
					warpsColumn,
					warpVerticalLengthWork,
					NTHREADSBLOCK_DWT_F / WARPSIZE,
					writeLLOffset,
					lastLevel,
					1,
					currentLevel,
					_qSize,
					this
					);
		}
		else
		{
			kernelDWTForward<T, Y> << <CUDABlocksNumber, NTHREADSBLOCK_DWT_F, 0, mainStream >> >
				(
					DDataInitial_aux,
					DDataFinal,
					DSizeCurrentX,
					_img->getAdaptedWidth(),
					DSizeCurrentY,
					warpsRow,
					warpsColumn,
					warpVerticalLengthWork,
					NTHREADSBLOCK_DWT_F / WARPSIZE,
					writeLLOffset,
					lastLevel,
					1,
					this
					);
		}
		cudaStreamSynchronize(mainStream);
		KERNEL_ERROR_HANDLER;

		DDataInitial_aux = DDataFinal + (writeLLOffset);

		DSizeCurrentX >>= 1;
		DSizeCurrentY >>= 1;
			

	}
}

/*
DWT Reverse function controlled by the host which manages the amount of wavelet levels that must be executed, calling the kernel
as many times as needed, providing the proper pointers to the input/output memory structures.
*/
template<class T, class Y>
inline void DWTEngine<T, Y>::DWTReverse(int* DDataInitial, T* DDataFinal, cudaStream_t mainStream) {

	int 		warpsRow, warpsColumn, warpVerticalLengthWork, CUDABlocksNumber, CUDAWarpsAmount;
	int 		readLLOffset = 0;
	int 		writeOffset = 0;
	int			firstLevel = 1;
	int 		DSizeCurrentX = _img->getAdaptedWidth() >> (_numberOfWaveletLevels - 1);
	int 		DSizeCurrentY = _img->getAdaptedHeight() >> (_numberOfWaveletLevels - 1);
	T*		DDataFinal_aux = DDataFinal;

	//A CUDA kernel is launched for every DWT level
	for (int currentWLevel = _numberOfWaveletLevels; currentWLevel>0; --currentWLevel)
	{
		warpsRow = (int)ceil(DSizeCurrentX / ((float)(WARPSIZE*NELEMENTS_THREAD_X) - _overLap));
		warpVerticalLengthWork = (NELEMENTS_THREAD_Y)-_overLap;
		warpsColumn = (int)ceil(((DSizeCurrentY - _overLap) / (float)(warpVerticalLengthWork)));
		CUDAWarpsAmount = warpsRow *	warpsColumn;
		CUDABlocksNumber = (int)ceil((CUDAWarpsAmount*WARPSIZE) / (float)NTHREADSBLOCK_DWT_R);


		if (_isLossy)
		{
			int level = currentWLevel - 1;
			kernelDWTReverseLossy<T, Y> << <CUDABlocksNumber, NTHREADSBLOCK_DWT_R, 0 , mainStream >> >
				(
					DDataInitial,
					DDataFinal_aux,
					DSizeCurrentX,
					_img->getAdaptedWidth(),
					DSizeCurrentY,
					warpsRow,
					warpsColumn,
					warpVerticalLengthWork,
					NTHREADSBLOCK_DWT_R / WARPSIZE,
					readLLOffset,
					writeOffset,
					firstLevel,
					1,
					level,
					_qSize,
					this
					);
		}
		else
		{
			kernelDWTReverse<T, Y> << <CUDABlocksNumber, NTHREADSBLOCK_DWT_R, 0 , mainStream >> >
				(
					DDataInitial,
					DDataFinal_aux,
					DSizeCurrentX,
					_img->getAdaptedWidth(),
					DSizeCurrentY,
					warpsRow,
					warpsColumn,
					warpVerticalLengthWork,
					NTHREADSBLOCK_DWT_R / WARPSIZE,
					readLLOffset,
					writeOffset,
					firstLevel,
					1,
					this
					);
		}
		cudaStreamSynchronize(mainStream);
		KERNEL_ERROR_HANDLER;

		firstLevel = 0;

		readLLOffset += (writeOffset - readLLOffset);
		writeOffset += (DSizeCurrentX*DSizeCurrentY);
		DDataFinal_aux = DDataFinal + writeOffset;

		DSizeCurrentX <<= 1;
		DSizeCurrentY <<= 1;
	}
}

template<class T, class Y>
inline DWTEngine<T, Y>::DWTEngine(Image *image, int numOfWaveletLevels, int olap, bool lossy, float qs)
{

	//Precompilation stuff that must be replaced. We are not doing fullshared mod
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	
	_img					= image;
	_numberOfWaveletLevels	= numOfWaveletLevels;
	_overLap				= olap;
	_isLossy				= lossy;
	_qSize					= qs;
}
