#pragma once
#ifndef BITSTREAMBUILDER_CUH
#define BITSTREAMBUILDER_CUH

#include "../Image/Image.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef __linux__ 
#include <math.h>
#endif


#ifndef GPU_HANDLE_ERROR
#define GPU_HANDLE_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#endif

#ifndef KERNEL_ERROR_HANDLER
#define KERNEL_ERROR_HANDLER { cudaKernelAssert(__FILE__, __LINE__);}
#endif



class BSEngine
{
public:

	BSEngine(Image *img, unsigned short* bitStreamData, int* codeStreamData, int cbWidth, int cbHeight, int* pArray = NULL, int* sArray = NULL, int* hSArray = NULL);
	void deviceMemoryAllocator(bool isForward, int totalSize, unsigned short* extraInformation, int sizeOfPrefixedArray, int iter, cudaStream_t mainStream);

	void launchKernel(bool isForward, int prefixedArraySize, int cbSize, cudaStream_t mainStream);

	void launchPrefixArrayGeneration(int sizeOfPrefixedArray, int* DTempStorageBytes, int* HTotalBSSize, cudaStream_t mainStream, int cbSize);
	void launchLUTBSGeneration(cudaStream_t mainStream, int HLUTBSTableSteps, int* DLUTBSTable, int sizeOfPrefixedArray);
	void LUTBSGenerator(cudaStream_t mainStream, int sizeOfPrefixedArray);
	void setBitStreamValues(unsigned short* bitStreamData);

	void setCodeStreamValues(int* codeStreamData);

private:

	void gpuAssert(cudaError_t, const char *, int);
	void cudaKernelAssert(const char *, int);

	Image *_image;
	int _codeBlockWidth;
	int _codeBlockHeight;
    unsigned short* _DBitStreamData;
	int* _DCodeStreamData;
	int* _DPrefixedArray;
	int* _DSizeArray;
	int* _HSizeArray;
	int _amountOfCodedValuesWithoutHeader;
	size_t _tempStorageBytes = 0;

	//Special Binary Search LUT Table
	int *_DLUTBSTable;
	int _LUTBSTableSteps;
	int _LUTBSTableStepsRange;
	
};

#endif