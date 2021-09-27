#pragma once

#include "BitStreamBuilder.cuh"
#include <iostream>
#ifndef CUB_DATA
#define CUB_DATA
#include "../cub/cub.cuh"


#endif
void BSEngine::gpuAssert(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
		exit(EXIT_FAILURE);
	}
}

void BSEngine::cudaKernelAssert(const char *file, int line)
{
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		std::cout << "cudaKernelAssert() failed at " << file << ":" << line << ":" << cudaGetErrorString(err) << std::endl;
		exit(-1);
	}

}

/*
Performs the binary search per thread inside each warp to look where the information is located in the origin structure and be able to locate where to place it in the output when performing the copying. Also,
it detects whether the information is to be placed in the header as part of the side information.
*/
__device__ void binarySearchLUTBS(int* prefixedArray, int *bitStreamDataStart, int *codeblockProcessed, int *codeStreamDataStart, int *codeStreamDataRealStart, int *size, int *headerThread, int prefixedArraySize, int threadId, int cbSize, const int* LUTBSTable, int LUTBSStepsRange, int LUTBSTableSteps)
{
	bool found = false;
	register int lowestPosition;
	register int highestPosition;
	register int midPosition;
	register int firstValue;
	register int secondValue;
	register int value;


	register float LUTPositionF = (float)threadId / (float)LUTBSStepsRange;

	register int LUTPosition = (int)LUTPositionF;
	if (LUTPosition == 0)
	{
		lowestPosition = LUTBSTable[LUTPosition];
		highestPosition = LUTBSTable[LUTPosition + 1];
	}
	else if ((LUTPosition > 0) && (LUTPosition < LUTBSTableSteps))
	{
		lowestPosition = LUTBSTable[LUTPosition-1];
		highestPosition = LUTBSTable[LUTPosition+1];
	}
	else
	{
		lowestPosition = LUTBSTable[LUTPosition-1];
		highestPosition = prefixedArraySize - 1;
	}
	midPosition = (lowestPosition + highestPosition) / 2;

	while (midPosition < highestPosition)
	{
		value = threadId - prefixedArray[midPosition];
		if (value >= 0)
		{
			lowestPosition = midPosition + 1;
		}
		else
		{
			highestPosition = midPosition;
		}
		midPosition = (lowestPosition + highestPosition) / 2;
	}

	if ((prefixedArray[highestPosition] <= threadId) && highestPosition + 1 < prefixedArraySize)
		highestPosition++;

	if (highestPosition == 0)
	{
		*bitStreamDataStart = threadId;
		*codeblockProcessed = 0;
		*codeStreamDataStart = 0;
		*size = prefixedArray[0];
		*headerThread = 0;
		*codeStreamDataRealStart = threadId;
	}
	else
	{

		*codeblockProcessed = highestPosition;
		*bitStreamDataStart = threadId - highestPosition;
		*codeStreamDataStart = cbSize * (highestPosition);
		*codeStreamDataRealStart = cbSize * (highestPosition)+threadId - prefixedArray[highestPosition - 1];
		*size = prefixedArray[highestPosition] - prefixedArray[highestPosition - 1];
		*headerThread = prefixedArray[highestPosition - 1];
	}

}

/*
CR function, responsible of relocating the codeblocks, tightening the structure to create an effective, compressed data structure which can be then saved to a file.
*/
__global__ void buildBitStreamLUTBS(int *input, unsigned short *output, int *prefixedArray, int prefixedArraySize, int totalRealData, int cbSize, const int* LUTBSTable, int LUTBSStepsRange, int LUTBSTableSteps)
{

	register int bitStreamDataStart = -1;
	register int codeblockProcessed = -1;
	register int codeStreamDataStart = -1;
	register int codeStreamDataRealStart = -1;
	register int size = -1;
	register int headerThread = -1;

	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadId < totalRealData)
	{
		// Header offset which considers the side information included in the compressed frame.
		int headerOffset = -1 + 9 + prefixedArraySize * 2;
		binarySearchLUTBS(prefixedArray, &bitStreamDataStart, &codeblockProcessed, &codeStreamDataStart, &codeStreamDataRealStart, &size, &headerThread, prefixedArraySize, threadId, cbSize, LUTBSTable, LUTBSStepsRange, LUTBSTableSteps);

		//First, check if we are dealing with header data. If so, copy the header to its corresponding place.
		//Copy values.
		__syncwarp();
		if (threadId == headerThread)
		{
			output[9 + codeblockProcessed * 2] = input[codeStreamDataStart];
			output[9 + codeblockProcessed * 2 + 1] = size;
		}
		else
		{
			output[headerOffset + bitStreamDataStart] = input[codeStreamDataRealStart];
		}
	}

}

/*
Function responsible of reversing the realocation of CR to decode a compressed frame. Its behaviour is a mirror of the previous CR function detailed above.
*/
__global__ void buildCodeStreamLUTBS(unsigned short *input, int *output, int *prefixedArray, int prefixedArraySize, int totalRealData, int cbSize, const int* LUTBSTable, int LUTBSStepsRange, int LUTBSTableSteps)
{
	register int bitStreamDataStart = -1;
	register int codeblockProcessed = -1;
	register int codeStreamDataStart = -1;
	register int codeStreamDataRealStart = -1;
	register int size = -1;
	register int headerThread = -1;


	int threadId = threadIdx.x + blockIdx.x * blockDim.x;

	if (threadId < totalRealData)
	{
		int headerOffset = -1 + 9 + prefixedArraySize * 2;
		binarySearchLUTBS(prefixedArray, &bitStreamDataStart, &codeblockProcessed, &codeStreamDataStart, &codeStreamDataRealStart, &size, &headerThread, prefixedArraySize, threadId, cbSize, LUTBSTable, LUTBSStepsRange, LUTBSTableSteps);

		//First, check if we are dealing with header data. If so, copy the header to its corresponding place.
		//Copy values.
		__syncwarp();
		if (threadId == headerThread)
		{
			output[codeStreamDataRealStart] = input[9 + codeblockProcessed * 2];
		}
		else
		{
			output[codeStreamDataRealStart] = input[headerOffset + bitStreamDataStart];
		}
	}
}

/*
Function responsible of deciding which kernel to launch depending on the mode - coding or decoding. Threadblock sizes were found empirically.
*/
void BSEngine::launchKernel(bool isForward, int prefixedArraySize, int cbSize, cudaStream_t mainStream)
{
	if (isForward)
	{
		int totalBlocks = (int)ceil((_amountOfCodedValuesWithoutHeader / 256));
		buildBitStreamLUTBS <<<totalBlocks + 1, 256, 0, mainStream >> >(_DCodeStreamData, _DBitStreamData, _DPrefixedArray, prefixedArraySize, _amountOfCodedValuesWithoutHeader, cbSize, _DLUTBSTable, _LUTBSTableStepsRange, _LUTBSTableSteps);
		cudaStreamSynchronize(mainStream);	
		KERNEL_ERROR_HANDLER;
	}
	else
	{
		int totalBlocks = (int)ceil(((_amountOfCodedValuesWithoutHeader) / 256));
		buildCodeStreamLUTBS <<<totalBlocks + 1, 256, 0, mainStream >> > (_DBitStreamData, _DCodeStreamData, _DPrefixedArray, prefixedArraySize, _amountOfCodedValuesWithoutHeader, cbSize, _DLUTBSTable, _LUTBSTableStepsRange, _LUTBSTableSteps);
		cudaStreamSynchronize(mainStream);
		KERNEL_ERROR_HANDLER;
	}

}

/*
Binary Search function used to create the LUT table used in the binary search employed in the CR functions. It creates a much smaller list of possible values acting as an index of the real map array used in CR.
*/
__global__ void binarySearchLUTBSB(int* prefixedArray, int valueToSearch, int* LUTBSTable, int prefixedArraySize, int numberOfThreads)
{
	bool found = false;
	register int lowestPosition = 0;
	register int highestPosition = prefixedArraySize;
	register int midPosition = (lowestPosition + highestPosition) / 2;
	register int firstValue;
	register int secondValue;
	register int value;

	int threadId = blockIdx.x * blockDim.x + threadIdx.x + 1;
	valueToSearch = valueToSearch*threadId;
	LUTBSTable[0] = 0;
	if (threadId <= numberOfThreads)
	{
		while (midPosition < highestPosition)
		{
			value = valueToSearch - prefixedArray[midPosition];
			if (value >= 0)
			{
				lowestPosition = midPosition + 1;
			}
			else
			{
				highestPosition = midPosition;
			}
			midPosition = (lowestPosition + highestPosition) / 2;
		}
		__syncwarp();
		LUTBSTable[threadId] = midPosition;
	}
}

/*
Function responsible of calling the LUT Kernel, which generates the LUT table that acts as an index, effectively increasing the throughput of any of the CR functions, either in coding or decoding mode.
*/
void BSEngine::LUTBSGenerator(cudaStream_t mainStream, int sizeOfPrefixedArray)
{
	//1.- Memory Allocation of the special LUT Table (Depends on the sizing we want to include).
	//2.- Divide the maximum amount of data (amountOfValuesInCodedImages) by the size of the LUT Table.
	//3.- Fill the table with values - binary search of 1 thread - 1 position.

	int blocks = 1;
	int threadsPerBlock = _LUTBSTableSteps;
	if (_LUTBSTableSteps > 1024)
	{
		blocks = (_LUTBSTableSteps / 1024);
		threadsPerBlock = (_LUTBSTableSteps % 1024);
		if (threadsPerBlock == 0)
			threadsPerBlock = 1024;

	}
	int numberOfThreads = blocks * threadsPerBlock;
	binarySearchLUTBSB << <blocks, threadsPerBlock, 0, mainStream >> > (_DPrefixedArray, _LUTBSTableStepsRange, _DLUTBSTable, sizeOfPrefixedArray, numberOfThreads);
	cudaStreamSynchronize(mainStream);
	KERNEL_ERROR_HANDLER;
}

/*
Function which prepare the variables needed in the LUT generation.
*/
void BSEngine::launchLUTBSGeneration(cudaStream_t mainStream, int HLUTBSTableSteps, int* DLUTBSTable, int sizeOfPrefixedArray)
{
	_DLUTBSTable = DLUTBSTable;
	_LUTBSTableSteps = HLUTBSTableSteps;
	_LUTBSTableStepsRange = (int)ceil((float)_amountOfCodedValuesWithoutHeader / (float)_LUTBSTableSteps);
	LUTBSGenerator(mainStream, sizeOfPrefixedArray);
}

/*
Memory management function which prepares the variables used in the relocating kernels, resetting them to default values as those are reused by every frame processed by a given stream.
*/
void BSEngine::deviceMemoryAllocator(bool isForward, int totalSize, unsigned short* extraInformation, int sizeOfPrefixedArray, int iter, cudaStream_t mainStream)
{

	if (isForward)
	{
		
		GPU_HANDLE_ERROR(cudaMemsetAsync(_DBitStreamData, -1, totalSize * sizeof(unsigned short), mainStream));
		if (iter == 0)
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_DBitStreamData, extraInformation, 9 * sizeof(unsigned short), cudaMemcpyHostToDevice, mainStream));
	}
	else
	{
		GPU_HANDLE_ERROR(cudaMemsetAsync(_DCodeStreamData, -1, _image->getAdaptedHeight() * _image->getAdaptedWidth() * sizeof(int), mainStream));
		GPU_HANDLE_ERROR(cudaMemcpyAsync(_DSizeArray, _HSizeArray, sizeOfPrefixedArray * sizeof(int), cudaMemcpyHostToDevice, mainStream));
	}
}

/*
Function managing the invocation of the prefix array kernel of the CUB framework.
*/
void BSEngine::launchPrefixArrayGeneration(int sizeOfPrefixedArray, int* DTempStorageBytes, int* HTotalBSSize, cudaStream_t mainStream, int cbSize)
{
	if (mainStream != NULL)
	{
		_tempStorageBytes = _image->getAdaptedWidth() * _image->getAdaptedHeight() / (cbSize * 2);
		if (_tempStorageBytes  < 1000)
			_tempStorageBytes = 1000;
		cudaError_t eDos = cub::DeviceScan::InclusiveSum(DTempStorageBytes, _tempStorageBytes, _DSizeArray, _DPrefixedArray, sizeOfPrefixedArray, mainStream);

		
		GPU_HANDLE_ERROR(cudaMemcpyAsync(HTotalBSSize, _DPrefixedArray + sizeOfPrefixedArray - 1, sizeof(int), cudaMemcpyDeviceToHost, mainStream));
		_amountOfCodedValuesWithoutHeader = HTotalBSSize[0];
		//Extra amount in bytes refer to the addition of the header (for both, size and MSB) and the 9 positions taken by the general header info.
		//the size of the prefixed array is equal to the amount of codeblocks coded in the frame/image.
		int extraAmountInBytes = 9 + sizeOfPrefixedArray * 2;
		HTotalBSSize[0] = HTotalBSSize[0] + extraAmountInBytes - sizeOfPrefixedArray + 1;
	}
	else
	{
		_tempStorageBytes = _image->getAdaptedWidth() * _image->getAdaptedHeight() / (cbSize * 2);
		if (_tempStorageBytes  < 1000)
			_tempStorageBytes = 1000;
		cudaError_t eDos = cub::DeviceScan::InclusiveSum(DTempStorageBytes, _tempStorageBytes, _DSizeArray, _DPrefixedArray, sizeOfPrefixedArray);
		
		
		GPU_HANDLE_ERROR(cudaMemcpy(HTotalBSSize, _DPrefixedArray + sizeOfPrefixedArray - 1, sizeof(int), cudaMemcpyDeviceToHost));
		_amountOfCodedValuesWithoutHeader = HTotalBSSize[0];
		//Extra amount in bytes refer to the addition of the header (for both, size and MSB) and the 9 positions taken by the general header info.
		//the size of the prefixed array is equal to the amount of codeblocks coded in the frame/image.
		int extraAmountInBytes = 9 + sizeOfPrefixedArray * 2;
		HTotalBSSize[0] = HTotalBSSize[0] + extraAmountInBytes - sizeOfPrefixedArray + 1;
	}
	
}

BSEngine::BSEngine(Image *img, unsigned short* bitStreamData, int* codeStreamData, int cbWidth, int cbHeight, int* pArray, int* sArray, int* hSArray)
{
	_image = img;
	_codeBlockWidth = cbWidth;
	_codeBlockHeight = cbHeight;
	_DBitStreamData = bitStreamData;
	_DCodeStreamData = codeStreamData;
	if (pArray != NULL)
		_DPrefixedArray = pArray;
	if (sArray != NULL)
		_DSizeArray = sArray;
	if (hSArray != NULL)
		_HSizeArray = hSArray;
}

void BSEngine::setBitStreamValues(unsigned short* bitStreamData)
{
	_DBitStreamData = bitStreamData;
}

void BSEngine::setCodeStreamValues(int* codeStreamData)
{
	_DCodeStreamData = codeStreamData;
}