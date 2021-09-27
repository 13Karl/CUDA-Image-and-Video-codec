#pragma once

#ifndef BITSTREAMBUILDER_HPP
#define BITSTREAMBUILDER_HPP


#include "../Image/Image.hpp"
#include "BitStreamBuilder.cuh"

class BitStreamBuilder
{
public:

	BitStreamBuilder(Image *img, int* data, int cbWidth, int cbHeight, int* sArray, int* pArray, int* DTStorageBytes, bool isL, float qSize, int numBlocks, int cp, int wLevels, int* LUTBSTable, int LUTBSTableSteps, float kFactor);
	BitStreamBuilder(int size, unsigned short* hData, int cbSize, Image* img);
	void createBitStream(int* totalSize, int prefixedArraySize, int cbSize, unsigned short* HExtraInformation, unsigned short* DBitStreamValues, int iter, cudaStream_t mainStream, int numberOfFrames);
	void createCodeStream(int* DPrefixedArray, int* DSizeArray, int* HBasicInformation, int* DTempStoragePArray, unsigned short* DBitStreamValues, int* DCodeStreamValues, int* HSizeArray, int* HTotalBSSize, cudaStream_t mainStream, int HLUTBSTableSteps, int* DLUTBSTable);
	
	void retrieveSizeArray(int* sizeArray, unsigned short* hostData, int amountOfCodeblocks);


private:

	void setExtraInformation(unsigned short* extraInformation, int numberOfFrames);
	
	Image *_image;
	int _DWTCodeBlockWidth;
	int _DWTCodeBlockHeight;
	bool _isLossy;
	float _quantizationSize;
	int _numberOfBlocks;
	int _codingPasses;
	int _waveletLevels;
	int *_DCodeStream;
	int *_prefixedArray;
	int *_sizeArray;
	unsigned short *_HBitStreamOrganized;
	int *_HCodeStreamDecoded;
	int* _DLUTBSTable;
	int _LUTBSStepsRange;
	int _sizeOfStream;
	unsigned short* _hostData;
	int _codeblockSizing;
	int *_codeStreamDecoded;
	int* _extraInformation;
	int _HLUTBSTableSteps;
	int* _DTempStorageBytes;
	float _kFactor;
};

#endif