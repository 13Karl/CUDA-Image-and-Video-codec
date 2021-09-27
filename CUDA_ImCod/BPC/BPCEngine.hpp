#pragma once
#include "../Image/Image.hpp"
#include "../IO/IOManager.hpp"
#include "BPCEngine.cuh"
#include "../BitStreamBuilder/BitStreamBuilder.hpp"

#ifndef BPCENGINE_HPP
#define BPCENGINE_HPP

template<class T>
class BPCCuda
{
public:

	BPCCuda(Image *img, T* hostData, int wLevels, int cbWidth, int cbHeight, int cp, bool isL, float qSize, float kFactor, int amountOfLUTFiles);
	void Code(int LUTNumberOfBitplanes, int LUTNumberOfSubbands, int LUTContextRefinement, int LUTContextSign, int LUTContextSignificance, int LUTMultPrecision, int* LUTInformation, int* DCodeStreamValues, int* DPrefixedArray, int* DTempStoragePArray, int* DSizeArray, unsigned short* HExtraInformation, unsigned short* DBitStreamValues, int* HTotalBSSize, int* DLUTBSTable, int HLUTBSTableSteps, int iter, cudaStream_t mainStream, int numberOfFrames, double *measurementsBPC);
	void Decode(int size, int LUTNumberOfBitplanes_, int LUTNumberOfSubbands_, int LUTContextRefinement_, int LUTContextSign_, int LUTContextSignificance_, int LUTMultPrecision_, int* LUTInformation_, int* DPrefixedArray, int* DSizeArray, int* HBasicInformation, int* DTempStoragePArray, unsigned short* DBitStreamValues, int* DCodeStreamValues, int* HSizeArray, int* HTotalBSSize, int* DWaveletCoefficients, cudaStream_t mainStream, int HLUTBSTableSteps, int* DLUTBSTable, double *measurementsBPC);

private:

	Image* _image;
	int _numberOfWaveletLevels;
	int _codeBlockWidth;
	int _codeBlockHeight;
	int _codingPasses;
	bool _isLossy;
	float _quantizationSize;
	T* _DInitialData;
	float _kFactor;
	int _amountOfLUTFiles;
};

#include "BPCEngine.ipp"

#endif