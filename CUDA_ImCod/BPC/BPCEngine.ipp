#pragma once
#include "BPCEngine.hpp"
#include "../IO/IOManager.hpp"


template<class T>
BPCCuda<T>::BPCCuda(Image *img, T* initialData, int wLevels, int cbWidth, int cbHeight, int cp, bool isL, float qSize, float kFactor, int amountOfLUTFiles)
{
	_image = img;
	_codingPasses = cp;
	_DInitialData = initialData;
	_codeBlockWidth = cbWidth;
	_codeBlockHeight = cbHeight;
	_numberOfWaveletLevels = wLevels;
	_isLossy = isL;
	_quantizationSize = qSize;
	_kFactor = kFactor;
	_amountOfLUTFiles = amountOfLUTFiles;
}

/*
Encoding process, including BPC-PaCo and CR kernels. Intermediate steps needed to reset the output memory regions are taken in consideration to set default values.
*/
template<class T>
void BPCCuda<T>::Code(int LUTNumberOfBitplanes, int LUTNumberOfSubbands, int LUTContextRefinement, int LUTContextSign, int LUTContextSignificance, int LUTMultPrecision, int* LUTInformation, int* DCodeStreamValues, int* DPrefixedArray, int* DTempStoragePArray, int* DSizeArray, unsigned short* HExtraInformation, unsigned short* DBitStreamValues,int* HTotalBSSize, int* DLUTBSTable, int HLUTBSTableSteps, int iter, cudaStream_t mainStream, int numberOfFrames, double *measurementsBPC)
{

	BPCEngine<T>* BPCEng = new BPCEngine<T>(_numberOfWaveletLevels, _codingPasses, _kFactor, _amountOfLUTFiles);
	BPCEng->deviceMemoryAllocator(_image->getAdaptedWidth() * _image->getAdaptedHeight(), &DCodeStreamValues, CODE, mainStream);
	BPCEng->kernelLauncher(CODE, _image->getAdaptedWidth(), _image->getAdaptedHeight(), _DInitialData, DCodeStreamValues, LUTNumberOfBitplanes, LUTNumberOfSubbands, LUTContextRefinement, LUTContextSign, LUTContextSignificance, LUTMultPrecision, LUTInformation, DSizeArray, mainStream, measurementsBPC);
	
	BitStreamBuilder *BSB = new BitStreamBuilder(_image, DCodeStreamValues, _codeBlockWidth, _codeBlockHeight, DSizeArray, DPrefixedArray, DTempStoragePArray, _isLossy, _quantizationSize, BPCEng->getNumberOfCodeblocks(), _codingPasses, _numberOfWaveletLevels, DLUTBSTable, HLUTBSTableSteps, _kFactor);
	BSB->createBitStream(HTotalBSSize, BPCEng->getPrefixedArraySize(), CBLOCK_WIDTH * CBLOCK_LENGTH, HExtraInformation, DBitStreamValues, iter, mainStream, numberOfFrames);
	
	delete BPCEng;
	delete BSB;
}

/*
Decoding process, including BPC-PaCo and CR kernels. Intermediate steps needed to reset the output memory regions are taken in consideration to set default values.
*/
template<class T>
void BPCCuda<T>::Decode(int size, int LUTNumberOfBitplanes_, int LUTNumberOfSubbands_, int LUTContextRefinement_, int LUTContextSign_, int LUTContextSignificance_, int LUTMultPrecision_, int* LUTInformation_, int* DPrefixedArray, int* DSizeArray, int* HBasicInformation, int* DTempStoragePArray, unsigned short* DBitStreamValues, int* DCodeStreamValues, int* HSizeArray, int* HTotalBSSize, int* DWaveletCoefficients, cudaStream_t mainStream, int HLUTBSTableSteps, int* DLUTBSTable, double *measurementsBPC)
{
	int frameSize = _image->getAdaptedHeight()*_image->getAdaptedWidth();
	int codingPasses = HBasicInformation[1];
	int waveletLevels = HBasicInformation[4];

	BitStreamBuilder *BSB = new BitStreamBuilder(size, _DInitialData, CBLOCK_WIDTH * CBLOCK_LENGTH, _image);
	BSB->createCodeStream(DPrefixedArray, DSizeArray, HBasicInformation, DTempStoragePArray, DBitStreamValues, DCodeStreamValues, HSizeArray, HTotalBSSize, mainStream, HLUTBSTableSteps, DLUTBSTable);
	
	BPCEngine<int>* BPCEng = new BPCEngine<int>(waveletLevels, codingPasses, _kFactor, _amountOfLUTFiles);
	BPCEng->deviceMemoryAllocator(frameSize, &DWaveletCoefficients, DECODE, mainStream);
	BPCEng->kernelLauncher(DECODE, _image->getAdaptedWidth(), _image->getAdaptedHeight(), DCodeStreamValues, DWaveletCoefficients, LUTNumberOfBitplanes_, LUTNumberOfSubbands_, LUTContextRefinement_, LUTContextSign_, LUTContextSignificance_, LUTMultPrecision_, LUTInformation_, DSizeArray, mainStream, measurementsBPC);

	delete BPCEng;
	delete BSB;
}
