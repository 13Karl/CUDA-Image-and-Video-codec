#pragma once
#include <fstream>
#include <math.h>
#include "../IO/IOManager.hpp"
#include "DWTGenerator.cuh"


template<class T, class Y>
inline DWT<T, Y>::DWT(Image* img, bool lossy, int wLevels, int cbWidth, int cbHeight, float qs)
{
	_isLossy = lossy;
	_numberOfWaveletLevels = wLevels;
	_codeBlockWidth = cbWidth;
	_codeBlockHeight = cbHeight;
	_qSize = qs;
	_imageToProcess = img;
}

template<class T, class Y>
inline DWT<T, Y>::~DWT()
{

}

template<class T, class Y>
void DWT<T, Y>::DWTEncodeChar(unsigned char* DInitialData, T* DFinalData, cudaStream_t mainStream)
{
	//In the DWT 9/7 each data block is overlapped with its neighbours 8 samples. For 5/3, it's only 4.
	int		overlap;
	if (_isLossy)
		overlap = 8;
	else
		overlap = 4;

	DWTEngine<T, Y> *DWTEng = new DWTEngine<T, Y>(_imageToProcess, _numberOfWaveletLevels, overlap, _isLossy, _qSize);

	DWTEng->DWTForwardChar(DInitialData, DFinalData, mainStream);

	delete DWTEng;
}

template<class T, class Y>
void DWT<T, Y>::DWTEncode(T* DInitialData, T* DFinalData, cudaStream_t mainStream)
{
	//In the DWT 9/7 each data block is overlapped with its neighbours 8 samples. For 5/3, it's only 4.
	int		overlap;
	if (_isLossy)
		overlap = 8;
	else
		overlap = 4;

	DWTEngine<T, Y> *DWTEng = new DWTEngine<T, Y>(_imageToProcess, _numberOfWaveletLevels, overlap, _isLossy, _qSize);
	DWTEng->DWTForward(DInitialData, DFinalData, mainStream);
	delete DWTEng;
}


template<class T, class Y>
void DWT<T, Y>::DWTDecode(int* decodedBitstream, T* DImagePixels, cudaStream_t mainStream)
{

	//In the DWT 9/7 each data block is overlapped with its neighbours 8 samples. For 5/3, it's only 4.
	int		overlap;
	if (_isLossy)
		overlap = 8;
	else
		overlap = 4;

	DWTEngine<T, Y> *DWTEng = new DWTEngine<T, Y>(_imageToProcess, _numberOfWaveletLevels, overlap, _isLossy, _qSize);
	DWTEng->DWTReverse(decodedBitstream, DImagePixels, mainStream);
	delete DWTEng;
}