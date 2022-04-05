#pragma once
#include <string>
#include "../Image/Image.hpp"
#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"

#include <thread>
#ifndef DWTGENERATOR_HPP
#define DWTGENERATOR_HPP

typedef enum DWTEnum {
	FORWARD,
	REVERSE,
}DWTEnum;

template <class T, class Y>
class DWT
{
public:

	//Constructor
	DWT(Image* img, bool lossy, int wLevels, int cbWidth, int cbHeight, float qs);
	//Destructor
	~DWT();

	//Main Functions
	void DWTEncode(T* DInitialData, T* DFinalData, cudaStream_t mainStream);
	void DWTDecode(int* decodedBitstream, T* DImagePixels, cudaStream_t mainStream);
	void DWTEncodeChar(unsigned char* DInitialData, T* DFinalData, cudaStream_t mainStream);

private:

	Image* _imageToProcess;
	bool _isLossy;
	int _numberOfWaveletLevels;
	int _codeBlockWidth;
	int _codeBlockHeight;
	float _qSize;
};

#include "DWTGenerator.ipp"
#endif