#pragma once
#ifndef ENGINE_CUH
#define ENGINE_CUH
#include "../IO/IOManager.hpp"
#include "../SupportFunctions/AuxiliarFunctions.hpp"
#include "../DWT/DWTGenerator.hpp"
#include "../BPC/BPCEngine.hpp"
#include <cuda_runtime.h>
#include <vector>

#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"

#include <cuda.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <string>
#include <math.h>
#include <thread>
#include <future>
#include <chrono>

#define IMAGE 0
#define VIDEO 1
#define LOSSLESS false
#define LOSSY true

class Engine
{
public:
	virtual void engineManager(int cType) = 0;
	Engine();
	void setCodingPasses(int cp);
	void setWaveletLevels(int wL);
	void setNumberOfFrames(int nFr);
	void setOutputFile(std::string oF);
	void setWType(bool wt);
	void setCBWidth(int cbw);
	void setCBHeight(int cbh);
	void setQSizes(float qs);
	void setFrameStructure(Image* fS);
	void setLUTPath(std::string lp);
	void setNumberOfStreams(int nOfS);
	void setKFactor(float k);
protected:

	void loadLUTHeaders(std::string fullPath, IOManager<int, int2>* IOM);
	void initLUT();
	

	virtual void runVideo(int iteration) = 0;
	virtual void runImage() = 0;
	virtual bool readFileParallel() = 0;
	virtual bool writeFileParallel() = 0;
	virtual void initMemory(bool typeOfCoding) = 0;

	/*
	Variables for the Engine
	*/
	int _codingPasses;
	int _waveletLevels;
	int _numberOfFrames;
	std::string _outputFile;
	bool _waveletType;
	int _DWTCBWidth;
	int _DWTCBHeight;
	float _quantizationSize;
	Image* _frameStructure;
	std::string _LUTPath;
	float _k;

	/*
	Video Specific Codec Variables
	*/
	std::vector< unsigned short* > _codedFrames;
	std::vector< unsigned char* > _framesC;
	std::vector< int* > _frames;
	std::vector< float* > _framesLossy;
	std::vector< int > _framesSizes;
	int *_doubleBufferOutput;
	int *_doubleBufferOutputRGB;
	int *_doubleBufferInput;


	unsigned char** _DImagePixelsCharDB;
	unsigned char* _DImagePixelsChar;
	unsigned char** _DImagePixelsCharRGB;
	int** _DImagePixelsRGBTransformed;
	float** _DImagePixelsRGBTransformedLossy;
	unsigned char** _HImagePixelsCharDB;
	unsigned char* _HImagePixelsChar;
	unsigned char** _HImagePixelsCharRGB;
	cudaStream_t *_cStreams;
	cudaEvent_t *_cEvents;

	int** _DImagePixelsDB; //Input Data from file to encode - Output Data from file to decode
	float** _DImagePixelsDBLossy; //Input Data from file to encode Lossy - Output Data from file to decode Lossy
	int** _HImagePixelsDB; //Host version from the variable immediately above this one.
	float** _HImagePixelsDBLossy;
	int** _DWaveletCoefficientsDB; //Output data from Pixels to encode - Input data from file to decode
	float** _DWaveletCoefficientsDBLossy; //Output data from Pixels to encode Lossy - Input data from file to decode Lossy
	float* _DWaveletCoefficientsLossy;
	int** _DCodeStreamValuesDB; //Output data from DWT Coefficients to encode - Input Data for DWT to decode
	unsigned short** _DBitStreamValuesDB; //Output data from BitCon to encode and finish the process - Input data from file to start decoding.
	unsigned short** _HBitStreamValuesDB; //Host version from the variable immediately above this one.

	unsigned short* _HExtraInformation;
	int** _DPrefixedArrayDB;
	int** _DTempStoragePArrayDB; 
	int** _DSizeArrayDB;
	int** _DLUTBSTableDB;
	int** _HTotalBSSizeDB;

	/*
	LUTVariables
	*/

	int _LUTNumberOfBitplanes;
	int _LUTNumberOfSubbands;
	int _LUTContextRefinement;
	int _LUTContextSign;
	int _LUTContextSignificance;
	int _LUTMultPrecision;
	int _LUTNFiles;
	int** _LUTInformation;
	int _LUTAmountOfBitplaneFiles;

	/*
	Image Codec and general variables
	*/

	int* _DImagePixels; //Input Data from file to encode - Output Data from file to decode
	int* _HImagePixels; //Host version from the variable immediately above this one.
	float* _DImagePixelsLossy; //Input Data from file to encode - Output Data from file to decode
	float* _HImagePixelsLossy; //Host version from the variable immediately above this one.
	int* _DWaveletCoefficients; //Output data from Pixels to encode - Input data from file to decode
	int* _DCodeStreamValues; //Output data from DWT Coefficients to encode - Input Data for DWT to decode
	unsigned short* _DBitStreamValues; //Output data from BitCon to encode and finish the process - Input data from file to start decoding.
	unsigned short* _HBitStreamValues; //Host version from the variable immediately above this one.
	unsigned short** _HBitStreamValuesRGB;
	int* _DPrefixedArray;
	int* _DTempStoragePArray;
	int* _HSizeArray;
	int* _DSizeArray;
	int* _HTotalBSSize;
	int* _DLUTBSTable;
	int _HLUTBSTableSteps;
	int _extraWaveletAllocation;

	/*
	Parallelism parameters
	*/
	
	int _numOfStreams;
	int _numOfCPUThreads;
	int _numOfIOStreams;

	double* _measurementsBPC;

};

#endif