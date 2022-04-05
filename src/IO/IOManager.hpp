#pragma once
#include "../Image/Image.hpp"
#include <string>
#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"

#include <thread>
#ifndef IOMANAGER_H
#define IOMANAGER_H

template <class T, class Y>
class IOManager
{
public:

	//Constructor
	IOManager(std::string input, std::string output = "");
	IOManager();
	//Functions
	void loadBasicInfo(T* HBasicInformation, int amountOfValuesToRead, std::string inputFile);
	void loadImageChar(Image* image, unsigned char** waveletCoefficients);
	void loadFrameC(Image* image, unsigned char* waveletCoefficients, int iter);
	void loadFrameCAdaptedSizes(Image* image, unsigned char* waveletCoefficients, int iter);
	int loadCodedFrame(Image* img, T* currentFrame, int iter, int frameSize, long long int offset);
	void readBulkSizes (int *frameSizes, Image* img, int frames);
	void replaceExistingFile(std::string file);
	void writeImage(T* image, int xSize, int ySize, int bitDepth, std::string filename);
	void writeCodedFrame(Image* image, T* waveletCoefficients, int iter, int sizeOfBitStream, std::string outputName);
	void writeDecodedFrameUChar(Image* img, unsigned char** data, std::string outputName);
	void writeDecodedFrameComponentUChar(Image* img, unsigned char* currentFrame, int iter, std::string outputName);
	void writeDecodedFrame(Image* img, T* currentFrame, int iter, std::string outputName);
	int* loadLUTHeaders();
	void loadLUTUpgraded(int LUTNumberOfSubbands, int LUTContextRefinement, int LUTContextSignificance, int LUTContextSign, int codingPasses, int numberOfWaveletLevels, int LUTNumberOfBitplanes, int* hostLUT, int iter, float qStep, int amountOfBitplanes);
	char* concat(const char *s1, char *s2);
	void setInputFile(std::string iFile);
	void setInputFolder(std::string iFolder);
	void writeBitStreamFile(T *data, int sizeOfBitStream, std::string outputFileName);
	
	void readBitStreamFile(T* values, int sizeOfBitStream);

	//Consts
	int* getOutputData() const;

private:

	int convertStringToInt(std::string str, char delimiter);
	std::string _input;
	std::string _output;
	std::string _inputFolder;
	int* _outputData;

};

#include "IOManager.ipp"

#endif