
#pragma once
#ifndef DECODINGENGINE_CUH
#define DECODINGENGINE_CUH
#include "Engine.cuh"

class DecodingEngine : public Engine
{

	void engineManager(int cType);
	void initMemory(bool typeOfCoding);

	void runVideo(int iteration);
	void runImage();
	bool readFileParallel();
	bool readFileParallelRGB();
	void readCompressedImage();
	bool writeFileParallel();
	bool writeFileParallelLossy();
	bool writeFileParallelRGB();
	void writeDecompressedImage();
	void prepareRGBImage(cudaStream_t mainStream);
	void prepareRGBFrame(cudaStream_t mainStream, int iteration);
	void writeRGBImage();
	bool readRGBCompressedBitStream();

private:

	void getExtraInformation();
	void retrieveBasicImageInformation(std::string inputFile);


	int* _HBasicInformation;
	int** _HSizeArrayDB;
	int* _frameSizes;
	int* _componentSizes;
	int _frameSize;
	int* _bufferReadingValueRGB;

};
__device__ float const _irreversibleColorTransformBackward[3][3] = { { 1, 0, 1.402f },{ 1, -0.344136f, -0.714136f },{ 1, 1.772f, 0 } };
__global__ void RGBTransformLossless(int* inputR, int* inputG, int* inputB, unsigned char* outputR, unsigned char* outputG, unsigned char* outputB, int bitdepth, bool uSigned);
__global__ void RGBTransformLossy(float* inputR, float* inputG, float* inputB, unsigned char* outputR, unsigned char* outputG, unsigned char* outputB, int bitdepth, bool uSigned);
#endif