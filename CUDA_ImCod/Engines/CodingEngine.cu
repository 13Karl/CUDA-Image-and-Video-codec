#include "CodingEngine.cuh"


/*
* Memory initialization function which preallocates every memory needed in the process.
* It takes in consideration everything from Image/Video, Lossy/Lossless and GrayScale/RGB.
*/
void CodingEngine::initMemory(bool typeOfCoding)
{
	SupportFunctions::fixImageProportions(this->_frameStructure, CBLOCK_LENGTH, CBLOCK_WIDTH);
	if (typeOfCoding == VIDEO)
	{
		int extraAlloc = 1;
		if (_frameStructure->getIsRGB() == true)
		{
			extraAlloc = 3;
		}
		_codedFrames.resize(_numOfCPUThreads * 2 *extraAlloc);
		_framesSizes.resize(_numOfCPUThreads * 2 *extraAlloc);
		int numberOfStreams = _numOfIOStreams / 2;
		_DImagePixelsDB = (int**)malloc(numberOfStreams * sizeof(int**));
		_DImagePixelsDBLossy = (float**)malloc(numberOfStreams * sizeof(float**));
		_DBitStreamValuesDB = (unsigned short**)malloc(numberOfStreams * sizeof(unsigned short**) * extraAlloc);
		_DImagePixelsCharDB = (unsigned char**)malloc(numberOfStreams * sizeof(unsigned char**) * extraAlloc);
		_HBitStreamValuesDB = (unsigned short**)malloc(numberOfStreams * sizeof(unsigned short**) * extraAlloc);
		_DCodeStreamValuesDB = (int**)malloc(_numOfCPUThreads * sizeof(int**));
		_DSizeArrayDB = (int**)malloc(_numOfCPUThreads * sizeof(int**));
		_DPrefixedArrayDB = (int**)malloc(_numOfCPUThreads * sizeof(int**));
		_DTempStoragePArrayDB = (int**)malloc(_numOfCPUThreads * sizeof(int**));
		_HTotalBSSizeDB = (int**)malloc(_numOfCPUThreads * sizeof(int**));
		_DLUTBSTableDB = (int**)malloc(_numOfCPUThreads * sizeof(int**));
		_DWaveletCoefficientsDB = (int**)malloc(_numOfCPUThreads * sizeof(int**));
		_DWaveletCoefficientsDBLossy = (float**)malloc(_numOfCPUThreads * sizeof(float**));
		_cStreams = new cudaStream_t[_numOfIOStreams + _numOfCPUThreads];
		_cEvents = new cudaEvent_t[_numOfIOStreams + _numOfCPUThreads];
		_doubleBufferOutput = new int[numberOfStreams];
		memset(_doubleBufferOutput, 0, numberOfStreams * sizeof(int));
		_doubleBufferInput = new int[numberOfStreams];
		memset(_doubleBufferInput, 0, numberOfStreams * sizeof(int));
		for (int i = 0; i < _numOfIOStreams + _numOfCPUThreads; i++)
		{
			GPU_HANDLE_ERROR(cudaStreamCreate(&_cStreams[i]));
			GPU_HANDLE_ERROR(cudaEventCreate(&(_cEvents[i])));
		}
		_extraWaveletAllocation = 0;

		int size = _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(int);
		//Memory Allocation for BitCon Coding Process
		_HExtraInformation = (unsigned short*)malloc(9 * sizeof(unsigned short));

		for (int l = 1; l<_waveletLevels; ++l)
			_extraWaveletAllocation += (_frameStructure->getAdaptedWidth() / (2 << (l - 1)))* (_frameStructure->getAdaptedHeight() / (2 << (l - 1)));
		_HLUTBSTableSteps = 256;

		if (_frameStructure->getIsRGB() == true)
		{
			if (_waveletType == LOSSLESS)
			{
				_DImagePixelsRGBTransformed = new int*[3*numberOfStreams];
				for (int i = 0; i < numberOfStreams * 3; i++)
				{
					GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformed[i], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(int)));
				}
			}
			else
			{
				_DImagePixelsRGBTransformedLossy = new float*[3*numberOfStreams];
				for (int i = 0; i < numberOfStreams * 3; i++)
				{
					GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformedLossy[i], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(float)));
				}
			}
		}

		for (int j = 0; j < numberOfStreams * extraAlloc; j++)
		{
			GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsCharDB[j], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()));
			GPU_HANDLE_ERROR(cudaMalloc(&_DBitStreamValuesDB[j], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(unsigned short)));
			cudaHostAlloc(&_HBitStreamValuesDB[j], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(unsigned short), 0);
		}

		int storage = _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight() / (CBLOCK_WIDTH*CBLOCK_LENGTH * 2);
		//This way we make sure that for really small frames or images, the temporal storage needed by CUB is covered. For really big images or frames, the equation above is enough - tested empirically.
		if (storage < 1000)
			storage = 1000;
		for (int k = 0; k < _numOfCPUThreads; k++)
		{
			//Memory Allocation for the BPC Coding Process.
			GPU_HANDLE_ERROR(cudaMalloc(&_DCodeStreamValuesDB[k], size));
			GPU_HANDLE_ERROR(cudaMalloc(&_DSizeArrayDB[k], (int)ceil(_frameStructure->getAdaptedWidth() / (float)CBLOCK_WIDTH) * (int)ceil(_frameStructure->getAdaptedHeight() / (float)CBLOCK_LENGTH) * sizeof(int)));
			GPU_HANDLE_ERROR(cudaMalloc(&_DPrefixedArrayDB[k], ((int)ceil(_frameStructure->getAdaptedWidth() / (float)CBLOCK_WIDTH) * (int)ceil(_frameStructure->getAdaptedHeight() / (float)CBLOCK_LENGTH)) * sizeof(int) + sizeof(int)));
			GPU_HANDLE_ERROR(cudaMalloc(&_DTempStoragePArrayDB[k], storage));
			_HTotalBSSizeDB[k] = (int*)malloc(sizeof(int));
			GPU_HANDLE_ERROR(cudaMalloc(&_DLUTBSTableDB[k], _HLUTBSTableSteps * sizeof(int) + 4));
			if (_waveletType == LOSSLESS)
			{
				GPU_HANDLE_ERROR(cudaMalloc(&_DWaveletCoefficientsDB[k], size + _extraWaveletAllocation * sizeof(int)));
				if (_frameStructure->getIsRGB() == false)
				{
					if (_frameStructure->getSignedOrUnsigned() == 0)
					{
						GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsDB[k], size));
					}
				}
			}
			else
			{
				GPU_HANDLE_ERROR(cudaMalloc(&_DWaveletCoefficientsDBLossy[k], size + _extraWaveletAllocation * sizeof(float)));
				if (_frameStructure->getIsRGB() == false)
				{
					if (_frameStructure->getSignedOrUnsigned() == 0)
					{
						GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsDBLossy[k], size));
					}
				}
			}
		}
	}
	else
	{
		int DDataExtra = 0;
		int size = _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(int);

		if (_frameStructure->getIsRGB() == true)
		{
			int sizeOfImage = _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight();
			_HImagePixelsCharRGB = new unsigned char*[3];
			_DImagePixelsCharRGB = new unsigned char*[3];
			cudaHostAlloc(&(_HImagePixelsCharRGB[0]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight(), 0);
			cudaHostAlloc(&(_HImagePixelsCharRGB[1]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight(), 0);
			cudaHostAlloc(&(_HImagePixelsCharRGB[2]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight(), 0);
			GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsCharRGB[0], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()));
			GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsCharRGB[1], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()));
			GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsCharRGB[2], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()));

			if (_waveletType == LOSSLESS)
			{
				_DImagePixelsRGBTransformed = new int*[3];
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformed[0], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(int)));
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformed[1], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(int)));
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformed[2], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(int)));

			}
			else
			{
				_DImagePixelsRGBTransformedLossy = new float*[3];
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformedLossy[0], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(float)));
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformedLossy[1], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(float)));
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformedLossy[2], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(float)));
				
			}
		}
		else
		{

			//Memory Allocation for the DWT Coding Process.
			GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsChar, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()));
			if (_frameStructure->getSignedOrUnsigned() == 0)
			{
				if (_waveletType == LOSSY)
				{
					GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsLossy, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(float)));
				}
				else
					GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixels, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(int)));
					
			}
		}

		for (int i = 1; i<_waveletLevels; ++i)
			DDataExtra += (_frameStructure->getAdaptedWidth() / (2 << (i - 1)))* (_frameStructure->getAdaptedHeight() / (2 << (i - 1)));
		if (_waveletType == LOSSLESS)
		{
			GPU_HANDLE_ERROR(cudaMalloc(&_DWaveletCoefficients, size + DDataExtra * sizeof(int)));
		}
		else
			GPU_HANDLE_ERROR(cudaMalloc(&_DWaveletCoefficientsLossy, size + DDataExtra * sizeof(float)));

		//Memory Allocation for the BPC Coding Process.
		GPU_HANDLE_ERROR(cudaMalloc(&_DCodeStreamValues, size));

		GPU_HANDLE_ERROR(cudaMalloc(&_DSizeArray, (int)ceil(_frameStructure->getAdaptedWidth() / (float)CBLOCK_WIDTH) * (int)ceil(_frameStructure->getAdaptedHeight() / (float)CBLOCK_LENGTH) * sizeof(int)));
		GPU_HANDLE_ERROR(cudaMalloc(&_DPrefixedArray, ((int)ceil(_frameStructure->getAdaptedWidth() / (float)CBLOCK_WIDTH) * (int)ceil(_frameStructure->getAdaptedHeight() / (float)CBLOCK_LENGTH)) * sizeof(int) + sizeof(int)));
		
		int storage = _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight() / (CBLOCK_WIDTH*CBLOCK_LENGTH * 2);
		//This way we make sure that for really small frames or images, the temporal storage needed by CUB is covered. For really big images or frames, the equation above is enough - tested empirically.
		if (storage < 1000)
			storage = 1000;
		GPU_HANDLE_ERROR(cudaMalloc(&_DTempStoragePArray, storage));

		_HLUTBSTableSteps = 256;
		_HTotalBSSize = (int*)malloc(sizeof(int));
		GPU_HANDLE_ERROR(cudaMalloc(&_DLUTBSTable, _HLUTBSTableSteps * sizeof(int) + 4));

		//Memory Allocation for BitCon Coding Process
		_HExtraInformation = (unsigned short*)malloc(9 * sizeof(unsigned short));
		GPU_HANDLE_ERROR(cudaMalloc(&_DBitStreamValues, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(unsigned short)));
		_HBitStreamValues = (unsigned short*)malloc(_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(unsigned short));
	}
}

/*
I/O Function in charge of reading a video frame by frame and placing it in the corresponding memory buffer. Managed by a CPU thread, it controls a 
double structure memory buffer. With the aid of status flags inside the array _doubleBufferInput it knows when a buffer
becomes available for new information. This module reads only grayscale videos.
Statuses for buffer queue:
0: Not used yet (init state).
1: Being copied from host to GPU.
2: Being processed by the algorithm.
3: Information already used, ready to be replaced.
*/
bool CodingEngine::readFileParallel()
{
	bool ret = false;
	IOManager<int, int2> *IOM = new IOManager<int, int2>();
	int iter = 0;
	int bufferValue;
	auto accumulated = 0.0;
	auto startTime = std::chrono::high_resolution_clock::now();
	auto endTime = std::chrono::high_resolution_clock::now();
	unsigned char* framesC;
	unsigned char* currentFrameC;
	int numberOfIStreams = _numOfIOStreams / 2;

	for (int i = 0; i < numberOfIStreams; i++)
	{
		cudaHostAlloc(&framesC, _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight(), 0);
		_framesC.push_back(framesC);
	}

	for (iter; iter < _numberOfFrames; iter++)
	{
		bufferValue = _doubleBufferInput[(iter) % numberOfIStreams];

		startTime = std::chrono::high_resolution_clock::now();
		while (bufferValue == 1 || (bufferValue == 2))
		{
			bufferValue = _doubleBufferInput[(iter) % numberOfIStreams];
		}

		SupportFunctions::markInitProfilerCPUSection("readFileParallelDoubleBuffer", "readFileParallel");
		endTime = std::chrono::high_resolution_clock::now();
		accumulated = accumulated + std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
		currentFrameC = _framesC.at(iter % numberOfIStreams);
		
		IOM->loadFrameCAdaptedSizes(_frameStructure, currentFrameC, iter);

		GPU_HANDLE_ERROR(cudaMemcpyAsync(_DImagePixelsCharDB[iter % numberOfIStreams], currentFrameC, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(unsigned char), cudaMemcpyHostToDevice, _cStreams[iter % numberOfIStreams]));
		GPU_HANDLE_ERROR(cudaEventRecord(_cEvents[iter % numberOfIStreams], _cStreams[iter % numberOfIStreams]));

		_doubleBufferInput[(iter) % numberOfIStreams] = 1;

		SupportFunctions::markEndProfilerCPUSection();
	}
	if (iter == _numberOfFrames)
		ret = true;

	std::cout << "Acumulado de Lectura: " << accumulated << std::endl;

	delete IOM;
	return ret;
}

/*
* I/O Function in charge of reading a video frame by frame and placing it in the corresponding memory buffer. Managed by a CPU thread, it controls a
* double structure memory buffer and with the aid of status flags inside the array _doubleBufferInput it knows when a buffer
* becomes available for new information. This module reads only RGB videos.
* Statuses for buffer queue:
* 0: Not used yet (init state).
* 1: Being copied from host to GPU.
* 2: Being processed by the algorithm.
* 3: Information already used, ready to be replaced.
*/
bool CodingEngine::readFileParallelRGB()
{
	bool ret = false;
	IOManager<int, int2> *IOM = new IOManager<int, int2>();
	int iter = 0;
	int bufferValue;
	auto accumulated = 0.0;
	auto startTime = std::chrono::high_resolution_clock::now();
	auto endTime = std::chrono::high_resolution_clock::now();
	unsigned char* framesC;
	unsigned char* currentFrameC;
	int numberOfIStreams = _numOfIOStreams / 2;
	int loopCounter = 0;
	for (int i = 0; i < numberOfIStreams*3; i++)
	{
		cudaHostAlloc(&framesC, _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight(), 0);
		_framesC.push_back(framesC);
	}

	for (iter; iter < _numberOfFrames; iter++)
	{
		bufferValue = _doubleBufferInput[(iter) % numberOfIStreams];

		startTime = std::chrono::high_resolution_clock::now();
		while (bufferValue == 1 || (bufferValue == 2))
		{
			bufferValue = _doubleBufferInput[(iter) % numberOfIStreams];
		}

		SupportFunctions::markInitProfilerCPUSection("readFileParallelDoubleBuffer", "readFileParallel");
		endTime = std::chrono::high_resolution_clock::now();
		accumulated = accumulated + std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
		for (loopCounter = 0; loopCounter < 3; loopCounter++)
		{
			currentFrameC = _framesC.at((iter*3+loopCounter) % (numberOfIStreams*3));
			
			IOM->loadFrameCAdaptedSizes(_frameStructure, currentFrameC, iter*3+loopCounter);
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_DImagePixelsCharDB[(iter*3+loopCounter) % (numberOfIStreams*3)], currentFrameC, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(unsigned char), cudaMemcpyHostToDevice, _cStreams[iter % numberOfIStreams]));
		}
		
		GPU_HANDLE_ERROR(cudaEventRecord(_cEvents[iter % numberOfIStreams], _cStreams[iter % numberOfIStreams]));
		_doubleBufferInput[(iter) % numberOfIStreams] = 1;

		SupportFunctions::markEndProfilerCPUSection();
	}
	if (iter == _numberOfFrames)
		ret = true;

	std::cout << "Acumulado de Lectura: " << accumulated << std::endl;

	delete IOM;
	return ret;
}

void CodingEngine::readGrayScaleImage()
{
	SupportFunctions::markInitProfilerCPUSection("IO", "Disk Reading");
	IOManager<int, int2> *IOM = new IOManager<int, int2>();
	int sizeOfImage = _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight();
	cudaHostAlloc(&_HImagePixelsChar, _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight(), 0);
	IOM->loadFrameCAdaptedSizes(_frameStructure, _HImagePixelsChar, 0);
	GPU_HANDLE_ERROR(cudaMemcpyAsync(_DImagePixelsChar, _HImagePixelsChar, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(unsigned char), cudaMemcpyHostToDevice, cudaStreamDefault));
	delete IOM;
	SupportFunctions::markEndProfilerCPUSection();
}

void CodingEngine::readRGBImage()
{
	SupportFunctions::markInitProfilerCPUSection("IO", "Disk Reading");
	IOManager<int, int2> *IOM = new IOManager<int, int2>();

	for (int i = 0; i < 3; i++)
	{
		 IOM->loadFrameCAdaptedSizes(_frameStructure, _HImagePixelsCharRGB[i], i);
		 GPU_HANDLE_ERROR(cudaMemcpyAsync(_DImagePixelsCharRGB[i], _HImagePixelsCharRGB[i], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(unsigned char), cudaMemcpyHostToDevice, cudaStreamDefault));
	}
	delete IOM;
	SupportFunctions::markEndProfilerCPUSection();
}

/*
* Kernel which launches the lossless color transformation, changing from RGB color space to YCbCr color space. It also reduces the size of the samples by applying an offset if the data type is unsigned.
*/
__global__ void RGBTransformLossless(unsigned char* inputR, unsigned char* inputG, unsigned char* inputB, int* outputR, int* outputG, int* outputB, int bitdepth, bool uSigned)
{
	int offset = 0;
	if (uSigned == false)
	{
		offset = 1 << (bitdepth - 1);
	}
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	float componentR = 0;
	float componentG = 0;
	float componentB = 0;
	componentR = (float)inputR[threadId] - offset;
	componentG = (float)inputG[threadId] - offset;
	componentB = (float)inputB[threadId] - offset;

	int componentRTransformed = (int)floor((componentR + (componentG * 2) + componentB) / 4);
	int componentGTransformed = (int)(componentB - componentG);
	int componentBTransformed = (int)(componentR - componentG);

	outputR[threadId] = componentRTransformed;
	outputG[threadId] = componentGTransformed;
	outputB[threadId] = componentBTransformed;
}

/*
* Kernel which launches the lossy color transformation, changing from RGB color space to YCbCr color space. It also reduces the size of the samples by applying an offset if the data type is unsigned.
*/
__global__ void RGBTransformLossy(unsigned char* inputR, unsigned char* inputG, unsigned char* inputB, float* outputR, float* outputG, float* outputB, int bitdepth, bool uSigned)
{
	int offset = 0;
	if (uSigned == false)
	{
		offset = 1 << (bitdepth - 1);
	}
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	float componentR = (float)inputR[threadId] - offset;
	float componentG = (float)inputG[threadId] - offset;
	float componentB = (float)inputB[threadId] - offset;

	float componentRTransformed = _irreversibleColorTransformForward[0][0] * componentR + _irreversibleColorTransformForward[0][1] * componentG + _irreversibleColorTransformForward[0][2] * componentB;
	float componentGTransformed = _irreversibleColorTransformForward[1][0] * componentR + _irreversibleColorTransformForward[1][1] * componentG + _irreversibleColorTransformForward[1][2] * componentB;
	float componentBTransformed = _irreversibleColorTransformForward[2][0] * componentR + _irreversibleColorTransformForward[2][1] * componentG + _irreversibleColorTransformForward[2][2] * componentB;

	outputR[threadId] = componentRTransformed;
	outputG[threadId] = componentGTransformed;
	outputB[threadId] = componentBTransformed;
}

/*
* Host function which launches the color transformation for images
*/
void CodingEngine::prepareRGBImage(cudaStream_t mainStream)
{
	if (_waveletType == LOSSLESS)
	{
		int numberOfThreadsPerBlock = 256;
		int numberOfBlocks = (int)ceil(_frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth() / numberOfThreadsPerBlock);
		RGBTransformLossless <<<numberOfBlocks, numberOfThreadsPerBlock, 0, mainStream>>>	(_DImagePixelsCharRGB[0], _DImagePixelsCharRGB[1], _DImagePixelsCharRGB[2], _DImagePixelsRGBTransformed[0], _DImagePixelsRGBTransformed[1], _DImagePixelsRGBTransformed[2], _frameStructure->getBitDepth(), _frameStructure->getSignedOrUnsigned());
		cudaStreamSynchronize(mainStream);
		KERNEL_ERROR_HANDLER;
	}
	else
	{
		int numberOfThreadsPerBlock = 256;
		int numberOfBlocks = (int)ceil(_frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth() / numberOfThreadsPerBlock);
		RGBTransformLossy << <numberOfBlocks, numberOfThreadsPerBlock, 0, mainStream >> >	(_DImagePixelsCharRGB[0], _DImagePixelsCharRGB[1], _DImagePixelsCharRGB[2], _DImagePixelsRGBTransformedLossy[0], _DImagePixelsRGBTransformedLossy[1], _DImagePixelsRGBTransformedLossy[2], _frameStructure->getBitDepth(), _frameStructure->getSignedOrUnsigned());
		cudaStreamSynchronize(mainStream);
		KERNEL_ERROR_HANDLER;
	}
}

/*
* Host function which launches the color transformation for frames
*/
void CodingEngine::prepareRGBFrame(cudaStream_t mainStream, int iteration)
{
	if (_waveletType == LOSSLESS)
	{
		int numberOfThreadsPerBlock = 256;
		int numberOfBlocks = (int)ceil(_frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth() / numberOfThreadsPerBlock);
		RGBTransformLossless << <numberOfBlocks, numberOfThreadsPerBlock, 0, mainStream >> >	(_DImagePixelsCharDB[(iteration*3)% (_numOfIOStreams/2*3)], _DImagePixelsCharDB[(iteration * 3 + 1) % (_numOfIOStreams / 2 * 3)], _DImagePixelsCharDB[(iteration * 3 + 2) % (_numOfIOStreams / 2 * 3)], _DImagePixelsRGBTransformed[(iteration * 3) % (_numOfIOStreams / 2 * 3)], _DImagePixelsRGBTransformed[(iteration * 3 + 1) % (_numOfIOStreams / 2 * 3)], _DImagePixelsRGBTransformed[(iteration * 3 + 2) % (_numOfIOStreams / 2 * 3)], _frameStructure->getBitDepth(), _frameStructure->getSignedOrUnsigned());
		cudaStreamSynchronize(mainStream);
		KERNEL_ERROR_HANDLER;
	}
	else
	{
		int numberOfThreadsPerBlock = 256;
		int numberOfBlocks = (int)ceil(_frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth() / numberOfThreadsPerBlock);
		RGBTransformLossy << <numberOfBlocks, numberOfThreadsPerBlock, 0, mainStream >> >	(_DImagePixelsCharDB[(iteration * 3) % (_numOfIOStreams / 2 * 3)], _DImagePixelsCharDB[(iteration * 3 + 1) % (_numOfIOStreams / 2 * 3)], _DImagePixelsCharDB[(iteration * 3 + 2) % (_numOfIOStreams / 2 * 3)], _DImagePixelsRGBTransformedLossy[(iteration * 3) % (_numOfIOStreams / 2 * 3)], _DImagePixelsRGBTransformedLossy[(iteration * 3 + 1) % (_numOfIOStreams / 2 * 3)], _DImagePixelsRGBTransformedLossy[(iteration * 3 + 2) % (_numOfIOStreams / 2 * 3)], _frameStructure->getBitDepth(), _frameStructure->getSignedOrUnsigned());
		cudaStreamSynchronize(mainStream);
		KERNEL_ERROR_HANDLER;
	}
}

/*
* I/O Function in charge of writing back to the disk the compressed frames coded by the engine. This function manages only grayscale videos,
* and it uses a double buffer strategy to manage the information coming from the engine. Once a stream finishes coding a frame, it is placed
* in a buffer. This method is managed by a single CPU thread and constantly checks for new information. Once new information
* arrives, it is written back to the disk. The way it detects the incoming information is by making use of a system of flags placed inside
* the array _doubleBufferOutput.
* Statuses for buffer queue:
* 0: Not used yet (init state).
* 1: Being copied from host to GPU.
* 2: Being processed by the algorithm.
* 3: Information already used, ready to be replaced.
*/
bool CodingEngine::writeFileParallel()
{
	bool ret = true;
	int bufferValue = 0;
	IOManager<unsigned short, ushort2> *IOM = new IOManager<unsigned short, ushort2>();
	unsigned short* codedFrame;
	auto accumulated = 0.0;
	auto startTime = std::chrono::high_resolution_clock::now();
	auto endTime = std::chrono::high_resolution_clock::now();
	IOM->replaceExistingFile(_outputFile);
	IOM->replaceExistingFile(_outputFile + "_SIZE");
	int numberOfOStreams = _numOfIOStreams / 2;
	for (int i = 0; i < _numberOfFrames; i++)
	{
		bufferValue = _doubleBufferOutput[(i) % numberOfOStreams];
		startTime = std::chrono::high_resolution_clock::now();
		while (bufferValue != 1)
		{
			bufferValue = _doubleBufferOutput[(i) % numberOfOStreams];
		}
		SupportFunctions::markInitProfilerCPUSection("writeCodedFileParallelDoubleBuffer", "writeCodedFile");
		endTime = std::chrono::high_resolution_clock::now();
		if (i != 0)
			accumulated = accumulated + std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

		GPU_HANDLE_ERROR(cudaEventSynchronize(_cEvents[(i % numberOfOStreams) + numberOfOStreams]));
		codedFrame = _codedFrames.at(i % numberOfOStreams);
		_doubleBufferOutput[(i) % numberOfOStreams] = 2;
		IOM->writeCodedFrame(_frameStructure, codedFrame, i, _framesSizes.at(i % numberOfOStreams), _outputFile);
		_doubleBufferOutput[(i) % numberOfOStreams] = 3;
		SupportFunctions::markEndProfilerCPUSection();
	}
	std::cout << "Acumulado de Escritura: " << accumulated << std::endl;
	delete IOM;
	return ret;
}

/*
* I/O Function in charge of writing back to the disk the compressed frames coded by the engine. This function manages only RGB videos,
* and it uses a double buffer strategy to manage the information coming from the engine. Once a stream finishes coding a frame, it is placed
* in a corresponding buffer. This method is managed by a single CPU thread and constantly checks for new information. Once new information
* arrives, it is written back to the disk. The way it detects the incoming information is by making use of a system of flags placed inside
* the array _doubleBufferOutput.
* Statuses for buffer queue:
* 0: Not used yet (init state).
* 1: Being copied from host to GPU.
* 2: Being processed by the algorithm.
* 3: Information already used, ready to be replaced.
*/
bool CodingEngine::writeFileParallelRGB()
{
	bool ret = true;
	int bufferValue = 0;
	IOManager<unsigned short, ushort2> *IOM = new IOManager<unsigned short, ushort2>();
	unsigned short* codedFrame;
	auto accumulated = 0.0;
	auto startTime = std::chrono::high_resolution_clock::now();
	auto endTime = std::chrono::high_resolution_clock::now();
	IOM->replaceExistingFile(_outputFile);
	IOM->replaceExistingFile(_outputFile + "_SIZE");
	int numberOfOStreams = _numOfIOStreams / 2;
	for (int i = 0; i < _numberOfFrames; i++)
	{
		bufferValue = _doubleBufferOutput[(i) % numberOfOStreams];
		startTime = std::chrono::high_resolution_clock::now();
		while (bufferValue != 1)
		{
			bufferValue = _doubleBufferOutput[(i) % numberOfOStreams];
		}
		SupportFunctions::markInitProfilerCPUSection("writeCodedFileParallelDoubleBuffer", "writeCodedFile");
		endTime = std::chrono::high_resolution_clock::now();
		if (i != 0)
			accumulated = accumulated + std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

		GPU_HANDLE_ERROR(cudaEventSynchronize(_cEvents[(i % numberOfOStreams) + numberOfOStreams]));
		_doubleBufferOutput[(i) % numberOfOStreams] = 2;
		for (int j = 0; j < 3; j++)
		{
			codedFrame = _codedFrames.at((i*3+j) % (numberOfOStreams*3));
			IOM->writeCodedFrame(_frameStructure, codedFrame, (i*3+j), _framesSizes.at((i*3+j) % (numberOfOStreams*3)), _outputFile);
		}
		_doubleBufferOutput[(i) % numberOfOStreams] = 3;
		SupportFunctions::markEndProfilerCPUSection();
	}
	std::cout << "Acumulado de Escritura: " << accumulated << std::endl;
	delete IOM;
	return ret;
}


/*
* Function which calls the IO Manager to write a grayscale image.
*/
void CodingEngine::writeCodedBitStream()
{
	IOManager<unsigned short, ushort2> *IOM = new IOManager<unsigned short, ushort2>();
	IOM->writeBitStreamFile(_HBitStreamValues, _HTotalBSSize[0], _outputFile);
	delete IOM;
}

/*
* Function which calls the IO Manager to write an RGB image.
*/
void CodingEngine::writeRGBBitStream(cudaStream_t mainStream, int iter)
{
	IOManager<unsigned short, ushort2> *IOM = new IOManager<unsigned short, ushort2>();
	if (iter == 0)
	{
		IOM->replaceExistingFile(_outputFile);
		IOM->replaceExistingFile(_outputFile + "_SIZE");
	}
	IOM->writeCodedFrame(_frameStructure, _HBitStreamValues, iter, _HTotalBSSize[0], _outputFile);
	delete IOM;
}

/*
* Transforms the image taken into the proper data type.
*/
template<class T>
__global__ void offsetImage(unsigned char* inputData, T* outputData, int bitDepth)
{
	int threadId = threadIdx.x + blockIdx.x*blockDim.x;
	T charData = (T)(inputData[threadId]);
	charData = charData - (1 << (bitDepth - 1));
	outputData[threadId] = charData;
}

/*
* Function which manages the general flow of instructions to code images, either in lossy / lossless or grayscale / RGB.
*/
void CodingEngine::runImage()
{
	if (_waveletType == LOSSLESS)
	{

		if (_frameStructure->getIsRGB())
		{
			
			readRGBImage();
			auto startProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
			SupportFunctions::markInitProfilerCPUSection("ColorTransform", "Color Tranform Kernel");
			prepareRGBImage(cudaStreamDefault);
			SupportFunctions::markEndProfilerCPUSection();
			for (int i = 0; i < 3; i++)
			{
				SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Init");
				DWT<int, int2>* DWTGen = new DWT<int, int2>(_frameStructure, _waveletType, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _quantizationSize);
				SupportFunctions::markEndProfilerCPUSection();
				SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Encoding");

				DWTGen->DWTEncode(_DImagePixelsRGBTransformed[i], _DWaveletCoefficients, cudaStreamDefault);
				SupportFunctions::markEndProfilerCPUSection();

				SupportFunctions::markInitProfilerCPUSection("BPC", "BPC");
				BPCCuda<int>* BPC = new BPCCuda<int>(_frameStructure, _DWaveletCoefficients, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);
				BPC->Code(_LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[i], _DCodeStreamValues, _DPrefixedArray, _DTempStoragePArray, _DSizeArray, _HExtraInformation, _DBitStreamValues, _HTotalBSSize, _DLUTBSTable, _HLUTBSTableSteps, i, cudaStreamDefault, _numberOfFrames, &_measurementsBPC[0]);
				SupportFunctions::markEndProfilerCPUSection();
				SupportFunctions::markInitProfilerCPUSection("Copying", "Writing to disk");

				GPU_HANDLE_ERROR(cudaMemcpy(_HBitStreamValues, _DBitStreamValues, (_HTotalBSSize[0] * sizeof(unsigned short)), cudaMemcpyDeviceToHost));
				this->writeRGBBitStream(cudaStreamDefault, i);

				delete DWTGen;
				delete BPC;
				SupportFunctions::markEndProfilerCPUSection();
			}
			auto finishProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
			double elapsedTimeProcessDisregardAllocationTimings = std::chrono::duration_cast<std::chrono::duration<double>>(finishProcessDisregardAllocationTimings - startProcessDisregardAllocationTimings).count();
			std::cout << "The time spent with the app without considering allocation periods is: " << elapsedTimeProcessDisregardAllocationTimings << std::endl;
			std::cout << "BPC acum time is: " << _measurementsBPC[0] << std::endl;
		}
		else
		{
			SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Init");
			DWT<int, int2>* DWTGen = new DWT<int, int2>(_frameStructure, _waveletType, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _quantizationSize);
			SupportFunctions::markEndProfilerCPUSection();
			this->readGrayScaleImage();
			SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Encoding");



			if (_frameStructure->getSignedOrUnsigned() == 0)
			{
				int numberOfThreadsPerBlock = 256;
				int numberOfBlocks = (int)ceil(_frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth() / numberOfThreadsPerBlock);
				offsetImage << < numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStreamDefault >> > (_DImagePixelsChar, _DImagePixels, _frameStructure->getBitDepth());
				cudaStreamSynchronize(cudaStreamDefault);
				KERNEL_ERROR_HANDLER;
				DWTGen->DWTEncode(_DImagePixels, _DWaveletCoefficients, cudaStreamDefault);
			}
			else
			{
				DWTGen->DWTEncodeChar(_DImagePixelsChar, _DWaveletCoefficients, cudaStreamDefault);
			}

			SupportFunctions::markEndProfilerCPUSection();

			SupportFunctions::markInitProfilerCPUSection("BPC", "BPC");
			BPCCuda<int>* BPC = new BPCCuda<int>(_frameStructure, _DWaveletCoefficients, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);
			BPC->Code(_LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[0], _DCodeStreamValues, _DPrefixedArray, _DTempStoragePArray, _DSizeArray, _HExtraInformation, _DBitStreamValues, _HTotalBSSize, _DLUTBSTable, _HLUTBSTableSteps, 0, cudaStreamDefault, _numberOfFrames, &_measurementsBPC[0]);
			SupportFunctions::markEndProfilerCPUSection();
			SupportFunctions::markInitProfilerCPUSection("Copying", "Writing to disk");

			GPU_HANDLE_ERROR(cudaMemcpy(_HBitStreamValues, _DBitStreamValues, (_HTotalBSSize[0] * sizeof(unsigned short)), cudaMemcpyDeviceToHost));
			this->writeCodedBitStream();

			delete DWTGen;
			delete BPC;
			SupportFunctions::markEndProfilerCPUSection();

			std::cout << "BPC acum time is: " << _measurementsBPC[0] << std::endl;
		}
			

	}
	else
	{
		if (_frameStructure->getIsRGB())
		{
			readRGBImage();
			auto startProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
			SupportFunctions::markInitProfilerCPUSection("ColorTransform", "Color Tranform Kernel");
			prepareRGBImage(cudaStreamDefault);
			SupportFunctions::markEndProfilerCPUSection();
			DWT<float, float2>* DWTGen = new DWT<float, float2>(_frameStructure, _waveletType, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _quantizationSize);
			for (int i = 0; i < 3; i++)
			{
				SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Init");
				SupportFunctions::markEndProfilerCPUSection();
				SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Encoding");
				DWTGen->DWTEncode(_DImagePixelsRGBTransformedLossy[i], _DWaveletCoefficientsLossy, cudaStreamDefault);
				
				SupportFunctions::markEndProfilerCPUSection();
				SupportFunctions::markInitProfilerCPUSection("BPC", "BPC");
				BPCCuda<float>* BPC = new BPCCuda<float>(_frameStructure, _DWaveletCoefficientsLossy, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);
				BPC->Code(_LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[i], _DCodeStreamValues, _DPrefixedArray, _DTempStoragePArray, _DSizeArray, _HExtraInformation, _DBitStreamValues, _HTotalBSSize, _DLUTBSTable, _HLUTBSTableSteps, i, cudaStreamDefault, _numberOfFrames, &_measurementsBPC[0]);
				SupportFunctions::markEndProfilerCPUSection();
				SupportFunctions::markInitProfilerCPUSection("Copying", "Writting to disk");

				GPU_HANDLE_ERROR(cudaMemcpy(_HBitStreamValues, _DBitStreamValues, (_HTotalBSSize[0] * sizeof(unsigned short)), cudaMemcpyDeviceToHost));
				this->writeRGBBitStream(cudaStreamDefault, i);
				delete BPC;
				SupportFunctions::markEndProfilerCPUSection();
			}
			delete DWTGen;
			auto finishProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
			double elapsedTimeProcessDisregardAllocationTimings = std::chrono::duration_cast<std::chrono::duration<double>>(finishProcessDisregardAllocationTimings - startProcessDisregardAllocationTimings).count();
			std::cout << "The time spent with the app without considering allocation periods is: " << elapsedTimeProcessDisregardAllocationTimings << std::endl;
			std::cout << "BPC acum time is: " << _measurementsBPC[0] << std::endl;
		}
		else
		{
			SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Init");
			DWT<float, float2>* DWTGen = new DWT<float, float2>(_frameStructure, _waveletType, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _quantizationSize);
			SupportFunctions::markEndProfilerCPUSection();
			this->readGrayScaleImage();

			SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Encoding");
			if (_frameStructure->getSignedOrUnsigned() == 0)
			{
				int numberOfThreadsPerBlock = 256;
				int numberOfBlocks = (int)ceil(_frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth() / numberOfThreadsPerBlock);
				offsetImage << < numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStreamDefault >> > (_DImagePixelsChar, _DImagePixelsLossy, _frameStructure->getBitDepth());
				cudaStreamSynchronize(cudaStreamDefault);
				KERNEL_ERROR_HANDLER;
				DWTGen->DWTEncode(_DImagePixelsLossy, _DWaveletCoefficientsLossy, cudaStreamDefault);
			}
			else
			{
				DWTGen->DWTEncodeChar(_DImagePixelsChar, _DWaveletCoefficientsLossy, cudaStreamDefault);
			}
			
			SupportFunctions::markEndProfilerCPUSection();
			SupportFunctions::markInitProfilerCPUSection("BPC", "BPC");
			BPCCuda<float>* BPC = new BPCCuda<float>(_frameStructure, _DWaveletCoefficientsLossy, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);
			BPC->Code(_LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[0], _DCodeStreamValues, _DPrefixedArray, _DTempStoragePArray, _DSizeArray, _HExtraInformation, _DBitStreamValues, _HTotalBSSize, _DLUTBSTable, _HLUTBSTableSteps, 0, cudaStreamDefault, _numberOfFrames, &_measurementsBPC[0]);
			SupportFunctions::markEndProfilerCPUSection();
			SupportFunctions::markInitProfilerCPUSection("Copying", "Writting to disk");

			GPU_HANDLE_ERROR(cudaMemcpy(_HBitStreamValues, _DBitStreamValues, (_HTotalBSSize[0] * sizeof(unsigned short)), cudaMemcpyDeviceToHost));
			this->writeCodedBitStream();


			std::cout << "BPC acum time is: " << _measurementsBPC[0] << std::endl;

			delete DWTGen;
			delete BPC;
			SupportFunctions::markEndProfilerCPUSection();
		}
	}
}

/*
* Function which manages the general flow of instructions to code videos, either in lossy / lossless or grayscale / RGB.
*/
void CodingEngine::runVideo(int iteration)
{
	int doubleBufferSize = _numOfIOStreams / 2;
	int auxiliarFrameSizes[3];
	if (_waveletType == LOSSLESS)
	{
		if (_frameStructure->getIsRGB())
		{
			SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - PreSynch");
			int bufferValue;
			DWT<int, int2>* DWTGen;
			BPCCuda<int>* BPC;
			SupportFunctions::markEndProfilerCPUSection();

			SupportFunctions::markInitProfilerCPUSection("Copying Synching", "Copying - Synching");
			GPU_HANDLE_ERROR(cudaStreamWaitEvent(_cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams], _cEvents[iteration % (doubleBufferSize)], 0));
			SupportFunctions::markEndProfilerCPUSection();

			SupportFunctions::markInitProfilerCPUSection("ColorTransform", "Color Tranform Kernel");
			prepareRGBFrame(_cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)], iteration);
			SupportFunctions::markEndProfilerCPUSection();
			_doubleBufferInput[iteration % (doubleBufferSize)] = 2;

			for (int i = 0; i < 3; i++)
			{
				SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Encoding");
				DWTGen = new DWT<int, int2>(_frameStructure, _waveletType, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _quantizationSize);
				DWTGen->DWTEncode(_DImagePixelsRGBTransformed[(iteration * 3 + i) % (doubleBufferSize * 3)], _DWaveletCoefficientsDB[iteration % _numOfCPUThreads], _cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)]);
				SupportFunctions::markEndProfilerCPUSection();

				SupportFunctions::markInitProfilerCPUSection("BPC", "BPC");
				BPC = new BPCCuda<int>(_frameStructure, _DWaveletCoefficientsDB[iteration % _numOfCPUThreads], _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);
				BPC->Code(_LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[i], _DCodeStreamValuesDB[iteration % _numOfCPUThreads], _DPrefixedArrayDB[iteration % _numOfCPUThreads], _DTempStoragePArrayDB[iteration % _numOfCPUThreads], _DSizeArrayDB[iteration % _numOfCPUThreads], _HExtraInformation, _DBitStreamValuesDB[(iteration * 3 + i) % (doubleBufferSize * 3)], _HTotalBSSizeDB[iteration % _numOfCPUThreads], _DLUTBSTableDB[iteration % _numOfCPUThreads], _HLUTBSTableSteps, iteration, _cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)], _numberOfFrames, &_measurementsBPC[iteration % _numOfCPUThreads]);
				auxiliarFrameSizes[i] = _HTotalBSSizeDB[iteration % _numOfCPUThreads][0];
				SupportFunctions::markEndProfilerCPUSection();

				delete BPC;
				delete DWTGen;
			}
			cudaStreamSynchronize(_cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)]);
			_doubleBufferInput[iteration % (doubleBufferSize)] = 3;
			bufferValue = _doubleBufferOutput[iteration % (doubleBufferSize)];
			SupportFunctions::markInitProfilerCPUSection("Copying", "Copying - Waiting for write buffer.");
			while ((bufferValue == 1) || (bufferValue == 2))
			{
				bufferValue = _doubleBufferOutput[iteration % (doubleBufferSize)];
			}
			_framesSizes[(iteration * 3) % (doubleBufferSize * 3)] = auxiliarFrameSizes[0];
			_framesSizes[(iteration * 3 + 1) % (doubleBufferSize * 3)] = auxiliarFrameSizes[1];
			_framesSizes[(iteration * 3 + 2) % (doubleBufferSize * 3)] = auxiliarFrameSizes[2];
			SupportFunctions::markEndProfilerCPUSection();
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_HBitStreamValuesDB[(iteration * 3) % (doubleBufferSize * 3)], _DBitStreamValuesDB[(iteration * 3) % (doubleBufferSize * 3)], (_framesSizes[(iteration * 3) % (doubleBufferSize * 3)]) * sizeof(unsigned short), cudaMemcpyDeviceToHost, _cStreams[(iteration % (doubleBufferSize)) + (doubleBufferSize)]));
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_HBitStreamValuesDB[(iteration * 3 + 1) % (doubleBufferSize * 3)], _DBitStreamValuesDB[(iteration * 3 + 1) % (doubleBufferSize * 3)], (_framesSizes[(iteration * 3 + 1) % (doubleBufferSize * 3)]) * sizeof(unsigned short), cudaMemcpyDeviceToHost, _cStreams[(iteration % (doubleBufferSize)) + (doubleBufferSize)]));
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_HBitStreamValuesDB[(iteration * 3 + 2) % (doubleBufferSize * 3)], _DBitStreamValuesDB[(iteration * 3 + 2) % (doubleBufferSize * 3)], (_framesSizes[(iteration * 3 + 2) % (doubleBufferSize * 3)]) * sizeof(unsigned short), cudaMemcpyDeviceToHost, _cStreams[(iteration % (doubleBufferSize)) + (doubleBufferSize)]));
			GPU_HANDLE_ERROR(cudaEventRecord(_cEvents[(iteration % (doubleBufferSize)) + (doubleBufferSize)], _cStreams[(iteration % (doubleBufferSize)) + (doubleBufferSize)]));
			_codedFrames.at((iteration * 3) % (doubleBufferSize * 3)) = _HBitStreamValuesDB[(iteration * 3) % (doubleBufferSize * 3)];
			_codedFrames.at((iteration * 3 + 1) % (doubleBufferSize * 3)) = _HBitStreamValuesDB[(iteration * 3 + 1) % (doubleBufferSize * 3)];
			_codedFrames.at((iteration * 3 + 2) % (doubleBufferSize * 3)) = _HBitStreamValuesDB[(iteration * 3 + 2) % (doubleBufferSize * 3)];
			_doubleBufferOutput[iteration % (doubleBufferSize)] = 1;

		}
		else
		{
			SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - PreSynch");
			int bufferValue;
			DWT<int, int2>* DWTGen = new DWT<int, int2>(_frameStructure, _waveletType, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _quantizationSize);
			SupportFunctions::markEndProfilerCPUSection();

			SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Synching");
			GPU_HANDLE_ERROR(cudaStreamWaitEvent(_cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams], _cEvents[iteration % (doubleBufferSize)], 0));
			SupportFunctions::markEndProfilerCPUSection();

			_doubleBufferInput[iteration % (doubleBufferSize)] = 2;
			SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Encoding");


			if (_frameStructure->getSignedOrUnsigned() == 0)
			{
				int numberOfThreadsPerBlock = 256;
				int numberOfBlocks = (int)ceil(_frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth() / numberOfThreadsPerBlock);
				offsetImage << < numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStreamDefault >> > (_DImagePixelsCharDB[iteration % (doubleBufferSize)], _DImagePixelsDB[iteration % (_numOfCPUThreads)], _frameStructure->getBitDepth());
				DWTGen->DWTEncode(_DImagePixelsDB[iteration % (_numOfCPUThreads)], _DWaveletCoefficientsDB[iteration % _numOfCPUThreads], _cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)]);
			}
			else
			{
				DWTGen->DWTEncodeChar(_DImagePixelsCharDB[iteration % (doubleBufferSize)], _DWaveletCoefficientsDB[iteration % _numOfCPUThreads], _cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)]);
			}

			SupportFunctions::markEndProfilerCPUSection();
			// This instruction below could lead to race issues. The DWT is so fast that it is not currently a problem; therefore, no race condition is added to increase throughput.
			_doubleBufferInput[iteration % (doubleBufferSize)] = 3;

			SupportFunctions::markInitProfilerCPUSection("BPC", "BPC");
			BPCCuda<int>* BPC = new BPCCuda<int>(_frameStructure, _DWaveletCoefficientsDB[iteration % _numOfCPUThreads], _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);
			BPC->Code(_LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[0], _DCodeStreamValuesDB[iteration % _numOfCPUThreads], _DPrefixedArrayDB[iteration % _numOfCPUThreads], _DTempStoragePArrayDB[iteration % _numOfCPUThreads], _DSizeArrayDB[iteration % _numOfCPUThreads], _HExtraInformation, _DBitStreamValuesDB[iteration % (doubleBufferSize)], _HTotalBSSizeDB[iteration % _numOfCPUThreads], _DLUTBSTableDB[iteration % _numOfCPUThreads], _HLUTBSTableSteps, iteration, _cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)], _numberOfFrames, &_measurementsBPC[iteration % _numOfCPUThreads]);
			bufferValue = _doubleBufferOutput[iteration % (doubleBufferSize)];

			cudaStreamSynchronize(_cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)]);
			SupportFunctions::markEndProfilerCPUSection();
			SupportFunctions::markInitProfilerCPUSection("Copying", "Copying - Waiting for write buffer.");
			while ((bufferValue == 1) || (bufferValue == 2))
			{
				bufferValue = _doubleBufferOutput[iteration % (doubleBufferSize)];
			}
			int size = _HTotalBSSizeDB[iteration % _numOfCPUThreads][0];
			_framesSizes[iteration % (doubleBufferSize)] = size;
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_HBitStreamValuesDB[iteration % (doubleBufferSize)], _DBitStreamValuesDB[iteration % (doubleBufferSize)], (_HTotalBSSizeDB[iteration % _numOfCPUThreads][0]) * sizeof(unsigned short), cudaMemcpyDeviceToHost, _cStreams[(iteration % (doubleBufferSize)) + (doubleBufferSize)]));
			GPU_HANDLE_ERROR(cudaEventRecord(_cEvents[(iteration % (doubleBufferSize)) + (doubleBufferSize)], _cStreams[(iteration % (doubleBufferSize)) + (doubleBufferSize)]));

			_doubleBufferOutput[iteration % (doubleBufferSize)] = 1;
			_codedFrames.at(iteration % (doubleBufferSize)) = _HBitStreamValuesDB[iteration % (doubleBufferSize)];
			delete DWTGen;
			delete BPC;
			SupportFunctions::markEndProfilerCPUSection();
		}
	}
	else
	{
		if (_frameStructure->getIsRGB())
		{
			SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - PreSynch");
			int bufferValue;
			DWT<float, float2>* DWTGen;
			BPCCuda<float>* BPC;
			SupportFunctions::markEndProfilerCPUSection();

			SupportFunctions::markInitProfilerCPUSection("Copying Synching", "Copying - Synching");
			GPU_HANDLE_ERROR(cudaStreamWaitEvent(_cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams], _cEvents[iteration % (doubleBufferSize)], 0));
			SupportFunctions::markEndProfilerCPUSection();

			SupportFunctions::markInitProfilerCPUSection("ColorTransform", "Color Tranform Kernel");
			prepareRGBFrame(_cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)], iteration);
			SupportFunctions::markEndProfilerCPUSection();
			_doubleBufferInput[iteration % (doubleBufferSize)] = 2;

			for (int i = 0; i < 3; i++)
			{
				SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Encoding");
				DWTGen = new DWT<float, float2>(_frameStructure, _waveletType, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _quantizationSize);
				DWTGen->DWTEncode(_DImagePixelsRGBTransformedLossy[(iteration * 3 + i) % (doubleBufferSize * 3)], _DWaveletCoefficientsDBLossy[iteration % _numOfCPUThreads], _cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)]);
				SupportFunctions::markEndProfilerCPUSection();

				SupportFunctions::markInitProfilerCPUSection("BPC", "BPC");
				BPC = new BPCCuda<float>(_frameStructure, _DWaveletCoefficientsDBLossy[iteration % _numOfCPUThreads], _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);
				BPC->Code(_LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[i], _DCodeStreamValuesDB[iteration % _numOfCPUThreads], _DPrefixedArrayDB[iteration % _numOfCPUThreads], _DTempStoragePArrayDB[iteration % _numOfCPUThreads], _DSizeArrayDB[iteration % _numOfCPUThreads], _HExtraInformation, _DBitStreamValuesDB[(iteration * 3 + i) % (doubleBufferSize * 3)], _HTotalBSSizeDB[iteration % _numOfCPUThreads], _DLUTBSTableDB[iteration % _numOfCPUThreads], _HLUTBSTableSteps, iteration, _cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)], _numberOfFrames, &_measurementsBPC[iteration % _numOfCPUThreads]);
				auxiliarFrameSizes[i] = _HTotalBSSizeDB[iteration % _numOfCPUThreads][0];
				SupportFunctions::markEndProfilerCPUSection();
				delete BPC;
				delete DWTGen;
			}

			cudaStreamSynchronize(_cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)]);
			_doubleBufferInput[iteration % (doubleBufferSize)] = 3;
			bufferValue = _doubleBufferOutput[iteration % (doubleBufferSize)];
			SupportFunctions::markInitProfilerCPUSection("Copying", "Copying - Waiting for write buffer.");
			while ((bufferValue == 1) || (bufferValue == 2))
			{
				bufferValue = _doubleBufferOutput[iteration % (doubleBufferSize)];
			}
			_framesSizes[(iteration * 3) % (doubleBufferSize * 3)] = auxiliarFrameSizes[0];
			_framesSizes[(iteration * 3 + 1) % (doubleBufferSize * 3)] = auxiliarFrameSizes[1];
			_framesSizes[(iteration * 3 + 2) % (doubleBufferSize * 3)] = auxiliarFrameSizes[2];
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_HBitStreamValuesDB[(iteration * 3) % (doubleBufferSize * 3)], _DBitStreamValuesDB[(iteration * 3) % (doubleBufferSize * 3)], (_framesSizes[(iteration * 3) % (doubleBufferSize * 3)]) * sizeof(unsigned short), cudaMemcpyDeviceToHost, _cStreams[(iteration % (doubleBufferSize)) + (doubleBufferSize)]));
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_HBitStreamValuesDB[(iteration * 3 + 1) % (doubleBufferSize * 3)], _DBitStreamValuesDB[(iteration * 3 + 1) % (doubleBufferSize * 3)], (_framesSizes[(iteration * 3 + 1) % (doubleBufferSize * 3)]) * sizeof(unsigned short), cudaMemcpyDeviceToHost, _cStreams[(iteration % (doubleBufferSize)) + (doubleBufferSize)]));
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_HBitStreamValuesDB[(iteration * 3 + 2) % (doubleBufferSize * 3)], _DBitStreamValuesDB[(iteration * 3 + 2) % (doubleBufferSize * 3)], (_framesSizes[(iteration * 3 + 2) % (doubleBufferSize * 3)]) * sizeof(unsigned short), cudaMemcpyDeviceToHost, _cStreams[(iteration % (doubleBufferSize)) + (doubleBufferSize)]));
			GPU_HANDLE_ERROR(cudaEventRecord(_cEvents[(iteration % (doubleBufferSize)) + (doubleBufferSize)], _cStreams[(iteration % (doubleBufferSize)) + (doubleBufferSize)]));
			_codedFrames.at((iteration * 3) % (doubleBufferSize * 3)) = _HBitStreamValuesDB[(iteration * 3) % (doubleBufferSize * 3)];
			_codedFrames.at((iteration * 3 + 1) % (doubleBufferSize * 3)) = _HBitStreamValuesDB[(iteration * 3 + 1) % (doubleBufferSize * 3)];
			_codedFrames.at((iteration * 3 + 2) % (doubleBufferSize * 3)) = _HBitStreamValuesDB[(iteration * 3 + 2) % (doubleBufferSize * 3)];
			_doubleBufferOutput[iteration % (doubleBufferSize)] = 1;
			SupportFunctions::markEndProfilerCPUSection();
		}
		else
		{
			SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - PreSynch");
			int bufferValue;
			DWT<float, float2>* DWTGen = new DWT<float, float2>(_frameStructure, _waveletType, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _quantizationSize);
			SupportFunctions::markEndProfilerCPUSection();

			SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Synching");
			GPU_HANDLE_ERROR(cudaStreamWaitEvent(_cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams], _cEvents[iteration % (doubleBufferSize)], 0));
			SupportFunctions::markEndProfilerCPUSection();

			_doubleBufferInput[iteration % (doubleBufferSize)] = 2;
			SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Encoding");

			if (_frameStructure->getSignedOrUnsigned() == 0)
			{
				int numberOfThreadsPerBlock = 256;
				int numberOfBlocks = (int)ceil(_frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth() / numberOfThreadsPerBlock);
				offsetImage << < numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStreamDefault >> > (_DImagePixelsCharDB[iteration % (doubleBufferSize)], _DImagePixelsDBLossy[iteration % (_numOfCPUThreads)], _frameStructure->getBitDepth());
				DWTGen->DWTEncode(_DImagePixelsDBLossy[iteration % (_numOfCPUThreads)], _DWaveletCoefficientsDBLossy[iteration % _numOfCPUThreads], _cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)]);
			}
			else
			{
				DWTGen->DWTEncodeChar(_DImagePixelsCharDB[iteration % (doubleBufferSize)], _DWaveletCoefficientsDBLossy[iteration % _numOfCPUThreads], _cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)]);
			}

			SupportFunctions::markEndProfilerCPUSection();
			_doubleBufferInput[iteration % (doubleBufferSize)] = 3;

			SupportFunctions::markInitProfilerCPUSection("BPC", "BPC");
			BPCCuda<float>* BPC = new BPCCuda<float>(_frameStructure, _DWaveletCoefficientsDBLossy[iteration % _numOfCPUThreads], _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);
			BPC->Code(_LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[0], _DCodeStreamValuesDB[iteration % _numOfCPUThreads], _DPrefixedArrayDB[iteration % _numOfCPUThreads], _DTempStoragePArrayDB[iteration % _numOfCPUThreads], _DSizeArrayDB[iteration % _numOfCPUThreads], _HExtraInformation, _DBitStreamValuesDB[iteration % (doubleBufferSize)], _HTotalBSSizeDB[iteration % _numOfCPUThreads], _DLUTBSTableDB[iteration % _numOfCPUThreads], _HLUTBSTableSteps, iteration, _cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)], _numberOfFrames, &_measurementsBPC[iteration % _numOfCPUThreads]);
			bufferValue = _doubleBufferOutput[iteration % (doubleBufferSize)];
			SupportFunctions::markEndProfilerCPUSection();

			SupportFunctions::markInitProfilerCPUSection("Copying", "Copying - Waiting for write buffer");
			cudaStreamSynchronize(_cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)]);
			while ((bufferValue == 1) || (bufferValue == 2))
			{
				bufferValue = _doubleBufferOutput[iteration % (doubleBufferSize)];
			}
			int size = _HTotalBSSizeDB[iteration % _numOfCPUThreads][0];
			_framesSizes[iteration % (doubleBufferSize)] = size;
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_HBitStreamValuesDB[iteration % (doubleBufferSize)], _DBitStreamValuesDB[iteration % (doubleBufferSize)], (_HTotalBSSizeDB[iteration % _numOfCPUThreads][0]) * sizeof(unsigned short), cudaMemcpyDeviceToHost, _cStreams[(iteration % (doubleBufferSize)) + (doubleBufferSize)]));
			GPU_HANDLE_ERROR(cudaEventRecord(_cEvents[(iteration % (doubleBufferSize)) + (doubleBufferSize)], _cStreams[(iteration % (doubleBufferSize)) + (doubleBufferSize)]));

			_doubleBufferOutput[iteration % (doubleBufferSize)] = 1;
			_codedFrames.at(iteration % (doubleBufferSize)) = _HBitStreamValuesDB[iteration % (doubleBufferSize)];
			delete DWTGen;
			delete BPC;
			SupportFunctions::markEndProfilerCPUSection();
		}
	}
}

/*
* General function which initializes the CPU threads needed depending on the amount of GPU streams used.
*/
void CodingEngine::engineManager(int cType)
{
	if (cType == VIDEO)
	{
		_numOfCPUThreads = _numOfStreams;
		_numOfIOStreams = _numOfStreams * 4;

		std::future<bool> readThread;
		std::future<bool> writeThread;
		std::future<void> *processingThreads;
		processingThreads = new std::future<void>[_numOfCPUThreads];
		int doubleBufferSize = _numOfCPUThreads * 2;
		_measurementsBPC = new double[_numOfCPUThreads];
		for (int zx = 0; zx < _numOfCPUThreads; zx++)
		{
			_measurementsBPC[zx] = 0;
		}
		initLUT();
		initMemory(VIDEO);
		auto startProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
		if (_frameStructure->getIsRGB())
		{
			readThread = std::async(std::launch::async, &CodingEngine::readFileParallelRGB, this);
			writeThread = std::async(std::launch::async, &CodingEngine::writeFileParallelRGB, this);
		}
		else
		{
			readThread = std::async(std::launch::async, &CodingEngine::readFileParallel, this);
			writeThread = std::async(std::launch::async, &CodingEngine::writeFileParallel, this);
		}
		int prevPercentage = -1;
		auto accumulated = 0.0;
		auto startTime = std::chrono::high_resolution_clock::now();
		auto endTime = std::chrono::high_resolution_clock::now();
		int doubleBufferValue;
		std::cout << "Starting process";
		while (_framesC.empty())
		{
			std::cout << ".";
		}
		std::cout << std::endl;
		for (int iteration = 0; iteration < _numberOfFrames; iteration++)
		{

			if (iteration >= _numOfCPUThreads)
				processingThreads[iteration % _numOfCPUThreads].get();

			SupportFunctions::markInitProfilerCPUSection("GPU", "GPU THREAD IDLE");
			startTime = std::chrono::high_resolution_clock::now();
			doubleBufferValue = _doubleBufferInput[iteration % (doubleBufferSize)];
			while (doubleBufferValue != 1)
			{
				doubleBufferValue = _doubleBufferInput[iteration % (doubleBufferSize)];
			}
			endTime = std::chrono::high_resolution_clock::now();
			if (iteration != 0)
				accumulated = accumulated + std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

			SupportFunctions::markEndProfilerCPUSection();
			processingThreads[iteration % _numOfCPUThreads] = std::async(std::launch::async, &CodingEngine::runVideo, this, iteration);
		}
		std::cout << "Tiempo GPU en ms: " << accumulated << std::endl;
		readThread.get();
		writeThread.get();
		auto finishProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
		double elapsedTimeProcessDisregardAllocationTimings = std::chrono::duration_cast<std::chrono::duration<double>>(finishProcessDisregardAllocationTimings - startProcessDisregardAllocationTimings).count();
		std::cout << "The time spent with the app without considering allocation periods is: " << elapsedTimeProcessDisregardAllocationTimings << std::endl;
		double acumMeasurementsBPC = 0;
		for (int zy = 0; zy < _numOfCPUThreads; zy++)
		{
			acumMeasurementsBPC = acumMeasurementsBPC + _measurementsBPC[zy];
		}
		std::cout << "BPC Acum time: " << acumMeasurementsBPC << std::endl;
	}
	else if (cType == IMAGE)
	{
		_measurementsBPC = new double[1];
		initLUT();
		initMemory(IMAGE);
		runImage();
	}
}
