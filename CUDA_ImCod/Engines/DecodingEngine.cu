#include "DecodingEngine.cuh"

/*
* Memory initialization function which preallocates every memory needed in the process.
* It takes in consideration everything from Image/Video, Lossy/Lossless and GrayScale/RGB.
*/
void DecodingEngine::initMemory(bool typeOfCoding)
{
	if (typeOfCoding == VIDEO)
	{
		int numberOfStreams = _numOfIOStreams / 2;
		_DCodeStreamValuesDB = (int**)malloc(_numOfCPUThreads * sizeof(int**));
		_DSizeArrayDB = (int**)malloc(_numOfCPUThreads * sizeof(int**));
		_DPrefixedArrayDB = (int**)malloc(_numOfCPUThreads * sizeof(int**));
		_DTempStoragePArrayDB = (int**)malloc(_numOfCPUThreads * sizeof(int**));
		_HTotalBSSizeDB = (int**)malloc(_numOfCPUThreads * sizeof(int**));
		_DLUTBSTableDB = (int**)malloc(_numOfCPUThreads * sizeof(int**));
		_DWaveletCoefficientsDB = (int**)malloc(_numOfCPUThreads * sizeof(int**));
		_HSizeArrayDB = (int**)malloc(_numOfCPUThreads * sizeof(int**));

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

		_HLUTBSTableSteps = 256;
		_HBasicInformation = (int*)malloc(16 * sizeof(int));

		retrieveBasicImageInformation(_frameStructure->getName());
		
		_frameStructure->setWidth(_HBasicInformation[0] / _HBasicInformation[10]);
		_frameStructure->setHeight(_HBasicInformation[10]);
		_frameStructure->setBitDepth(_HBasicInformation[5]);
		_frameStructure->setComponents(_HBasicInformation[8]);
		_frameStructure->setBitsPerSample(_HBasicInformation[12]);
		_frameStructure->setEndianess(_HBasicInformation[11]);
		_frameStructure->setIsRGB(_HBasicInformation[9]);
		_frameStructure->setSignedOrUnsigned(_HBasicInformation[13]);
		SupportFunctions::fixImageProportions(this->_frameStructure, CBLOCK_LENGTH, CBLOCK_WIDTH);
		this->setCodingPasses(_HBasicInformation[1]);
		this->setCBHeight(_HBasicInformation[2]);
		this->setCBWidth(_HBasicInformation[3]);
		this->setWaveletLevels(_HBasicInformation[4]);
		this->setWType(_HBasicInformation[6]);
		this->setQSizes(_HBasicInformation[7]/10000.0);
		this->setNumberOfFrames(_HBasicInformation[14]);
		this->setKFactor(_HBasicInformation[15] / 1000.0);
	
		_frameSize = _frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth();
		for (int i = 1; i<_HBasicInformation[4]; ++i)
			_extraWaveletAllocation += _frameSize / (pow(4, i));

		if (_frameStructure->getIsRGB() == true)
		{
			_framesC.resize(_numOfCPUThreads * 2 * 3);
			_framesLossy.resize(_numOfCPUThreads * 2 * 3);
			_DBitStreamValuesDB = (unsigned short**)malloc(numberOfStreams * sizeof(unsigned short**) * 3);
			if (_waveletType == LOSSLESS)
			{
				_DImagePixelsRGBTransformed = new int*[3 * numberOfStreams];
				for (int i = 0; i < numberOfStreams * 3; i++)
				{
					GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformed[i], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() + _extraWaveletAllocation) * sizeof(int)));
				}
			}
			else
			{
				_DImagePixelsRGBTransformedLossy = new float*[3 * numberOfStreams];
				for (int i = 0; i < numberOfStreams * 3; i++)
				{
					GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformedLossy[i], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() + _extraWaveletAllocation) * sizeof(float)));
				}
			}
			_DImagePixelsCharDB = new unsigned char*[3 * numberOfStreams];
			_HImagePixelsCharDB = new unsigned char*[3 * numberOfStreams];
			for (int i = 0; i < numberOfStreams * 3; i++)
			{
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsCharDB[i], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(unsigned char)));
				cudaHostAlloc(&_HImagePixelsCharDB[i], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight(), 0);
				GPU_HANDLE_ERROR(cudaMalloc(&_DBitStreamValuesDB[i], _frameSize * sizeof(unsigned short)));
			}
		}
		else
		{
			_frames.resize(_numOfCPUThreads * 2);
			_framesLossy.resize(_numOfCPUThreads * 2);
			_DBitStreamValuesDB = (unsigned short**)malloc(numberOfStreams * sizeof(unsigned short**));
			if (_waveletType == LOSSLESS)
			{
				_DImagePixelsDB = (int**)malloc(numberOfStreams * sizeof(int**));
				_HImagePixelsDB = (int**)malloc(numberOfStreams * sizeof(int**));
			}

			else
			{
				_HImagePixelsDBLossy = (float**)malloc(numberOfStreams * sizeof(float**));
				_DImagePixelsDBLossy = (float**)malloc(numberOfStreams * sizeof(float**));
			}
			for (int j = 0; j < numberOfStreams; j++)
			{
				if (_waveletType == LOSSLESS)
				{
					cudaHostAlloc(&_HImagePixelsDB[j], _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight() * sizeof(int), 0);
					GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsDB[j], (_frameSize + _extraWaveletAllocation) * sizeof(int)));
				}
				else
				{
					GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsDBLossy[j], (_frameSize + _extraWaveletAllocation) * sizeof(float)));
					cudaHostAlloc(&_HImagePixelsDBLossy[j], _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight() * sizeof(float), 0);
				}
				GPU_HANDLE_ERROR(cudaMalloc(&_DBitStreamValuesDB[j], _frameSize * sizeof(unsigned short)));
			}
		}

		int storage = _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight() / (CBLOCK_WIDTH*CBLOCK_LENGTH * 2);
		//This way we make sure that for really small frames or images, the temporal storage needed by CUB is covered. For really big images or frames, the equation above is enough - tested empirically.
		if (storage < 1000)
			storage = 1000;
		for (int k = 0; k < _numOfCPUThreads; k++)
		{
			GPU_HANDLE_ERROR(cudaMalloc(&_DLUTBSTableDB[k], (_HLUTBSTableSteps + 1) * sizeof(int)));
			_HSizeArrayDB[k] = (int*)malloc((_frameSize / (CBLOCK_WIDTH * CBLOCK_LENGTH)) * sizeof(int));
			_HTotalBSSizeDB[k] = (int*)malloc(sizeof(int));
			GPU_HANDLE_ERROR(cudaMalloc(&_DTempStoragePArrayDB[k], storage));
			GPU_HANDLE_ERROR(cudaMalloc(&_DSizeArrayDB[k], (int)ceil(_frameSize / ((float)CBLOCK_WIDTH * (float)CBLOCK_LENGTH)) * sizeof(int)));
			GPU_HANDLE_ERROR(cudaMalloc(&_DPrefixedArrayDB[k], (int)ceil(_frameSize / ((float)CBLOCK_WIDTH * (float)CBLOCK_LENGTH)) * sizeof(int)));
			GPU_HANDLE_ERROR(cudaMalloc(&_DCodeStreamValuesDB[k], _frameSize * sizeof(int)));
			GPU_HANDLE_ERROR(cudaMalloc(&_DWaveletCoefficientsDB[k], _frameSize * sizeof(int)));
		}
	}
	else
	{
		_extraWaveletAllocation = 0;
		_HBasicInformation = (int*)malloc(16 * sizeof(int));

		retrieveBasicImageInformation(_frameStructure->getName());
		_frameStructure->setWidth(_HBasicInformation[0] / _HBasicInformation[10]);
		_frameStructure->setHeight(_HBasicInformation[10]);
		_frameStructure->setBitDepth(_HBasicInformation[5]);
		_frameStructure->setComponents(_HBasicInformation[8]);
		_frameStructure->setBitsPerSample(_HBasicInformation[12]);
		_frameStructure->setEndianess(_HBasicInformation[11]);
		_frameStructure->setIsRGB(_HBasicInformation[9]);
		_frameStructure->setSignedOrUnsigned(_HBasicInformation[13]);
		SupportFunctions::fixImageProportions(this->_frameStructure, CBLOCK_LENGTH, CBLOCK_WIDTH);
		_frameSize = _frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth();
		this->setCodingPasses(_HBasicInformation[1]);
		this->setCBHeight(_HBasicInformation[2]);
		this->setCBWidth(_HBasicInformation[3]);
		this->setWaveletLevels(_HBasicInformation[4]);
		this->setWType(_HBasicInformation[6]);
		this->setQSizes(_HBasicInformation[7] / 10000.0);
		this->setNumberOfFrames(_HBasicInformation[14]);
		this->setKFactor(_HBasicInformation[15]/1000.0);
 		_HLUTBSTableSteps = 256;

		for (int i = 1; i<_HBasicInformation[4]; ++i)
			_extraWaveletAllocation += _frameSize / (pow(4, i));

		if (_frameStructure->getIsRGB() == true)
		{
			int sizeOfImage = _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight();
			_bufferReadingValueRGB = new int[3];
			_bufferReadingValueRGB[0] = 0;
			_bufferReadingValueRGB[1] = 0;
			_bufferReadingValueRGB[2] = 0;
			_HBitStreamValuesRGB = new unsigned short*[3];
			_HImagePixelsCharRGB = new unsigned char*[3];
			_DImagePixelsCharRGB = new unsigned char*[3];
			cudaHostAlloc(&(_HBitStreamValuesRGB[0]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight() * sizeof(unsigned short), 0);
			cudaHostAlloc(&(_HBitStreamValuesRGB[1]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight() * sizeof(unsigned short), 0);
			cudaHostAlloc(&(_HBitStreamValuesRGB[2]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight() * sizeof(unsigned short), 0);
			cudaHostAlloc(&(_HImagePixelsCharRGB[0]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight(), 0);
			cudaHostAlloc(&(_HImagePixelsCharRGB[1]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight(), 0);
			cudaHostAlloc(&(_HImagePixelsCharRGB[2]), _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight(), 0);
			GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsCharRGB[0], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()));
			GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsCharRGB[1], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()));
			GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsCharRGB[2], _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()));

			if (_waveletType == LOSSLESS)
			{
				_DImagePixelsRGBTransformed = new int*[3];
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformed[0], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() + _extraWaveletAllocation) * sizeof(int)));
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformed[1], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() + _extraWaveletAllocation) * sizeof(int)));
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformed[2], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() + _extraWaveletAllocation) * sizeof(int)));

			}
			else
			{
				_DImagePixelsRGBTransformedLossy = new float*[3];
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformedLossy[0], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() + _extraWaveletAllocation) * sizeof(float)));
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformedLossy[1], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() + _extraWaveletAllocation) * sizeof(float)));
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsRGBTransformedLossy[2], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() + _extraWaveletAllocation) * sizeof(float)));

			}
		}
		else
		{
			cudaHostAlloc(&_HBitStreamValues, _frameSize * sizeof(unsigned short), 0);
			if (_waveletType == LOSSLESS)
			{
				cudaHostAlloc(&_HImagePixels, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(int), 0);
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixels, (_frameSize + _extraWaveletAllocation) * sizeof(int)));
			}
			else
			{
				cudaHostAlloc(&_HImagePixelsLossy, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(float), 0);
				GPU_HANDLE_ERROR(cudaMalloc(&_DImagePixelsLossy, (_frameSize + _extraWaveletAllocation) * sizeof(float)));
			}
		}

		int storage = _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight() / (CBLOCK_WIDTH*CBLOCK_LENGTH * 2);
		//This way we make sure that for really small frames or images, the temporal storage needed by CUB is covered. For really big images or frames, the equation above is enough - tested empirically.
		if (storage < 1000)
			storage = 1000;

		GPU_HANDLE_ERROR(cudaMalloc(&_DBitStreamValues, _frameSize * sizeof(unsigned short)));
		GPU_HANDLE_ERROR(cudaMalloc(&_DSizeArray, (int)ceil(_frameSize / ((float)CBLOCK_WIDTH * (float)CBLOCK_LENGTH)) * sizeof(int)));
		GPU_HANDLE_ERROR(cudaMalloc(&_DPrefixedArray, (int)ceil(_frameSize / ((float)CBLOCK_WIDTH * (float)CBLOCK_LENGTH)) * sizeof(int)));
		GPU_HANDLE_ERROR(cudaMalloc(&_DWaveletCoefficients, _frameSize * sizeof(int)));
		GPU_HANDLE_ERROR(cudaMalloc(&_DCodeStreamValues, _frameSize * sizeof(int)));
		GPU_HANDLE_ERROR(cudaMalloc(&_DLUTBSTable, (_HLUTBSTableSteps + 1) * sizeof(int)));
		GPU_HANDLE_ERROR(cudaMalloc(&_DTempStoragePArray, storage));
		_HSizeArray = (int*)malloc((_frameSize / (CBLOCK_WIDTH * CBLOCK_LENGTH)) * sizeof(int));
		_HTotalBSSize = (int*)malloc(sizeof(int));
	}
}

/*
I/O Function in charge of reading a compressed video, bitstream by bitstream, and placing it in the corresponding memory buffer. Managed by a CPU thread, it controls a
double structure memory buffer. With the aid of status flags inside the array _doubleBufferInput it knows when a buffer
becomes available for new information. This module reads only grayscale compressed videos.
Statuses for buffer queue:
0: Not used yet (init state).
1: Being copied from host to GPU.
2: Being processed by the algorithm.
3: Information already used, ready to be replaced.
*/
bool DecodingEngine::readFileParallel()
{
	bool ret = false;
	IOManager<unsigned short, ushort2> *IOM = new IOManager<unsigned short, ushort2>();
	int iter = 0;
	long long int offset = 0;
	unsigned short* frames;
	unsigned short* currentFrame;
	int bufferValue;
	cudaHostAlloc(&_frameSizes, _numberOfFrames * sizeof(int), 0);
	IOM->readBulkSizes(_frameSizes, _frameStructure, _numberOfFrames);
	int numberOfIStreams = _numOfIOStreams / 2;
	auto accumulated = 0.0;
	auto startTime = std::chrono::high_resolution_clock::now();
	auto endTime = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < numberOfIStreams; i++)
	{
		cudaHostAlloc(&frames, _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight() * sizeof(unsigned short), 0);
		_codedFrames.push_back(frames);
	}

	for (iter; iter < _numberOfFrames; iter++)
	{
		bufferValue = _doubleBufferInput[(iter) % numberOfIStreams];
		startTime = std::chrono::high_resolution_clock::now();
		while (bufferValue == 1 || (bufferValue == 2))
		{
			bufferValue = _doubleBufferInput[(iter) % numberOfIStreams];
		}
		SupportFunctions::markInitProfilerCPUSection("readFileParallel", "readFileParallel");
		endTime = std::chrono::high_resolution_clock::now();
		currentFrame = _codedFrames.at(iter % numberOfIStreams);
		
		offset = offset + IOM->loadCodedFrame(_frameStructure, currentFrame, iter, _frameSizes[iter], offset);

		GPU_HANDLE_ERROR(cudaMemcpyAsync(_DBitStreamValuesDB[iter % numberOfIStreams], currentFrame, _frameSizes[iter] * sizeof(unsigned short), cudaMemcpyHostToDevice, _cStreams[iter % numberOfIStreams]));
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
I/O Function in charge of reading a compressed video, bitstream by bitstream, and placing it in the corresponding memory buffer. Managed by a CPU thread, it controls a
double structure memory buffer. With the aid of status flags inside the array _doubleBufferInput it knows when a buffer
becomes available for new information. This module reads only RGB compressed videos.
Statuses for buffer queue:
0: Not used yet (init state).
1: Being copied from host to GPU.
2: Being processed by the algorithm.
3: Information already used, ready to be replaced.
*/
bool DecodingEngine::readFileParallelRGB()
{
	bool ret = false;
	IOManager<unsigned short, ushort2> *IOM = new IOManager<unsigned short, ushort2>();
	int iter = 0;
	long long int offset = 0;
	unsigned short* frames;
	unsigned short* currentFrame;
	int bufferValue;
	cudaHostAlloc(&_frameSizes, _numberOfFrames * sizeof(int) * 3, 0);
	IOM->readBulkSizes(_frameSizes, _frameStructure, _numberOfFrames * 3);
	int numberOfIStreams = _numOfIOStreams / 2;
	auto accumulated = 0.0;
	auto startTime = std::chrono::high_resolution_clock::now();
	auto endTime = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < numberOfIStreams * 3; i++)
	{
		cudaHostAlloc(&frames, _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight() * sizeof(unsigned short), 0);
		_codedFrames.push_back(frames);
	}

	for (iter; iter < _numberOfFrames; iter++)
	{
		bufferValue = _doubleBufferInput[(iter) % numberOfIStreams];
		startTime = std::chrono::high_resolution_clock::now();
		while (bufferValue == 1 || (bufferValue == 2))
		{
			bufferValue = _doubleBufferInput[(iter) % numberOfIStreams];
		}
		SupportFunctions::markInitProfilerCPUSection("readFileParallel", "readFileParallel");
		endTime = std::chrono::high_resolution_clock::now();
		for (int j = 0; j < 3; j++)
		{
			currentFrame = _codedFrames.at((iter * 3 + j) % (numberOfIStreams*3));
			
			offset = offset + IOM->loadCodedFrame(_frameStructure, currentFrame, iter * 3 + j, _frameSizes[iter * 3 + j], offset);
			
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_DBitStreamValuesDB[(iter * 3 + j) % (numberOfIStreams * 3)], currentFrame, _frameSizes[iter * 3 + j] * sizeof(unsigned short), cudaMemcpyHostToDevice, _cStreams[iter % numberOfIStreams]));
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

void DecodingEngine::readCompressedImage()
{
	SupportFunctions::markInitProfilerCPUSection("IO", "Disk Reading");
	//Check whether it is RGB or GreyScale for components sake.
	IOManager<unsigned short, ushort2>* IOM = new IOManager<unsigned short, ushort2>(_frameStructure->getName());
	//Compressed size unkwnown at the moment. Pending modification.
	IOM->readBitStreamFile(_HBitStreamValues, _frameSize);
	GPU_HANDLE_ERROR(cudaMemcpyAsync(_DBitStreamValues, _HBitStreamValues, _frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight() * sizeof(unsigned short), cudaMemcpyHostToDevice, cudaStreamDefault));
	//We coded the bitStream file into a variable. Now, decode time.
	SupportFunctions::markEndProfilerCPUSection();
}

bool DecodingEngine::readRGBCompressedBitStream()
{
	bool ret = false;
	SupportFunctions::markInitProfilerCPUSection("IO", "Disk Reading");
	IOManager<unsigned short, ushort2>* IOM = new IOManager<unsigned short, ushort2>(_frameStructure->getName());
	cudaHostAlloc(&_componentSizes, 3 * sizeof(int), 0);
	IOM->readBulkSizes(_componentSizes, _frameStructure, 3);
	long long int offset = 0;
	int i = 0;
	for (i; i < 3; i++)
	{
		offset = offset + IOM->loadCodedFrame(_frameStructure, _HBitStreamValuesRGB[i], i, _componentSizes[i], offset);
		_bufferReadingValueRGB[i] = 1;
	}
	//We coded the bitStream file into a variable. Now, decode time.
	if (i == 3)
		ret = true;
	SupportFunctions::markEndProfilerCPUSection();
	return ret;
}

/*
* I/O Function in charge of writing back to the disk the uncompressed frames coded by the engine. This function manages only grayscale videos,
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
bool DecodingEngine::writeFileParallel()
{
	int bufferValue = 0;
	bool ret = true;
	IOManager<int, int2> *IOM = new IOManager<int, int2>();
	int* decodedFrame;
	int* frame;
	IOM->replaceExistingFile(_outputFile);
	int numberOfOStreams = _numOfIOStreams / 2;
	auto accumulated = 0.0;
	auto startTime = std::chrono::high_resolution_clock::now();
	auto endTime = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < _numberOfFrames; i++)
	{

		bufferValue = _doubleBufferOutput[(i) % numberOfOStreams];
		startTime = std::chrono::high_resolution_clock::now();
		while (bufferValue != 1)
		{
			bufferValue = _doubleBufferOutput[(i) % numberOfOStreams];
		}
		SupportFunctions::markInitProfilerCPUSection("writeCodedFileParallel", "writeCodedFile");
		endTime = std::chrono::high_resolution_clock::now();
		GPU_HANDLE_ERROR(cudaEventSynchronize(_cEvents[(i % numberOfOStreams) + numberOfOStreams]));
		decodedFrame = _frames.at(i % numberOfOStreams);
		_doubleBufferOutput[(i) % numberOfOStreams] = 2;
		IOM->writeDecodedFrame(_frameStructure, decodedFrame, i, _outputFile);
		_doubleBufferOutput[(i) % numberOfOStreams] = 3;
		SupportFunctions::markEndProfilerCPUSection();
	}

	std::cout << "Acumulado de Escritura: " << accumulated << std::endl;
	delete IOM;
	return ret;
}

/*
* I/O Function in charge of writing back to the disk the uncompressed frames coded by the engine. This function manages only RGB videos,
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
bool DecodingEngine::writeFileParallelRGB()
{
	int bufferValue = 0;
	bool ret = true;
	IOManager<int, int2> *IOM = new IOManager<int, int2>();
	unsigned char* decodedFrame;
	int* frame;
	IOM->replaceExistingFile(_outputFile);
	int numberOfOStreams = _numOfIOStreams / 2;
	auto accumulated = 0.0;
	auto startTime = std::chrono::high_resolution_clock::now();
	auto endTime = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < _numberOfFrames; i++)
	{

		bufferValue = _doubleBufferOutput[(i) % numberOfOStreams];
		startTime = std::chrono::high_resolution_clock::now();
		while (bufferValue != 1)
		{
			bufferValue = _doubleBufferOutput[(i) % numberOfOStreams];
		}
		SupportFunctions::markInitProfilerCPUSection("writeCodedFileParallel", "writeCodedFile");
		endTime = std::chrono::high_resolution_clock::now();
		_doubleBufferOutput[(i) % numberOfOStreams] = 2;
		cudaStreamSynchronize(_cStreams[(i % numberOfOStreams) + numberOfOStreams]);
		for (int j = 0; j < 3; j++)
		{
			decodedFrame = _framesC.at((i*3+j) % (numberOfOStreams*3));
			IOM->writeDecodedFrameComponentUChar(_frameStructure, decodedFrame, i*3+j, _outputFile);
		}
		
		_doubleBufferOutput[(i) % numberOfOStreams] = 3;
		SupportFunctions::markEndProfilerCPUSection();
	}

	std::cout << "Acumulado de Escritura: " << accumulated << std::endl;
	delete IOM;
	return ret;
}


/*
* I/O Function in charge of writing back to the disk the uncompressed frames coded by the engine. This function manages only grayscale videos,
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
bool DecodingEngine::writeFileParallelLossy()
{
	int bufferValue = 0;
	bool ret = true;
	IOManager<float, float2> *IOM = new IOManager<float, float2>();
	float* decodedFrameLossy;
	int* frame;
	IOM->replaceExistingFile(_outputFile);
	int numberOfOStreams = _numOfIOStreams / 2;
	auto accumulated = 0.0;
	auto startTime = std::chrono::high_resolution_clock::now();
	auto endTime = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < _numberOfFrames; i++)
	{

		bufferValue = _doubleBufferOutput[(i) % numberOfOStreams];
		startTime = std::chrono::high_resolution_clock::now();
		while (bufferValue != 1)
		{
			bufferValue = _doubleBufferOutput[(i) % numberOfOStreams];
		}
		SupportFunctions::markInitProfilerCPUSection("writeCodedFileParallel", "writeCodedFile");
		endTime = std::chrono::high_resolution_clock::now();
		GPU_HANDLE_ERROR(cudaEventSynchronize(_cEvents[(i % numberOfOStreams) + numberOfOStreams]));
		decodedFrameLossy = _framesLossy.at(i % numberOfOStreams);
		_doubleBufferOutput[(i) % numberOfOStreams] = 2;
		IOM->writeDecodedFrame(_frameStructure, decodedFrameLossy, i, _outputFile);
		_doubleBufferOutput[(i) % numberOfOStreams] = 3;
		SupportFunctions::markEndProfilerCPUSection();
	}
	std::cout << "Acumulado de Escritura: " << accumulated << std::endl;
	delete IOM;
	return ret;
}

void DecodingEngine::writeDecompressedImage()
{
	if (_waveletType == LOSSLESS)
	{
		IOManager<int, int2>* IOM = new IOManager<int, int2>(_frameStructure->getName());
		IOM->writeImage(_HImagePixels, _frameStructure->getWidth(), _frameStructure->getHeight(), _frameStructure->getBitDepth(), _outputFile);
	}
	else
	{
		IOManager<float, float2>* IOM = new IOManager<float, float2>(_frameStructure->getName());
		IOM->writeImage(_HImagePixelsLossy, _frameStructure->getWidth(), _frameStructure->getHeight(), _frameStructure->getBitDepth(), _outputFile);
	}
}

void DecodingEngine::writeRGBImage()
{
	IOManager<int, int2> *IOM = new IOManager<int, int2>();
	IOM->replaceExistingFile(_outputFile);
	SupportFunctions::markInitProfilerCPUSection("writeCodedFile", "writeCodedFile");
	IOM->writeDecodedFrameUChar(_frameStructure, _HImagePixelsCharRGB, _outputFile);
	SupportFunctions::markEndProfilerCPUSection();

}

/*
* Retrieves the information from a coded video / image and extracts the side information needed to decode it.
*/
void DecodingEngine::getExtraInformation()
{
	_HBasicInformation[0] = (_HExtraInformation[0] | (_HExtraInformation[1] << 16)); //Image Size
	_HBasicInformation[1] = (_HExtraInformation[2] & 1) == 1 ? 3 : 2; // Coding Passes
	_HBasicInformation[2] = ((_HExtraInformation[2] >> 1) & ((1 << 7) - 1)); // DWT Height
	_HBasicInformation[3] = ((_HExtraInformation[2] >> 8) & ((1 << 7) - 1)); // DWT Width
	_HBasicInformation[4] = (((_HExtraInformation[2] >> 15 & 1)) | (_HExtraInformation[3] & 7) << 1); // WLevels
	_HBasicInformation[5] = (_HExtraInformation[3] >> 3) & ((1 << 7) - 1); // BitDepth
	_HBasicInformation[6] = (_HExtraInformation[3] >> 10) & 1; // WType
	_HBasicInformation[7] = (((_HExtraInformation[3] >> 11) & 31) | (_HExtraInformation[4] & 511) << 5); // QSize
	_HBasicInformation[8] = ((_HExtraInformation[4] >> 9) & 127) | ((_HExtraInformation[5] & 127) << 9); // Components
	_HBasicInformation[9] = (_HExtraInformation[5] >> 7) & 1; // RGB
	_HBasicInformation[10] = ((_HExtraInformation[5] >> 8) & 255) | ((_HExtraInformation[6] & 255) << 8); // Image Height (used to recover width as well from the total size)
	_HBasicInformation[11] = ((_HExtraInformation[6] >> 8) & 1); // Endianess
	_HBasicInformation[12] = (_HExtraInformation[6] >> 9) & ((1<<5)-1); // BPS
	_HBasicInformation[13] = (_HExtraInformation[6] >> 14) & 1; // Signed/Unsigned
	_HBasicInformation[14] = (((_HExtraInformation[6] >> 15) & 1) | _HExtraInformation[7] << 1); // Number of frames
	_HBasicInformation[15] = _HExtraInformation[8]; // KFactor value
}

void DecodingEngine::retrieveBasicImageInformation(std::string inputFile)
{
	IOManager<unsigned short, ushort2> *IOM = new IOManager<unsigned short, ushort2>();
	_HExtraInformation = (unsigned short*)malloc(9 * sizeof(unsigned short));
	IOM->loadBasicInfo(_HExtraInformation, 9, inputFile);
	getExtraInformation();
	delete IOM;
}

/*
* Kernel which launches the reverse lossless color transformation, changing from YCbCr color space to RGB color space. It also recovers the size of the samples by applying an offset if the data type is unsigned.
*/ 
__global__ void RGBTransformLossless(int* inputR, int* inputG, int* inputB, unsigned char* outputR, unsigned char* outputG, unsigned char* outputB, int bitdepth, bool uSigned)
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
	componentR = (float)inputR[threadId];
	componentG = (float)inputG[threadId];
	componentB = (float)inputB[threadId];
	int componentGTransformed = componentR - (floor(((componentG) + (componentB)) / 4.0));
	int componentRTransformed = componentB + componentGTransformed;
	int componentBTransformed = componentG + componentGTransformed;

	componentRTransformed = componentRTransformed + offset;
	componentGTransformed = componentGTransformed + offset;
	componentBTransformed = componentBTransformed + offset;
	outputR[threadId] = (unsigned char)max(min(componentRTransformed, 255), 0);
	outputG[threadId] = (unsigned char)max(min(componentGTransformed, 255), 0);
	outputB[threadId] = (unsigned char)max(min(componentBTransformed, 255), 0);

}
/*
* Kernel which launches the reverse lossy color transformation, changing from YCbCr color space to RGB color space.It also recovers the size of the samples by applying an offset if the data type is unsigned.
*/
__global__ void RGBTransformLossy(float* inputR, float* inputG, float* inputB, unsigned char* outputR, unsigned char* outputG, unsigned char* outputB, int bitdepth, bool uSigned)
{
	int offset = 0;
	if (uSigned == false)
	{
		offset = 1 << (bitdepth - 1);
	}
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	float componentR = (float)inputR[threadId];
	float componentG = (float)inputG[threadId];
	float componentB = (float)inputB[threadId];
	int componentRTransformed = __float2int_rn((_irreversibleColorTransformBackward[0][0] * componentR + _irreversibleColorTransformBackward[0][1] * (componentG) + _irreversibleColorTransformBackward[0][2] * (componentB)) + 0.01f);
	int componentGTransformed = __float2int_rn((_irreversibleColorTransformBackward[1][0] * componentR + _irreversibleColorTransformBackward[1][1] * (componentG) + _irreversibleColorTransformBackward[1][2] * (componentB)) + 0.01f);
	int componentBTransformed = __float2int_rn((_irreversibleColorTransformBackward[2][0] * componentR + _irreversibleColorTransformBackward[2][1] * (componentG) + _irreversibleColorTransformBackward[2][2] * (componentB)) + 0.01f);

	componentRTransformed = componentRTransformed + offset;
	componentGTransformed = componentGTransformed + offset;
	componentBTransformed = componentBTransformed + offset;
	outputR[threadId] = (unsigned char)max(min(componentRTransformed, 255), 0);
	outputG[threadId] = (unsigned char)max(min(componentGTransformed, 255), 0);
	outputB[threadId] = (unsigned char)max(min(componentBTransformed, 255), 0);
}

/*
* Host function which launches the color transformation for images
*/
void DecodingEngine::prepareRGBImage(cudaStream_t mainStream)
{

	if (_waveletType == LOSSLESS)
	{
		int numberOfThreadsPerBlock = 256;
		int numberOfBlocks = (int)ceil(_frameStructure->getHeight() * _frameStructure->getWidth() / numberOfThreadsPerBlock);
		RGBTransformLossless << <numberOfBlocks, numberOfThreadsPerBlock, 0, mainStream >> >	(_DImagePixelsRGBTransformed[0] + _extraWaveletAllocation, _DImagePixelsRGBTransformed[1] + _extraWaveletAllocation, _DImagePixelsRGBTransformed[2] + _extraWaveletAllocation, _DImagePixelsCharRGB[0], _DImagePixelsCharRGB[1], _DImagePixelsCharRGB[2], _frameStructure->getBitDepth(), _frameStructure->getSignedOrUnsigned());
		cudaStreamSynchronize(mainStream);
		KERNEL_ERROR_HANDLER;
		for (int i = 0; i < 3; i++)
			GPU_HANDLE_ERROR(cudaMemcpy(_HImagePixelsCharRGB[i], _DImagePixelsCharRGB[i], _frameStructure->getWidth() * _frameStructure->getHeight() * sizeof(char), cudaMemcpyDeviceToHost));
	}
	else
	{
		int numberOfThreadsPerBlock = 256;
		int numberOfBlocks = (int)ceil(_frameStructure->getHeight() * _frameStructure->getWidth() / numberOfThreadsPerBlock);
		RGBTransformLossy << <numberOfBlocks, numberOfThreadsPerBlock, 0, mainStream >> >	(_DImagePixelsRGBTransformedLossy[0] + _extraWaveletAllocation, _DImagePixelsRGBTransformedLossy[1] + _extraWaveletAllocation, _DImagePixelsRGBTransformedLossy[2] + _extraWaveletAllocation, _DImagePixelsCharRGB[0], _DImagePixelsCharRGB[1], _DImagePixelsCharRGB[2], _frameStructure->getBitDepth(), _frameStructure->getSignedOrUnsigned());
		cudaStreamSynchronize(mainStream);
		KERNEL_ERROR_HANDLER;
		for (int i = 0; i < 3; i++)
			GPU_HANDLE_ERROR(cudaMemcpy(_HImagePixelsCharRGB[i], _DImagePixelsCharRGB[i], _frameStructure->getWidth() * _frameStructure->getHeight() * sizeof(char), cudaMemcpyDeviceToHost));
	}
}

/*
* Host function which launches the color transformation for videos
*/
void DecodingEngine::prepareRGBFrame(cudaStream_t mainStream, int iteration)
{
	if (_waveletType == LOSSLESS)
	{
		int numberOfThreadsPerBlock = 256;
		int numberOfBlocks = (int)ceil(_frameStructure->getHeight() * _frameStructure->getWidth() / numberOfThreadsPerBlock);
		RGBTransformLossless << <numberOfBlocks, numberOfThreadsPerBlock, 0, mainStream >> >	(_DImagePixelsRGBTransformed[(iteration * 3) % (_numOfIOStreams / 2 * 3)] + _extraWaveletAllocation, _DImagePixelsRGBTransformed[(iteration * 3 + 1) % (_numOfIOStreams / 2 * 3)] + _extraWaveletAllocation, _DImagePixelsRGBTransformed[(iteration * 3 + 2) % (_numOfIOStreams / 2 * 3)] + _extraWaveletAllocation, _DImagePixelsCharDB[(iteration*3) % (_numOfIOStreams / 2 * 3)], _DImagePixelsCharDB[(iteration*3+1) % (_numOfIOStreams / 2 * 3)], _DImagePixelsCharDB[(iteration*3+2)%(_numOfIOStreams / 2 * 3)], _frameStructure->getBitDepth(), _frameStructure->getSignedOrUnsigned());
		cudaStreamSynchronize(mainStream);
		KERNEL_ERROR_HANDLER;
	}
	else
	{
		int numberOfThreadsPerBlock = 256;
		int numberOfBlocks = (int)ceil(_frameStructure->getHeight() * _frameStructure->getWidth() / numberOfThreadsPerBlock);
		RGBTransformLossy << <numberOfBlocks, numberOfThreadsPerBlock, 0, mainStream >> >	(_DImagePixelsRGBTransformedLossy[(iteration * 3) % (_numOfIOStreams / 2 * 3)] + _extraWaveletAllocation, _DImagePixelsRGBTransformedLossy[(iteration * 3 + 1) % (_numOfIOStreams / 2 * 3)] + _extraWaveletAllocation, _DImagePixelsRGBTransformedLossy[(iteration * 3 + 2) % (_numOfIOStreams / 2 * 3)] + _extraWaveletAllocation, _DImagePixelsCharDB[(iteration * 3) % (_numOfIOStreams / 2 * 3)], _DImagePixelsCharDB[(iteration * 3 + 1) % (_numOfIOStreams / 2 * 3)], _DImagePixelsCharDB[(iteration * 3 + 2) % (_numOfIOStreams / 2 * 3)], _frameStructure->getBitDepth(), _frameStructure->getSignedOrUnsigned());
		cudaStreamSynchronize(mainStream);
		KERNEL_ERROR_HANDLER;
	}
}

/*
* Transforms the image taken into the proper data type.
*/
__global__ void removeOffsetAndApplyMaxMinLossy(float *data, int bitDepth, int signedOrUnsigned)
{
	int offset = 0;
	if (signedOrUnsigned == 0)
		offset = 1 << (bitDepth - 1);

	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	//0.01f used to avoid cases in which CUDA is not rounding correctly x.5 values. For example, 4.5 is rounded to 4 instead of 5, but 4.51 is correctly rounded to 5.
	data[threadId] = fmaxf(fminf(__float2int_rn(data[threadId] + (offset) + 0.01f), 255.0f), 0.0f);
}

/*
* Transforms the image taken into the proper data type.
*/
__global__ void removeOffsetAndApplyMaxMin(int *data, int bitDepth, int signedOrUnsigned)
{
	int offset = 0;
	if (signedOrUnsigned == 0)
		offset = 1 << (bitDepth - 1);

	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	//0.01f used to avoid cases in which CUDA is not rounding correctly x.5 values. For example, 4.5 is rounded to 4 instead of 5, but 4.51 is correctly rounded to 5.
	data[threadId] = max(min(data[threadId] + (offset), 255), 0);
}

/*
* Function which manages the general flow of processing instruction to decode images, either in lossy/lossless or grayscale/RGB.
*/
void DecodingEngine::runImage()
{
	if (_waveletType == LOSSLESS)
	{
		if (_frameStructure->getIsRGB())
		{
			std::future<bool> readThread;
			readRGBCompressedBitStream();
			int bufferValue = 0;
			auto startProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
			for (int i = 0; i < 3; i++)
			{
				bufferValue = _bufferReadingValueRGB[i];
				while (bufferValue == 0)
				{
					bufferValue = _bufferReadingValueRGB[i];
				}
				GPU_HANDLE_ERROR(cudaMemcpyAsync(_DBitStreamValues, _HBitStreamValuesRGB[i], _componentSizes[i] * sizeof(unsigned short), cudaMemcpyHostToDevice, cudaStreamDefault));
				SupportFunctions::markInitProfilerCPUSection("BPC", "BPC - Decoding");
				BPCCuda<unsigned short>* BPC = new BPCCuda<unsigned short>(_frameStructure, _HBitStreamValuesRGB[i], _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);
				BPC->Decode(_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight(), _LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[i], _DPrefixedArray, _DSizeArray, _HBasicInformation, _DTempStoragePArray, _DBitStreamValues, _DCodeStreamValues, _HSizeArray, _HTotalBSSize, _DWaveletCoefficients, cudaStreamDefault, _HLUTBSTableSteps, _DLUTBSTable, &_measurementsBPC[0]);
				SupportFunctions::markEndProfilerCPUSection();
				SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Decoding");
				DWT<int, int2>* DWTGen = new DWT<int, int2>(_frameStructure, _waveletType, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _quantizationSize);
				DWTGen->DWTDecode(_DWaveletCoefficients, _DImagePixelsRGBTransformed[i], cudaStreamDefault);
				SupportFunctions::markEndProfilerCPUSection();
			}
			SupportFunctions::markInitProfilerCPUSection("ColorTransform", "Color Tranform Kernel");
			prepareRGBImage(cudaStreamDefault);
			SupportFunctions::markEndProfilerCPUSection();
			auto finishProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
			double elapsedTimeProcessDisregardAllocationTimings = std::chrono::duration_cast<std::chrono::duration<double>>(finishProcessDisregardAllocationTimings - startProcessDisregardAllocationTimings).count();
			std::cout << "The time spent with the app without considering allocation periods and I/O is: " << elapsedTimeProcessDisregardAllocationTimings << std::endl;
			writeRGBImage();
			std::cout << "BPC acum time is: " << _measurementsBPC[0] << std::endl;
		}
		else
		{
			this->readCompressedImage();
			SupportFunctions::markInitProfilerCPUSection("BPC", "BPC - Decoding");
			BPCCuda<unsigned short>* BPC = new BPCCuda<unsigned short>(_frameStructure, _HBitStreamValues, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);
			BPC->Decode(_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight(), _LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[0], _DPrefixedArray, _DSizeArray, _HBasicInformation, _DTempStoragePArray, _DBitStreamValues, _DCodeStreamValues, _HSizeArray, _HTotalBSSize, _DWaveletCoefficients, cudaStreamDefault, _HLUTBSTableSteps, _DLUTBSTable, &_measurementsBPC[0]);
			SupportFunctions::markEndProfilerCPUSection();
			SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Decoding");
			DWT<int, int2>* DWTGen = new DWT<int, int2>(_frameStructure, _waveletType, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _quantizationSize);
			DWTGen->DWTDecode(_DWaveletCoefficients, _DImagePixels, cudaStreamDefault);
			SupportFunctions::markEndProfilerCPUSection();

			int numberOfThreadsPerBlock = 256;
			int numberOfBlocks = (int)ceil(_frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth() / numberOfThreadsPerBlock);
			removeOffsetAndApplyMaxMin << < numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStreamDefault >> > (_DImagePixels + _extraWaveletAllocation, _frameStructure->getBitDepth(), _frameStructure->getSignedOrUnsigned());
			cudaStreamSynchronize(cudaStreamDefault);
			KERNEL_ERROR_HANDLER;
			SupportFunctions::markInitProfilerCPUSection("IO", "Disk Writing");
			GPU_HANDLE_ERROR(cudaMemcpy(_HImagePixels, _DImagePixels + _extraWaveletAllocation, (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()) * sizeof(int), cudaMemcpyDeviceToHost));
			this->writeDecompressedImage();
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
			std::future<bool> readThread;
			this->readRGBCompressedBitStream();
			auto startProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
			int bufferValue = 0;
			for (int i = 0; i < 3; i++)
			{
				bufferValue = _bufferReadingValueRGB[i];
				while (bufferValue == 0)
				{
					bufferValue = _bufferReadingValueRGB[i];
				}
				GPU_HANDLE_ERROR(cudaMemcpyAsync(_DBitStreamValues, _HBitStreamValuesRGB[i], _componentSizes[i] * sizeof(unsigned short), cudaMemcpyHostToDevice, cudaStreamDefault));
				SupportFunctions::markInitProfilerCPUSection("BPC", "BPC - Decoding");
				BPCCuda<unsigned short>* BPC = new BPCCuda<unsigned short>(_frameStructure, _HBitStreamValuesRGB[i], _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);
				BPC->Decode(_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight(), _LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[i], _DPrefixedArray, _DSizeArray, _HBasicInformation, _DTempStoragePArray, _DBitStreamValues, _DCodeStreamValues, _HSizeArray, _HTotalBSSize, _DWaveletCoefficients, cudaStreamDefault, _HLUTBSTableSteps, _DLUTBSTable, &_measurementsBPC[0]);
				SupportFunctions::markEndProfilerCPUSection();
				SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Decoding");
				DWT<float, float2>* DWTGen = new DWT<float, float2>(_frameStructure, _waveletType, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _quantizationSize);
				DWTGen->DWTDecode(_DWaveletCoefficients, _DImagePixelsRGBTransformedLossy[i], cudaStreamDefault);
				SupportFunctions::markEndProfilerCPUSection();
			}
			SupportFunctions::markInitProfilerCPUSection("ColorTransform", "Color Tranform Kernel");
			prepareRGBImage(cudaStreamDefault);
			SupportFunctions::markEndProfilerCPUSection();
			auto finishProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
			double elapsedTimeProcessDisregardAllocationTimings = std::chrono::duration_cast<std::chrono::duration<double>>(finishProcessDisregardAllocationTimings - startProcessDisregardAllocationTimings).count();
			std::cout << "The time spent with the app without considering allocation periods and I/O is: " << elapsedTimeProcessDisregardAllocationTimings << std::endl;

			writeRGBImage();
			std::cout << "BPC acum time is: " << _measurementsBPC[0] << std::endl;
		}
		else
		{
			this->readCompressedImage();
			SupportFunctions::markInitProfilerCPUSection("BPC", "BPC - Decoding");
			BPCCuda<unsigned short>* BPC = new BPCCuda<unsigned short>(_frameStructure, _HBitStreamValues, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);
			BPC->Decode(_frameStructure->getAdaptedWidth()*_frameStructure->getHeight(), _LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[0], _DPrefixedArray, _DSizeArray, _HBasicInformation, _DTempStoragePArray, _DBitStreamValues, _DCodeStreamValues, _HSizeArray, _HTotalBSSize, _DWaveletCoefficients, cudaStreamDefault, _HLUTBSTableSteps, _DLUTBSTable, &_measurementsBPC[0]);
			SupportFunctions::markEndProfilerCPUSection();

			SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Decoding");
			
			DWT<float, float2>* DWTGen = new DWT<float, float2>(_frameStructure, _waveletType, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _quantizationSize);
			DWTGen->DWTDecode(_DWaveletCoefficients, _DImagePixelsLossy, cudaStreamDefault);
			SupportFunctions::markEndProfilerCPUSection();
			int numberOfThreadsPerBlock = 256;
			int numberOfBlocks = (int)ceil(_frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth() / numberOfThreadsPerBlock);

			removeOffsetAndApplyMaxMinLossy << < numberOfBlocks, numberOfThreadsPerBlock, 0, cudaStreamDefault >> > (_DImagePixelsLossy + _extraWaveletAllocation, _frameStructure->getBitDepth(), _frameStructure->getSignedOrUnsigned());
			cudaStreamSynchronize(cudaStreamDefault);
			KERNEL_ERROR_HANDLER;
			SupportFunctions::markInitProfilerCPUSection("IO", "Disk Writing");
			GPU_HANDLE_ERROR(cudaMemcpy(_HImagePixelsLossy, _DImagePixelsLossy + _extraWaveletAllocation, (_frameStructure->getAdaptedWidth()*_frameStructure->getHeight()) * sizeof(float), cudaMemcpyDeviceToHost));
			this->writeDecompressedImage();
			delete DWTGen;
			delete BPC;
			SupportFunctions::markEndProfilerCPUSection();
			std::cout << "BPC acum time is: " << _measurementsBPC[0] << std::endl;
		}
	}
}

/*
* Function which manages the general flow of processing instruction to decode videos, either in lossy/lossless or grayscale/RGB.
*/
void DecodingEngine::runVideo(int iteration)
{
	int doubleBufferSize = _numOfIOStreams / 2;
	if (_waveletType == LOSSLESS)
	{
		bool firstIter = false;
		if (iteration < _numOfCPUThreads)
			firstIter = true;

		int bufferValue;

		if (_frameStructure->getIsRGB())
		{
			BPCCuda<unsigned short>* BPC;
			DWT<int, int2>* DWTGen;
			_doubleBufferInput[iteration % doubleBufferSize] = 2;
			SupportFunctions::markInitProfilerCPUSection("Copying Synching", "Copying - Synching");
			GPU_HANDLE_ERROR(cudaStreamWaitEvent(_cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams], _cEvents[iteration % doubleBufferSize], 0));
			SupportFunctions::markEndProfilerCPUSection();

			for (int i = 0; i < 3; i++)
			{
				SupportFunctions::markInitProfilerCPUSection("BPC", "BPC");
				BPC = new BPCCuda<unsigned short>(_frameStructure, _codedFrames.at((iteration *3 +i) % (doubleBufferSize*3)), _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);
				
				BPC->Decode(_frameSizes[(iteration*3+i)], _LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[i], _DPrefixedArrayDB[iteration % _numOfCPUThreads], _DSizeArrayDB[iteration % _numOfCPUThreads], _HBasicInformation, _DTempStoragePArrayDB[iteration % _numOfCPUThreads], _DBitStreamValuesDB[(iteration * 3 + i) % (doubleBufferSize * 3)], _DCodeStreamValuesDB[iteration % _numOfCPUThreads], _HSizeArrayDB[iteration % _numOfCPUThreads], _HTotalBSSizeDB[iteration % _numOfCPUThreads], _DWaveletCoefficientsDB[iteration % _numOfCPUThreads], _cStreams[iteration % _numOfCPUThreads + _numOfIOStreams], _HLUTBSTableSteps, _DLUTBSTableDB[iteration % _numOfCPUThreads], &_measurementsBPC[iteration % _numOfCPUThreads]);
				
				SupportFunctions::markEndProfilerCPUSection();
				SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Decoding");

				DWTGen = new DWT<int, int2>(_frameStructure, _waveletType, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _quantizationSize);
				DWTGen->DWTDecode(_DWaveletCoefficientsDB[iteration % _numOfCPUThreads], _DImagePixelsRGBTransformed[(iteration * 3 + i) % (doubleBufferSize*3)], _cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams]);
				SupportFunctions::markEndProfilerCPUSection();
			}
			SupportFunctions::markInitProfilerCPUSection("ColorTransform", "Color Tranform Kernel");
			cudaStreamSynchronize(_cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams]);
			prepareRGBFrame(_cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)], iteration);
			_doubleBufferInput[iteration % doubleBufferSize] = 3;
			SupportFunctions::markEndProfilerCPUSection();
			SupportFunctions::markInitProfilerCPUSection("Copying", "Copying - Waiting for write buffer.");
			bufferValue = _doubleBufferOutput[iteration % doubleBufferSize];
			while ((bufferValue == 1) || (bufferValue == 2))
			{
				bufferValue = _doubleBufferOutput[iteration % doubleBufferSize];
			}
			SupportFunctions::markEndProfilerCPUSection();
			cudaStreamSynchronize(_cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams]);
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_HImagePixelsCharDB[(iteration * 3) % (doubleBufferSize*3)], _DImagePixelsCharDB[(iteration * 3) % (doubleBufferSize*3)], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()) * sizeof(char), cudaMemcpyDeviceToHost, _cStreams[(iteration % doubleBufferSize) + doubleBufferSize]));
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_HImagePixelsCharDB[(iteration * 3 + 1) % (doubleBufferSize * 3)], _DImagePixelsCharDB[(iteration*3+1) % (doubleBufferSize*3)],(_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()) * sizeof(char), cudaMemcpyDeviceToHost, _cStreams[(iteration % doubleBufferSize) + doubleBufferSize]));
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_HImagePixelsCharDB[(iteration * 3 + 2) % (doubleBufferSize * 3)], _DImagePixelsCharDB[(iteration*3+2) % (doubleBufferSize*3)], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()) * sizeof(char), cudaMemcpyDeviceToHost, _cStreams[(iteration % doubleBufferSize) + doubleBufferSize]));
			GPU_HANDLE_ERROR(cudaEventRecord(_cEvents[(iteration % doubleBufferSize) + doubleBufferSize], _cStreams[(iteration % doubleBufferSize) + doubleBufferSize]));

			_framesC.at((iteration * 3 ) % (doubleBufferSize*3)) = _HImagePixelsCharDB[(iteration*3)% (doubleBufferSize*3)];
			_framesC.at((iteration * 3 + 1) % (doubleBufferSize*3)) = _HImagePixelsCharDB[(iteration * 3 + 1) % (doubleBufferSize*3)];
			_framesC.at((iteration * 3 + 2) % (doubleBufferSize*3)) = _HImagePixelsCharDB[(iteration * 3 + 2) % (doubleBufferSize*3)];
			_doubleBufferOutput[iteration % doubleBufferSize] = 1;
			delete DWTGen;
			delete BPC;
		}
		else
		{
			BPCCuda<unsigned short>* BPC = new BPCCuda<unsigned short>(_frameStructure, _codedFrames.at(iteration % doubleBufferSize), _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);
			GPU_HANDLE_ERROR(cudaStreamWaitEvent(_cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams], _cEvents[iteration % doubleBufferSize], 0));
			_doubleBufferInput[iteration % doubleBufferSize] = 2;
			BPC->Decode(_frameSizes[iteration], _LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[0], _DPrefixedArrayDB[iteration % _numOfCPUThreads], _DSizeArrayDB[iteration % _numOfCPUThreads], _HBasicInformation, _DTempStoragePArrayDB[iteration % _numOfCPUThreads], _DBitStreamValuesDB[iteration % doubleBufferSize], _DCodeStreamValuesDB[iteration % _numOfCPUThreads], _HSizeArrayDB[iteration % _numOfCPUThreads], _HTotalBSSizeDB[iteration % _numOfCPUThreads], _DWaveletCoefficientsDB[iteration % _numOfCPUThreads], _cStreams[iteration % _numOfCPUThreads + _numOfIOStreams], _HLUTBSTableSteps, _DLUTBSTableDB[iteration % _numOfCPUThreads], &_measurementsBPC[iteration % _numOfCPUThreads]);

			_doubleBufferInput[iteration % doubleBufferSize] = 3;
			DWT<int, int2>* DWTGen = new DWT<int, int2>(_frameStructure, _waveletType, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _quantizationSize);
			DWTGen->DWTDecode(_DWaveletCoefficientsDB[iteration % _numOfCPUThreads], _DImagePixelsDB[iteration % doubleBufferSize], _cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams]);
			int numberOfThreadsPerBlock = 256;
			int numberOfBlocks = (int)ceil(_frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth() / numberOfThreadsPerBlock);

			removeOffsetAndApplyMaxMin << < numberOfBlocks, numberOfThreadsPerBlock, 0, _cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams] >> > (_DImagePixelsDB[iteration % doubleBufferSize] + _extraWaveletAllocation, _frameStructure->getBitDepth(), _frameStructure->getSignedOrUnsigned());
			
			bufferValue = _doubleBufferOutput[iteration % doubleBufferSize];
			while ((bufferValue == 1) || (bufferValue == 2))
			{
				bufferValue = _doubleBufferOutput[iteration % doubleBufferSize];
			}
			_frameSizes[iteration % doubleBufferSize] = _frameStructure->getAdaptedWidth() * _frameStructure->getAdaptedHeight();
			cudaStreamSynchronize(_cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams]);
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_HImagePixelsDB[iteration % doubleBufferSize], _DImagePixelsDB[iteration % doubleBufferSize] + _extraWaveletAllocation, (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()) * sizeof(int), cudaMemcpyDeviceToHost, _cStreams[(iteration % doubleBufferSize) + doubleBufferSize]));
			GPU_HANDLE_ERROR(cudaEventRecord(_cEvents[(iteration % doubleBufferSize) + doubleBufferSize], _cStreams[(iteration % doubleBufferSize) + doubleBufferSize]));
			_doubleBufferOutput[iteration % doubleBufferSize] = 1;

			_frames.at(iteration % doubleBufferSize) = _HImagePixelsDB[iteration % doubleBufferSize];
			delete DWTGen;
			delete BPC;
		}
		
	}
	else
	{
		bool firstIter = false;
		if (iteration < _numOfCPUThreads)
			firstIter = true;

		int bufferValue;
		if (_frameStructure->getIsRGB())
		{
			BPCCuda<unsigned short>* BPC;
			DWT<float, float2>* DWTGen;
			_doubleBufferInput[iteration % doubleBufferSize] = 2;
			SupportFunctions::markInitProfilerCPUSection("Copying Synching", "Copying - Synching");
			GPU_HANDLE_ERROR(cudaStreamWaitEvent(_cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams], _cEvents[iteration % doubleBufferSize], 0));
			SupportFunctions::markEndProfilerCPUSection();

			for (int i = 0; i < 3; i++)
			{
				SupportFunctions::markInitProfilerCPUSection("BPC", "BPC");
				BPC = new BPCCuda<unsigned short>(_frameStructure, _codedFrames.at((iteration * 3 + i) % (doubleBufferSize * 3)), _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);

				BPC->Decode(_frameSizes[(iteration * 3 + i)], _LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[i], _DPrefixedArrayDB[iteration % _numOfCPUThreads], _DSizeArrayDB[iteration % _numOfCPUThreads], _HBasicInformation, _DTempStoragePArrayDB[iteration % _numOfCPUThreads], _DBitStreamValuesDB[(iteration * 3 + i) % (doubleBufferSize * 3)], _DCodeStreamValuesDB[iteration % _numOfCPUThreads], _HSizeArrayDB[iteration % _numOfCPUThreads], _HTotalBSSizeDB[iteration % _numOfCPUThreads], _DWaveletCoefficientsDB[iteration % _numOfCPUThreads], _cStreams[iteration % _numOfCPUThreads + _numOfIOStreams], _HLUTBSTableSteps, _DLUTBSTableDB[iteration % _numOfCPUThreads], &_measurementsBPC[iteration % _numOfCPUThreads]);
				if (i == 2)
					_doubleBufferInput[iteration % doubleBufferSize] = 3;
				SupportFunctions::markEndProfilerCPUSection();
				SupportFunctions::markInitProfilerCPUSection("DWT", "DWT - Decoding");
				DWTGen = new DWT<float, float2>(_frameStructure, _waveletType, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _quantizationSize);
				DWTGen->DWTDecode(_DWaveletCoefficientsDB[iteration % _numOfCPUThreads], _DImagePixelsRGBTransformedLossy[(iteration * 3 + i) % (doubleBufferSize * 3)], _cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams]);
				SupportFunctions::markEndProfilerCPUSection();
			}
			SupportFunctions::markInitProfilerCPUSection("ColorTransform", "Color Tranform Kernel");
			cudaStreamSynchronize(_cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams]);
			prepareRGBFrame(_cStreams[_numOfIOStreams + (iteration % _numOfCPUThreads)], iteration);
			SupportFunctions::markEndProfilerCPUSection();
			SupportFunctions::markInitProfilerCPUSection("Copying", "Copying - Waiting for write buffer.");
			bufferValue = _doubleBufferOutput[iteration % doubleBufferSize];
			while ((bufferValue == 1) || (bufferValue == 2))
			{
				bufferValue = _doubleBufferOutput[iteration % doubleBufferSize];
			}
			SupportFunctions::markEndProfilerCPUSection();
			cudaStreamSynchronize(_cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams]);
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_HImagePixelsCharDB[(iteration * 3) % (doubleBufferSize * 3)], _DImagePixelsCharDB[(iteration * 3) % (doubleBufferSize * 3)], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()) * sizeof(char), cudaMemcpyDeviceToHost, _cStreams[(iteration % doubleBufferSize) + doubleBufferSize]));
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_HImagePixelsCharDB[(iteration * 3 + 1) % (doubleBufferSize * 3)], _DImagePixelsCharDB[(iteration * 3 + 1) % (doubleBufferSize * 3)], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()) * sizeof(char), cudaMemcpyDeviceToHost, _cStreams[(iteration % doubleBufferSize) + doubleBufferSize]));
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_HImagePixelsCharDB[(iteration * 3 + 2) % (doubleBufferSize * 3)], _DImagePixelsCharDB[(iteration * 3 + 2) % (doubleBufferSize * 3)], (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()) * sizeof(char), cudaMemcpyDeviceToHost, _cStreams[(iteration % doubleBufferSize) + doubleBufferSize]));
			GPU_HANDLE_ERROR(cudaEventRecord(_cEvents[(iteration % doubleBufferSize) + doubleBufferSize], _cStreams[(iteration % doubleBufferSize) + doubleBufferSize]));
			cudaStreamSynchronize(_cStreams[(iteration % doubleBufferSize) + doubleBufferSize]);

			_framesC.at((iteration * 3) % (doubleBufferSize * 3)) = _HImagePixelsCharDB[(iteration * 3) % (doubleBufferSize * 3)];
			_framesC.at((iteration * 3 + 1) % (doubleBufferSize * 3)) = _HImagePixelsCharDB[(iteration * 3 + 1) % (doubleBufferSize * 3)];
			_framesC.at((iteration * 3 + 2) % (doubleBufferSize * 3)) = _HImagePixelsCharDB[(iteration * 3 + 2) % (doubleBufferSize * 3)];
			_doubleBufferOutput[iteration % doubleBufferSize] = 1;
			delete DWTGen;
			delete BPC;
		}
		else 
		{
			BPCCuda<unsigned short>* BPC = new BPCCuda<unsigned short>(_frameStructure, _codedFrames.at(iteration % doubleBufferSize), _waveletLevels, _DWTCBWidth, _DWTCBHeight, _codingPasses, _waveletType, _quantizationSize, _k, _LUTAmountOfBitplaneFiles);
			GPU_HANDLE_ERROR(cudaStreamWaitEvent(_cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams], _cEvents[iteration % doubleBufferSize], 0));
			_doubleBufferInput[iteration % doubleBufferSize] = 2;
			BPC->Decode(_frameSizes[iteration], _LUTNumberOfBitplanes, _LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSign, _LUTContextSignificance, _LUTMultPrecision, _LUTInformation[0], _DPrefixedArrayDB[iteration % _numOfCPUThreads], _DSizeArrayDB[iteration % _numOfCPUThreads], _HBasicInformation, _DTempStoragePArrayDB[iteration % _numOfCPUThreads], _DBitStreamValuesDB[iteration % doubleBufferSize], _DCodeStreamValuesDB[iteration % _numOfCPUThreads], _HSizeArrayDB[iteration % _numOfCPUThreads], _HTotalBSSizeDB[iteration % _numOfCPUThreads], _DWaveletCoefficientsDB[iteration % _numOfCPUThreads], _cStreams[iteration % _numOfCPUThreads + _numOfIOStreams], _HLUTBSTableSteps, _DLUTBSTableDB[iteration % _numOfCPUThreads], &_measurementsBPC[iteration % _numOfCPUThreads]);

			_doubleBufferInput[iteration % doubleBufferSize] = 3;
			DWT<float, float2>* DWTGen = new DWT<float, float2>(_frameStructure, _waveletType, _waveletLevels, _DWTCBWidth, _DWTCBHeight, _quantizationSize);
			DWTGen->DWTDecode(_DWaveletCoefficientsDB[iteration % _numOfCPUThreads], _DImagePixelsDBLossy[iteration % doubleBufferSize], _cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams]);
			int numberOfThreadsPerBlock = 256;
			int numberOfBlocks = (int)ceil(_frameStructure->getAdaptedHeight() * _frameStructure->getAdaptedWidth() / numberOfThreadsPerBlock);

			removeOffsetAndApplyMaxMinLossy << < numberOfBlocks, numberOfThreadsPerBlock, 0, _cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams] >> > (_DImagePixelsDBLossy[iteration % doubleBufferSize] + _extraWaveletAllocation, _frameStructure->getBitDepth(), _frameStructure->getSignedOrUnsigned());
		
			bufferValue = _doubleBufferOutput[iteration % doubleBufferSize];
			while ((bufferValue == 1) || (bufferValue == 2))
			{
				bufferValue = _doubleBufferOutput[iteration % doubleBufferSize];
			}
			_frameSizes[iteration % doubleBufferSize] = _frameStructure->getWidth() * _frameStructure->getHeight();
			cudaStreamSynchronize(_cStreams[(iteration % _numOfCPUThreads) + _numOfIOStreams]);
			GPU_HANDLE_ERROR(cudaMemcpyAsync(_HImagePixelsDBLossy[iteration % doubleBufferSize], _DImagePixelsDBLossy[iteration % doubleBufferSize] + _extraWaveletAllocation, (_frameStructure->getAdaptedWidth()*_frameStructure->getAdaptedHeight()) * sizeof(float), cudaMemcpyDeviceToHost, _cStreams[(iteration % doubleBufferSize) + doubleBufferSize]));
			GPU_HANDLE_ERROR(cudaEventRecord(_cEvents[(iteration % doubleBufferSize) + doubleBufferSize], _cStreams[(iteration % doubleBufferSize) + doubleBufferSize]));
			_doubleBufferOutput[iteration % doubleBufferSize] = 1;

			_framesLossy.at(iteration % doubleBufferSize) = _HImagePixelsDBLossy[iteration % doubleBufferSize];
			delete DWTGen;
			delete BPC;
		}
	}
}

/*
* General function which initializes the CPU threads needed depending on the amount of streams running the decoding.
*/
void DecodingEngine::engineManager(int cType)
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
		Image* frameStructure = new Image();
		initMemory(VIDEO);
		initLUT();
		auto startProcessDisregardAllocationTimings = std::chrono::steady_clock::now();
		auto accumulated = 0.0;
		auto startTime = std::chrono::high_resolution_clock::now();
		auto endTime = std::chrono::high_resolution_clock::now();
		if ((_waveletType) == LOSSLESS)
		{
			if (_frameStructure->getIsRGB())
			{
				readThread = std::async(std::launch::async, &DecodingEngine::readFileParallelRGB, this);
				writeThread = std::async(std::launch::async, &DecodingEngine::writeFileParallelRGB, this);

			}
			else
			{
				readThread = std::async(std::launch::async, &DecodingEngine::readFileParallel, this);
				writeThread = std::async(std::launch::async, &DecodingEngine::writeFileParallel, this);

			}
		}
		else
		{
			if (_frameStructure->getIsRGB())
			{
				readThread = std::async(std::launch::async, &DecodingEngine::readFileParallelRGB, this);
				writeThread = std::async(std::launch::async, &DecodingEngine::writeFileParallelRGB, this);
			}
			else
			{
				readThread = std::async(std::launch::async, &DecodingEngine::readFileParallel, this);
				writeThread = std::async(std::launch::async, &DecodingEngine::writeFileParallelLossy, this);

			}
		}
		int doubleBufferValue;
		int prevPercentage = -1;
		for (int i = 0; i < _numberOfFrames; i++)
		{
			if (i >= _numOfCPUThreads)
			{
				processingThreads[i % _numOfCPUThreads].get();
			}
			SupportFunctions::markInitProfilerCPUSection("GPU", "GPU THREAD IDLE");
			startTime = std::chrono::high_resolution_clock::now();
			//prevPercentage = refreshPercentageCompleted(i, 950, prevPercentage);
			doubleBufferValue = _doubleBufferInput[i % doubleBufferSize];
			while (doubleBufferValue != 1)
			{ 
				doubleBufferValue = _doubleBufferInput[i % doubleBufferSize];
			}
			endTime = std::chrono::high_resolution_clock::now();
			SupportFunctions::markEndProfilerCPUSection();
			processingThreads[i % _numOfCPUThreads] = std::async(std::launch::async, &DecodingEngine::runVideo, this, i);
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
		initMemory(cType);
		initLUT();
		runImage();
	}
}