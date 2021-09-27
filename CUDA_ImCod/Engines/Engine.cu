#include "Engine.cuh"

Engine::Engine(){}

/*
* Processes LUT files to incorporate its information in order to make it available for BPC-PaCo algorithm
*/
void Engine::initLUT()
{

	IOManager<int, int2>* IOM = new IOManager<int, int2>();
	if (_k > 0) //Multi-file load. Even a small K (i.e.: 0.1) can lead some codeblocks to encode multiple bitplanes at once.
	{
		if (_codingPasses == 2)
		{
			loadLUTHeaders(_LUTPath + "/header.txt", IOM);
			IOM->setInputFile("");
			IOM->setInputFolder(_LUTPath);
			int** hostLUT;
			hostLUT = (int**)malloc(3 * sizeof(int**));
			_LUTInformation = (int**)malloc(3 * sizeof(int**));
			int dataSizeForLUTs = (((_LUTNumberOfSubbands) * (_LUTNumberOfBitplanes) * (_LUTContextRefinement) * (_waveletLevels)) + (_LUTNumberOfBitplanes * _LUTContextRefinement) +
				((_LUTNumberOfSubbands) * (_LUTNumberOfBitplanes) * (_LUTContextSignificance) * (_waveletLevels)) + (_LUTNumberOfBitplanes * _LUTContextSignificance) +
				((_LUTNumberOfSubbands) * (_LUTNumberOfBitplanes) * (_LUTContextSign) * (_waveletLevels)) + (_LUTNumberOfBitplanes * _LUTContextSign)) * sizeof(int);
			if (_LUTNFiles == 1)
			{
				GPU_HANDLE_ERROR(cudaMalloc(&(_LUTInformation[0]), _LUTAmountOfBitplaneFiles * dataSizeForLUTs));
				hostLUT[0] = (int*)malloc(_LUTAmountOfBitplaneFiles * dataSizeForLUTs);
				for (int j = 0; j < _LUTAmountOfBitplaneFiles; j++)
				{
					IOM->loadLUTUpgraded(_LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSignificance, _LUTContextSign, _codingPasses, _waveletLevels, _LUTNumberOfBitplanes, &(hostLUT[0][(j*dataSizeForLUTs/sizeof(int))]), 0, this->_quantizationSize, j);
				}
				GPU_HANDLE_ERROR(cudaMemcpy(_LUTInformation[0], hostLUT[0], _LUTAmountOfBitplaneFiles * ((IOM->getOutputData()[0] + IOM->getOutputData()[1] + IOM->getOutputData()[2]) * sizeof(int)), cudaMemcpyHostToDevice));
				free(hostLUT);
			}
			else
			{
				GPU_HANDLE_ERROR(cudaMalloc(&(_LUTInformation[0]), _LUTAmountOfBitplaneFiles * dataSizeForLUTs));
				GPU_HANDLE_ERROR(cudaMalloc(&(_LUTInformation[1]), _LUTAmountOfBitplaneFiles * dataSizeForLUTs));
				GPU_HANDLE_ERROR(cudaMalloc(&(_LUTInformation[2]), _LUTAmountOfBitplaneFiles * dataSizeForLUTs));
				hostLUT[0] = (int*)malloc(_LUTAmountOfBitplaneFiles * dataSizeForLUTs);
				hostLUT[1] = (int*)malloc(_LUTAmountOfBitplaneFiles * dataSizeForLUTs);
				hostLUT[2] = (int*)malloc(_LUTAmountOfBitplaneFiles * dataSizeForLUTs);
				for (int i = 0; i < _LUTNFiles; i++)
				{
					for (int j = 0; j < _LUTAmountOfBitplaneFiles; j++)
					{
						IOM->loadLUTUpgraded(_LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSignificance, _LUTContextSign, _codingPasses, _waveletLevels, _LUTNumberOfBitplanes, &(hostLUT[i][(j * dataSizeForLUTs / sizeof(int))]), i + 1, this->_quantizationSize, j);
					}
					GPU_HANDLE_ERROR(cudaMemcpy(_LUTInformation[i], hostLUT[i], _LUTAmountOfBitplaneFiles * ((IOM->getOutputData()[0] + IOM->getOutputData()[1] + IOM->getOutputData()[2]) * sizeof(int)), cudaMemcpyHostToDevice));
				}						

				free(hostLUT);
			}

		}
		else
		{
			loadLUTHeaders(_LUTPath + "/header.txt", IOM);
			IOM->setInputFile("");
			IOM->setInputFolder(_LUTPath);
			int** hostLUT;
			hostLUT = (int**)malloc(3 * sizeof(int***));
			_LUTInformation = (int**)malloc(3 * sizeof(int**));
			int dataSizeForLUTs = ((((_LUTNumberOfSubbands) * (_LUTNumberOfBitplanes) * (_LUTContextRefinement) * (_waveletLevels)) + (_LUTNumberOfBitplanes * _LUTContextRefinement)) +
				((((_LUTNumberOfSubbands) * (_LUTNumberOfBitplanes) * (_LUTContextSignificance) * (_waveletLevels)) + (_LUTNumberOfBitplanes * _LUTContextSignificance)) * 2) +
				((((_LUTNumberOfSubbands) * (_LUTNumberOfBitplanes) * (_LUTContextSign) * (_waveletLevels)) + (_LUTNumberOfBitplanes * _LUTContextSign)) * 2 )) * sizeof(int);
			if (_LUTNFiles == 1)
			{
				GPU_HANDLE_ERROR(cudaMalloc(&(_LUTInformation[0]), _LUTAmountOfBitplaneFiles * dataSizeForLUTs));
				hostLUT[0] = (int*)malloc(_LUTAmountOfBitplaneFiles * dataSizeForLUTs);
				for (int j = 0; j < _LUTAmountOfBitplaneFiles; j++)
				{
					IOM->loadLUTUpgraded(_LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSignificance, _LUTContextSign, _codingPasses, _waveletLevels, _LUTNumberOfBitplanes, &(hostLUT[0][(j * dataSizeForLUTs / sizeof(int))]), 0, this->_quantizationSize, j);
				}
				GPU_HANDLE_ERROR(cudaMemcpy(_LUTInformation[0], hostLUT[0], _LUTAmountOfBitplaneFiles * ((IOM->getOutputData()[0] + (IOM->getOutputData()[1] * 2) + (IOM->getOutputData()[2] * 2)) * sizeof(int)), cudaMemcpyHostToDevice));

				free(hostLUT);
			}
			else
			{
				GPU_HANDLE_ERROR(cudaMalloc(&(_LUTInformation[0]), _LUTAmountOfBitplaneFiles * dataSizeForLUTs));
				GPU_HANDLE_ERROR(cudaMalloc(&(_LUTInformation[1]), _LUTAmountOfBitplaneFiles * dataSizeForLUTs));
				GPU_HANDLE_ERROR(cudaMalloc(&(_LUTInformation[2]), _LUTAmountOfBitplaneFiles * dataSizeForLUTs));
				hostLUT[0] = (int*)malloc(_LUTAmountOfBitplaneFiles * dataSizeForLUTs);
				hostLUT[1] = (int*)malloc(_LUTAmountOfBitplaneFiles * dataSizeForLUTs);
				hostLUT[2] = (int*)malloc(_LUTAmountOfBitplaneFiles * dataSizeForLUTs);
				for (int i = 0; i < _LUTNFiles; i++)
				{
					for (int j = 0; j < _LUTAmountOfBitplaneFiles; j++)
					{
						IOM->loadLUTUpgraded(_LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSignificance, _LUTContextSign, _codingPasses, _waveletLevels, _LUTNumberOfBitplanes, &(hostLUT[i][(j * dataSizeForLUTs / sizeof(int))]), i + 1, this->_quantizationSize, j);
					}	
					GPU_HANDLE_ERROR(cudaMemcpy(_LUTInformation[i], hostLUT[i], _LUTAmountOfBitplaneFiles * ((IOM->getOutputData()[0] + (IOM->getOutputData()[1] * 2) + (IOM->getOutputData()[2] * 2)) * sizeof(int)), cudaMemcpyHostToDevice));

				}
				free(hostLUT);
			}
		}
	}
	else //Only one file per CodingPass + Channel will be present
	{
		if (_codingPasses == 2)
		{
			loadLUTHeaders(_LUTPath + "/header.txt", IOM);
			IOM->setInputFile("");
			IOM->setInputFolder(_LUTPath);
			int** hostLUT;
			hostLUT = (int**)malloc(3 * sizeof(int**));
			_LUTInformation = (int**)malloc(3 * sizeof(int**));
			int dataSizeForLUTs = (((_LUTNumberOfSubbands) * (_LUTNumberOfBitplanes) * (_LUTContextRefinement) * (_waveletLevels)) + (_LUTNumberOfBitplanes * _LUTContextRefinement) +
				((_LUTNumberOfSubbands) * (_LUTNumberOfBitplanes) * (_LUTContextSignificance) * (_waveletLevels)) + (_LUTNumberOfBitplanes * _LUTContextSignificance) +
				((_LUTNumberOfSubbands) * (_LUTNumberOfBitplanes) * (_LUTContextSign) * (_waveletLevels)) + (_LUTNumberOfBitplanes * _LUTContextSign)) * sizeof(int);
			if (_LUTNFiles == 1)
			{
				GPU_HANDLE_ERROR(cudaMalloc(&(_LUTInformation[0]), _LUTAmountOfBitplaneFiles * dataSizeForLUTs));
				hostLUT[0] = (int*)malloc(_LUTAmountOfBitplaneFiles * dataSizeForLUTs);

				IOM->loadLUTUpgraded(_LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSignificance, _LUTContextSign, _codingPasses, _waveletLevels, _LUTNumberOfBitplanes, &(hostLUT[0][0]), 0, this->_quantizationSize, 0);
				
				GPU_HANDLE_ERROR(cudaMemcpy(_LUTInformation[0], hostLUT[0], (IOM->getOutputData()[0] + IOM->getOutputData()[1] + IOM->getOutputData()[2]) * sizeof(int), cudaMemcpyHostToDevice));
				free(hostLUT);
			}
			else
			{
				GPU_HANDLE_ERROR(cudaMalloc(&(_LUTInformation[0]), _LUTAmountOfBitplaneFiles * dataSizeForLUTs));
				GPU_HANDLE_ERROR(cudaMalloc(&(_LUTInformation[1]), _LUTAmountOfBitplaneFiles * dataSizeForLUTs));
				GPU_HANDLE_ERROR(cudaMalloc(&(_LUTInformation[2]), _LUTAmountOfBitplaneFiles * dataSizeForLUTs));
				hostLUT[0] = (int*)malloc(_LUTAmountOfBitplaneFiles * dataSizeForLUTs);
				hostLUT[1] = (int*)malloc(_LUTAmountOfBitplaneFiles * dataSizeForLUTs);
				hostLUT[2] = (int*)malloc(_LUTAmountOfBitplaneFiles * dataSizeForLUTs);
				for (int i = 0; i < _LUTNFiles; i++)
				{
					IOM->loadLUTUpgraded(_LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSignificance, _LUTContextSign, _codingPasses, _waveletLevels, _LUTNumberOfBitplanes, &(hostLUT[i][0]), i + 1, this->_quantizationSize, 0);
					
					GPU_HANDLE_ERROR(cudaMemcpy(_LUTInformation[i], hostLUT[i], (IOM->getOutputData()[0] + IOM->getOutputData()[1] + IOM->getOutputData()[2]) * sizeof(int), cudaMemcpyHostToDevice));

				}
				free(hostLUT);
			}

		}
		else
		{
			loadLUTHeaders(_LUTPath + "/header.txt", IOM);
			IOM->setInputFile("");
			IOM->setInputFolder(_LUTPath);
			int** hostLUT;
			hostLUT = (int**)malloc(3 * sizeof(int**));
			_LUTInformation = (int**)malloc(3 * sizeof(int**));
			int dataSizeForLUTs = ((((_LUTNumberOfSubbands) * (_LUTNumberOfBitplanes) * (_LUTContextRefinement) * (_waveletLevels)) + (_LUTNumberOfBitplanes * _LUTContextRefinement)) +
				((((_LUTNumberOfSubbands) * (_LUTNumberOfBitplanes) * (_LUTContextSignificance) * (_waveletLevels)) + (_LUTNumberOfBitplanes * _LUTContextSignificance)) * 2) +
				((((_LUTNumberOfSubbands) * (_LUTNumberOfBitplanes) * (_LUTContextSign) * (_waveletLevels)) + (_LUTNumberOfBitplanes * _LUTContextSign)) * 2)) * sizeof(int);
			if (_LUTNFiles == 1)
			{
				GPU_HANDLE_ERROR(cudaMalloc(&(_LUTInformation[0]), _LUTAmountOfBitplaneFiles* dataSizeForLUTs));
				hostLUT[0] = (int*)malloc(_LUTAmountOfBitplaneFiles * dataSizeForLUTs);

				IOM->loadLUTUpgraded(_LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSignificance, _LUTContextSign, _codingPasses, _waveletLevels, _LUTNumberOfBitplanes, &(hostLUT[0][0]), 0, this->_quantizationSize, 0);

				GPU_HANDLE_ERROR(cudaMemcpy(_LUTInformation[0], hostLUT[0], (IOM->getOutputData()[0] + (IOM->getOutputData()[1] * 2) + (IOM->getOutputData()[2] * 2)) * sizeof(int), cudaMemcpyHostToDevice));

				free(hostLUT);
			}
			else
			{
				GPU_HANDLE_ERROR(cudaMalloc(&(_LUTInformation[0]), _LUTAmountOfBitplaneFiles* dataSizeForLUTs));
				GPU_HANDLE_ERROR(cudaMalloc(&(_LUTInformation[1]), _LUTAmountOfBitplaneFiles* dataSizeForLUTs));
				GPU_HANDLE_ERROR(cudaMalloc(&(_LUTInformation[2]), _LUTAmountOfBitplaneFiles* dataSizeForLUTs));
				hostLUT[0] = (int*)malloc(_LUTAmountOfBitplaneFiles * dataSizeForLUTs);
				hostLUT[1] = (int*)malloc(_LUTAmountOfBitplaneFiles * dataSizeForLUTs);
				hostLUT[2] = (int*)malloc(_LUTAmountOfBitplaneFiles * dataSizeForLUTs);
				for (int i = 0; i < _LUTNFiles; i++)
				{
					IOM->loadLUTUpgraded(_LUTNumberOfSubbands, _LUTContextRefinement, _LUTContextSignificance, _LUTContextSign, _codingPasses, _waveletLevels, _LUTNumberOfBitplanes, &(hostLUT[i][0]), i + 1, this->_quantizationSize, 0);

					GPU_HANDLE_ERROR(cudaMemcpy(_LUTInformation[i], hostLUT[i], (IOM->getOutputData()[0] + (IOM->getOutputData()[1] * 2) + (IOM->getOutputData()[2] * 2)) * sizeof(int), cudaMemcpyHostToDevice));

				}
				free(hostLUT);
			}
		}
	}

}
/*
* Function which loads the LUT Headers from the file headers.txt.
*/

void Engine::loadLUTHeaders(std::string fullPath, IOManager<int, int2>* IOM)
{
	int* values = new int[8];
	IOM->setInputFile(fullPath);
	values = IOM->loadLUTHeaders();

	_LUTNumberOfBitplanes = values[0];
	_LUTNumberOfSubbands = values[1];
	_LUTContextRefinement = values[2];
	_LUTContextSign = values[3];
	_LUTContextSignificance = values[4];
	_LUTMultPrecision = values[5];
	_LUTNFiles = values[6];
	_LUTAmountOfBitplaneFiles = values[7];
	if ((_LUTAmountOfBitplaneFiles) > 32)
	{
		_LUTAmountOfBitplaneFiles = 32;
		std::cout << "The amount of Bitplane Files set in the header.txt file exceeds the maximum allowed of 32. 32 will be the amount of Bitplane Files used." << std::endl;
	}
	delete[] values;
}



void Engine::setCodingPasses(int cp)
{
	_codingPasses = cp;
}

void Engine::setWaveletLevels(int wL)
{
	_waveletLevels = wL;
}

void Engine::setNumberOfFrames(int nFr)
{
	_numberOfFrames = nFr;
}

void Engine::setOutputFile(std::string oF)
{
	_outputFile = oF;
}

void Engine::setWType(bool wt)
{
	_waveletType = wt;
}

void Engine::setCBWidth(int cbw)
{
	_DWTCBWidth = cbw;
}

void Engine::setCBHeight(int cbh)
{
	_DWTCBHeight = cbh;
}

void Engine::setQSizes(float qs)
{
	_quantizationSize = qs;
}

void Engine::setFrameStructure(Image* fS)
{
	_frameStructure = fS;
}

void Engine::setLUTPath(std::string lp)
{
	_LUTPath = lp;
}

void Engine::setNumberOfStreams(int nOfS)
{
	_numOfStreams = nOfS;
}

void Engine::setKFactor(float k)
{
	_k = k;
}