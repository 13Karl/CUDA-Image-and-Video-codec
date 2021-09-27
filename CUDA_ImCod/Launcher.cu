#pragma once
#include <iostream>
#include <string>
#include "IO/CommandLineParser.hpp"
#include "Engines/CodingEngine.cuh"
#include "Engines/DecodingEngine.cuh"

std::string _inputFile;
std::string _outputFile;
std::string _LUTFolder = "";
int _cd = 2;
int _xSize = 0;
int _ySize = 0;
int _cbWidth = 64;
int _cbHeight = 18;
int _wLevels = 5;
int _cp = 2;
int _endianess = 0;
int _bps = 8;
int _signedOrUnsigned = 0;
int _isVideo = 0;
int _frames = 0;
int _avoidSizeCheck = 0;
int _components = 1;
int _numOfStreams = 2;
bool _isRGB = false;
bool _type = false;
float _qs = 1;
float _k = 0;


/*
* Parameters to be set when the application is started. The -cd one states if the application is to code or decode.
* For decoding, various attributes are retrieved from the compressed file, hence they are not read from the input line.
*/
void setVariablesWithCMDParameters(InputParser* iP)
{
	if (iP->cmdOptionExists("-cd")) {
		_cd = stoi(iP->getCmdOption("-cd"));
		std::cout << "User entered -cd command " << _cd << std::endl;
	}
	if (_cd == 0)
	{
		if (iP->cmdOptionExists("-wl")) {
			_wLevels = stoi(iP->getCmdOption("-wl"));
			std::cout << "User entered -wl command " << _wLevels << std::endl;
		}
		if (iP->cmdOptionExists("-cp")) {
			_cp = stoi(iP->getCmdOption("-cp"));
			std::cout << "User entered -cp command " << _cp << std::endl;
		}
		if (iP->cmdOptionExists("-type")) {
			_type = stoi(iP->getCmdOption("-type")) == 0 ? false : true;
			std::cout << "User entered -t command " << _type << std::endl;
		}
		if (iP->cmdOptionExists("-qs")) {
			_qs = stof(iP->getCmdOption("-qs"));
			std::cout << "User entered -qs command " << _qs << std::endl;
		}
		if (iP->cmdOptionExists("-i")) {
			_inputFile = iP->getCmdOption("-i");
			std::cout << "User entered -i command " << _inputFile << std::endl;
		}
		if (iP->cmdOptionExists("-o")) {
			_outputFile = iP->getCmdOption("-o");
			std::cout << "User entered -o command " << _outputFile << std::endl;
		}
		if (iP->cmdOptionExists("-cbWidth")) {
			_cbWidth = stoi(iP->getCmdOption("-cbWidth"));
			std::cout << "User entered -cbWidth command " << _cbWidth << std::endl;
		}
		if (iP->cmdOptionExists("-cbHeight")) {
			_cbHeight = stoi(iP->getCmdOption("-cbHeight"));
			std::cout << "User entered -cbHeight command " << _cbHeight << std::endl;
		}

		if (iP->cmdOptionExists("-xSize")) {
			_xSize = stoi(iP->getCmdOption("-xSize"));
			std::cout << "User entered -xSize command " << _xSize << std::endl;
		}
		if (iP->cmdOptionExists("-ySize")) {
			_ySize = stoi(iP->getCmdOption("-ySize"));
			std::cout << "User entered -ySize command " << _ySize << std::endl;
		}
		if (iP->cmdOptionExists("-endianess")) {
			_endianess = stoi(iP->getCmdOption("-endianess"));
			std::cout << "User entered -endianess command " << _endianess << std::endl;
		}
		if (iP->cmdOptionExists("-bps")) {
			_bps = stoi(iP->getCmdOption("-bps"));
			std::cout << "User entered -bps command " << _bps << std::endl;
		}
		if (iP->cmdOptionExists("-signedOrUnsigned")) {
			_signedOrUnsigned = stoi(iP->getCmdOption("-signedOrUnsigned"));
			std::cout << "User entered -signedOrUnsigned command " << _signedOrUnsigned << std::endl;
		}
		if (iP->cmdOptionExists("-LUTFolder")) {
			_LUTFolder = iP->getCmdOption("-LUTFolder");
			std::cout << "User entered -LUTFolder command " << _LUTFolder << std::endl;
		}
		if (iP->cmdOptionExists("-video")) {
			_isVideo = stoi(iP->getCmdOption("-video"));
			std::cout << "User entered -video command " << _isVideo << std::endl;
		}
		if (iP->cmdOptionExists("-frames")) {
			_frames = stoi(iP->getCmdOption("-frames"));
			std::cout << "User entered -frames command " << _frames << std::endl;
		}
		if (iP->cmdOptionExists("-avoidSizeCheck")) {
			_avoidSizeCheck = stoi(iP->getCmdOption("-avoidSizeCheck"));
			std::cout << "User entered -avoidSizeCheck command " << _avoidSizeCheck << std::endl;
		}
		if (iP->cmdOptionExists("-components")) {
			_components = stoi(iP->getCmdOption("-components"));
			std::cout << "User entered -components command " << _components << std::endl;
		}
		if (iP->cmdOptionExists("-isRGB")) {
			_isRGB = stoi(iP->getCmdOption("-isRGB"));
			std::cout << "User entered -isRGB command " << _isRGB << std::endl;
		}

		if (iP->cmdOptionExists("-numberOfStreams")) {
			_numOfStreams = stoi(iP->getCmdOption("-numberOfStreams"));
			std::cout << "User entered -numberOfStreams command " << _numOfStreams << std::endl;
		}

		if (iP->cmdOptionExists("-k")) {
			_k = stof(iP->getCmdOption("-k"));
			std::cout << "User entered -k command " << _k << std::endl;
		}

		if (((_qs < 0) || (_qs > 1)) || (_cd == 2) || ((_wLevels < 1) || (_xSize <= 0) || (_ySize <= 0) || (_wLevels > 10)) || (_inputFile == "") || (_outputFile == "") || (_cbWidth % 64 != 0) || (_cbHeight > 20) || (_cbHeight < 18) || (_cp < 2) || (_cp > 3) || (_k < 0) || (_k > 65.535))
		{
			std::cout << "Incorrect parameters. Please choose valid values." << std::endl;
			exit(-1);
		}
	}
	else
	{
		if (iP->cmdOptionExists("-i")) {
			_inputFile = iP->getCmdOption("-i");
			std::cout << "User entered -i command " << _inputFile << std::endl;
		}
		if (iP->cmdOptionExists("-o")) {
			_outputFile = iP->getCmdOption("-o");
			std::cout << "User entered -o command " << _outputFile << std::endl;
		}
		if (iP->cmdOptionExists("-LUTFolder")) {
			_LUTFolder = iP->getCmdOption("-LUTFolder");
			std::cout << "User entered -LUTFolder command " << _LUTFolder << std::endl;
		}
		if (iP->cmdOptionExists("-numberOfStreams")) {
			_numOfStreams = stoi(iP->getCmdOption("-numberOfStreams"));
			std::cout << "User entered -numberOfStreams command " << _numOfStreams << std::endl;
		}
		if (iP->cmdOptionExists("-video")) {
			_isVideo = stoi(iP->getCmdOption("-video"));
			std::cout << "User entered -video command " << _isVideo << std::endl;
		}
	}


}


/*
* Sets the image properties.
*/
void setImageParameters(Image* frameStructure)
{
	frameStructure->setBitDepth(_bps);
	frameStructure->setBitsPerSample(_bps);
	frameStructure->setComponents(_components);
	frameStructure->setEndianess(_endianess);
	frameStructure->setHeight(_ySize);
	frameStructure->setIsRGB(_isRGB);
	frameStructure->setName(_inputFile);
	frameStructure->setSignedOrUnsigned(_signedOrUnsigned);
	frameStructure->setWidth(_xSize);
}

/*
* Initializes the application, first reading the input from the user, then initializing the main engine and choosing either the coding engine or the decoding one.
* CUDA devices are initialized here. The chrono system is also initialized here.
*/
int main(int argc, char** argv)
{
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	auto start = std::chrono::steady_clock::now();
	InputParser* iP = new InputParser(argc, argv);
	if (iP->cmdOptionExists("-h"))
		iP->printHelp();
	else
	{
		setVariablesWithCMDParameters(iP);
		delete iP;

		Image* frameStructure = new Image();
		if (_cd == 0)
		{
			setImageParameters(frameStructure);
			Engine* codingEngine = new CodingEngine();
			codingEngine->setNumberOfStreams(_numOfStreams);
			codingEngine->setCBHeight(_cbHeight);
			codingEngine->setCBWidth(_cbWidth);
			codingEngine->setCodingPasses(_cp);
			codingEngine->setFrameStructure(frameStructure);
			codingEngine->setLUTPath(_LUTFolder);
			codingEngine->setNumberOfFrames(_frames);
			codingEngine->setOutputFile(_outputFile);
			codingEngine->setQSizes(_qs);
			codingEngine->setWaveletLevels(_wLevels);
			codingEngine->setWType(_type);
			codingEngine->setKFactor(_k);
			codingEngine->engineManager(_isVideo);
		}
		else if (_cd == 1)
		{
			frameStructure->setName(_inputFile);
			Engine* decodingEngine = new DecodingEngine();
			decodingEngine->setNumberOfStreams(_numOfStreams);
			decodingEngine->setFrameStructure(frameStructure);
			decodingEngine->setOutputFile(_outputFile);
			decodingEngine->setLUTPath(_LUTFolder);
			decodingEngine->engineManager(_isVideo);
		}
	}
	cudaDeviceReset();
	auto finish = std::chrono::steady_clock::now();
	double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(finish - start).count();
	std::cout << "The time spent with the app is: " << elapsed_seconds << std::endl;
	return (0);
}
