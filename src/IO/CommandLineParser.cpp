#pragma once
#include "CommandLineParser.hpp"

InputParser::InputParser(int &argc, char **argv) 
{
	for (int i = 1; i < argc; ++i)
		this->_tokens.push_back(std::string(argv[i]));
}
//author iain
const std::string& InputParser::getCmdOption(const std::string &option) const 
{
	std::vector<std::string>::const_iterator itr;
	itr = std::find(this->_tokens.begin(), this->_tokens.end(), option);
	if (itr != this->_tokens.end() && ++itr != this->_tokens.end()) {
		return *itr;
	}
	static const std::string emptyString("");
	return emptyString;
}
//author iain
bool InputParser::cmdOptionExists(const std::string &option) const 
{
	return std::find(this->_tokens.begin(), this->_tokens.end(), option)
		!= this->_tokens.end();
}


void InputParser::printHelp()
{
	std::cout << std::endl << std::endl << "Welcome to the GPU Codec \"PICSONG\" (Parallel Image Coding System Over Nvidia GPUs) instructions. " << std::endl << std::endl;
	std::cout << "Please, find below the available options for the right use of this codec." << std::endl << std::endl;
	std::cout << "-h: Shows up this information" << std::endl << std::endl;
	std::cout << "-wl: Sets the amount of Wavelet transforms that are used in the codification. Only available for CODING." << std::endl << std::endl;
	std::cout << "-cp [2..3]: Sets the amount of coding passes performed by the algorithm (either 2 or 3). 3 is deprecated." << std::endl << std::endl;
	std::cout << "-type [0..1]: If set to 0, the algorithm will make a Lossless DWT 5/3 transform. If set to 1, DWT 9/7 will be used, and quantization steps will be taken in consideration." << std::endl << std::endl;
	std::cout << "-qs [0..1]: This value will be used to make a quantization over the DWT coefficients after the transformed is applied. Please take into account that this value will only be used when the Lossy DWT is selected." << std::endl << std::endl;
	std::cout << "-i: This value is used to input the file to be coded/decoded. Take into account that currently only PGM, 1 band, greyscale images with .PGM \"P5\" format are compatible with the codec." << std::endl << std::endl;
	std::cout << "-o: This value is used to input the output file (either the coded file or the decoded one). The output is RAW always, with the addition of a ppm header for images." << std::endl << std::endl;
	std::cout << "-cbWidth: Sets the DWT tile width sizing." << std::endl << std::endl;
	std::cout << "-cbLength: Sets the DWT tile length sizing." << std::endl << std::endl;
	std::cout << "-cd [0..1]: 0 for Coding, 1 for Decoding." << std::endl << std::endl;
	std::cout << "-xSize: Length of the image in the x-axis." << std::endl << std::endl;
	std::cout << "-ySize: Lenght of the image in the y-axis." << std::endl << std::endl;
	std::cout << "-video[0..1]: Tells the coder whether we are dealing with an image or a RAW video/multispectral image." << std::endl << std::endl;
	std::cout << "-frames: Tells the coder how many frames the video has (or components from a multispectral image)." << std::endl << std::endl;
	std::cout << "-LUTFolder: Sets where the folder with the LUT files is located." << std::endl << std::endl;
	std::cout << "-isRGB[0..1]: Tells whether the image is RGB or not." << std::endl << std::endl;
	std::cout << "-numberOfStreams: Sets the amount processing lanes used in the codec. This parameter is limited by the GPU Architecture." << std::endl << std::endl;
	std::cout << "-k: Positive number that controls the complexity ratio in terms of throughput. Maximum value: 65.535" << std::endl;
}