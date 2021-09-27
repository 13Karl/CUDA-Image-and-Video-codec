#pragma once
#include "IOManager.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>


template<class T, class Y>
inline IOManager<T, Y>::IOManager(std::string input, std::string output)
{
	this->_input = input;
	this->_output = output;
}

template<class T, class Y>
inline IOManager<T, Y>::IOManager(){}

//Reads the initial bytes of the side information existing in a compressed file.
template<class T, class Y>
void IOManager<T, Y>::loadBasicInfo(T* HBasicInformation, int amountOfValuesToRead, std::string inputFile)
{
	std::ifstream infile;

	infile.open(inputFile, std::ios::binary | std::ios::in);

	std::streampos amount = amountOfValuesToRead;

	infile.seekg(0);
	infile.read(reinterpret_cast<char*>(HBasicInformation), amount * sizeof(T));
	infile.close();
}

//Reads a PGM image (P5 version) and returns the stream of data back to the caller.
template<class T, class Y>
void IOManager<T, Y>::loadImageChar(Image* image, unsigned char** waveletCoefficients)
{
	int sizeOfImage = image->getWidth() * image->getHeight();
	std::ifstream infile;

	infile.open(image->getName(), std::ios::binary | std::ios::ate | std::ios::in);

	int size = infile.tellg();

	char* memblock = new char[size];
	//We get the header size. Afterwards, we will begin reading in the following way:
	// First three chars will be: P, 5, \n
	// After those three first characters, there will come the sizing. It will be of x x\n space. For example, 8192 8192 \n
	// The last part of the header will contain the bitdepth. For example, 255\n.
	//infile.ignore(std::numeric_limits<std::streamsize>::max());


	int headerMemblock = size - (sizeOfImage);
	infile.seekg(0, std::ios::beg);

	infile.read(memblock, size);
	infile.close();
	
	//With "headerMemblock" we are discarding the first N bytes that hold meta information.
	for (int i = headerMemblock; i < size; i++)
	{
		((*waveletCoefficients)[i - headerMemblock]) = (unsigned char)memblock[i];
	}
	delete[] memblock;
}

/*
* Loads and reads a frame or image. Then, it mirrors either the rightmost or bottom side of it to fit a specific size.
* This additional size is needed to fit the image in the engine.
*/
template<class T, class Y>
void IOManager<T, Y>::loadFrameCAdaptedSizes(Image* image, unsigned char* waveletCoefficients, int iter)
{
	long long int sizeOfFrames = image->getWidth() * image->getHeight() * 1;

	std::ifstream infile;
	infile.open(image->getName(), std::ios::binary | std::ios::in);

	long long int size;
	std::streampos frame = sizeOfFrames;

	int addedRows = image->getAdaptedHeight() - image->getHeight();
	int addedCols = image->getAdaptedWidth() - image->getWidth();
	int totalAddedInfo = addedRows * image->getWidth() + addedCols * image->getAdaptedHeight();
	int sizeOfAdaptedFrame = totalAddedInfo + sizeOfFrames;

	std::vector<unsigned char> memblock;

	size = sizeOfFrames * (long long int)iter;
	infile.seekg(size);
	memblock.resize(sizeOfFrames);
	infile.read(reinterpret_cast<char*>(&*(memblock.data())), frame);


	int offsetPerLine = 0;
	int i = 0;
	int jAdded = 0;
	int j = 0;
	for (i = 0; i < image->getHeight(); i++)
	{
		for (jAdded = 0; jAdded < addedCols; jAdded++)
		{
			memblock.insert(memblock.begin()+(image->getWidth() + offsetPerLine + jAdded + i * image->getWidth()), (memblock[image->getWidth() + offsetPerLine - jAdded - 1 + i * image->getWidth()]));
		}
		offsetPerLine = offsetPerLine + jAdded;
	}
	for (int iAdded = 0; iAdded < addedRows * image->getAdaptedWidth(); iAdded++)
	{
		memblock.push_back(memblock[(i*image->getAdaptedWidth()) - ((iAdded / image->getAdaptedWidth()) * image->getAdaptedWidth() + image->getAdaptedWidth()) + (iAdded%image->getAdaptedWidth())]);
	}
	std::copy(memblock.begin(), memblock.end(), waveletCoefficients);
}

/*
* Loads and reads a frame or image.
*/
template<class T, class Y>
void IOManager<T, Y>::loadFrameC(Image* image, unsigned char* waveletCoefficients, int iter)
{
	long long int sizeOfFrames = image->getWidth() * image->getHeight() * 1;

	std::ifstream infile;

	infile.open(image->getName(), std::ios::binary | std::ios::in);

	long long int size;
	std::streampos frame = sizeOfFrames;
	char *memblock = new char[sizeOfFrames];

	size = sizeOfFrames * (long long int)iter;
	infile.seekg(size);
	infile.read(memblock, frame);
	for (int i = 0; i < sizeOfFrames; i++)
	{
		waveletCoefficients[i] = ((unsigned char)memblock[i]);
	}
	delete[] memblock;
}

/*
* Reads a compresssed image. It uses the offset variable to know which frame from a video or multispectral image is to be loaded.
*/
template<class T, class Y>
int IOManager<T, Y>::loadCodedFrame(Image* img, T* currentFrame, int iter, int frameSize, long long int offset)
{
	std::ifstream infile;

	infile.open(img->getName(), std::ios::binary | std::ios::in);

	std::streampos frame = frameSize;

	infile.seekg(offset *(long long int)sizeof(T));
	infile.read(reinterpret_cast<char*>(currentFrame), frame * sizeof(T));
	return frameSize;
}

template<class T, class Y>
void IOManager<T, Y>::replaceExistingFile(std::string file)
{
	std::ifstream f(file.c_str());
	if (f.is_open())
	{
		f.close();
		if (remove(file.c_str()) != 0)
		{
			std::cout << "Error deleting the existing file" << std::endl;
			exit(-1);
		}
	}
}

/*
* Stores a compressed frame into a file. It also stores a file with side information with a special extension _SIZE which includes the size of the bitstream.
*/
template<class T, class Y>
void IOManager<T, Y>::writeCodedFrame(Image* image, T* waveletCoefficients, int iter, int sizeOfBitStream, std::string outputName)
{

	std::ofstream myfile;
	std::ofstream sizes;
	myfile.open(outputName, std::ios::binary | std::ios::app | std::ios::out);
	myfile.write(reinterpret_cast<char *>(waveletCoefficients), sizeOfBitStream * sizeof(T));
	myfile.close();
	sizes.open(outputName + "_SIZE", std::ios::binary | std::ios::app | std::ios::out);
	if (iter == 0)
		sizes << sizeOfBitStream;
	else
		sizes << "," << sizeOfBitStream;
	sizes.close();
}

/*
* Function which reads the _SIZE file which has the size of each bitstream in a codestream.
*/
template<class T, class Y>
void IOManager<T,Y>::readBulkSizes(int *frameSizes, Image* img, int frames)
{
	std::ifstream inputSizes;
	std::string line;
	inputSizes.open(img->getName() + "_SIZE", std::ios::in);
	std::stringstream iss;
	for (int i = 0; i < frames; i++)
	{
		getline(inputSizes, line, ','); // get values, one at a time, delimited by the / character
		
		frameSizes[i] = std::stoi(line);
	}	
}

/*
* Appends a decoded frame into a file.
*/
template<class T, class Y>
void IOManager<T, Y>::writeDecodedFrame(Image* img, T* currentFrame, int iter, std::string outputName)
{

	std::ofstream myFile;
	
	
	char* memblock = new char[img->getWidth()*img->getHeight()];
	
	for (int i = 0; i < img->getWidth() * img->getHeight(); i++)
	{
		memblock[i] = (unsigned char)currentFrame[i];
	}
	myFile.open(outputName, std::ios::binary | std::ios::app | std::ios::out);
	myFile.write(memblock, img->getWidth() * img->getHeight());
	myFile.close();
	delete[] memblock;	
	
}

/*
* Appends a decoded frame into a file.
*/
template<class T, class Y>
void IOManager<T, Y>::writeDecodedFrameComponentUChar(Image* img, unsigned char* currentFrame, int iter, std::string outputName)
{

	std::ofstream myFile;
	myFile.open(outputName, std::ios::binary | std::ios::app | std::ios::out);
	myFile.write(reinterpret_cast<char *>(currentFrame), img->getWidth() * img->getHeight());
	myFile.close();

}

/*
* Appends a decoded RGB image into a file.
*/
template<class T, class Y>
void IOManager<T, Y>::writeDecodedFrameUChar(Image* img, unsigned char** data, std::string outputName)
{
	std::ofstream* stream = new std::ofstream(outputName, std::ios::binary | std::ios::app | std::ios::out);
	stream->write(reinterpret_cast<char *>(data[0]), img->getWidth() * img->getHeight());
	stream->seekp(img->getWidth() * img->getHeight(), std::ios::beg);
	stream->write(reinterpret_cast<char *>(data[1]), img->getWidth() * img->getHeight());
	stream->seekp((img->getWidth() * img->getHeight())*2, std::ios::beg);
	stream->write(reinterpret_cast<char *>(data[2]), img->getWidth() * img->getHeight());


}

/*
* Writes a.pgm file, including the P5 header.
*/
template<class T, class Y>
void IOManager<T, Y>::writeImage(T* image, int xSize, int ySize, int bitDepth, std::string filename)
{
	std::string data;
	
	int sizeOfString;
	char const *pchar;
	int memoryPointer;
	int actualMemoryPointer;
	int totalSize;
	//We don't know the total amount of information that will be in the header. 200 is an arbitrary number.
	char* memblockHeader = new char[200];
	//PGM Header, it is always P5\n.
	memblockHeader[0] = 'P';
	memblockHeader[1] = '5';
	memblockHeader[2] = '\n';
	memoryPointer = 3;

	//After the P5\n header, it shows the height and width of the image:
	data = std::to_string(xSize);
	sizeOfString = data.length();
	pchar = data.c_str();

	//We get the current pointer to move the position inside the loop statically.
	actualMemoryPointer = memoryPointer;
	for (int a = 0; a < sizeOfString; a++)
	{
		memblockHeader[a + actualMemoryPointer] = pchar[a];
		memoryPointer++;
	}
	//A blank space between height and width.
	memblockHeader[memoryPointer] = ' ';
	memoryPointer++;
	//Getting the ySize now
	data = std::to_string(ySize);
	sizeOfString = data.length();
	pchar = data.c_str();

	actualMemoryPointer = memoryPointer;
	for (int a = 0; a < sizeOfString; a++)
	{
		memblockHeader[a + actualMemoryPointer] = pchar[a];
		memoryPointer++;
	}
	memblockHeader[memoryPointer] = '\n';
	memoryPointer++;

	//Getting the bitDepth now

	data = std::to_string((2 << (bitDepth - 1)) - 1);
	sizeOfString = data.length();
	pchar = data.c_str();

	actualMemoryPointer = memoryPointer;
	for (int a = 0; a < sizeOfString; a++)
	{
		memblockHeader[a + actualMemoryPointer] = pchar[a];
		memoryPointer++;
	}
	memblockHeader[memoryPointer] = '\n';
	memoryPointer++;

	totalSize = memoryPointer + (xSize*ySize);
	char* memblock = new char[totalSize];
	memcpy(memblock, memblockHeader, memoryPointer);

	for (int i = memoryPointer; i < totalSize; i++)
	{
		memblock[i] = (unsigned char)image[i - memoryPointer];
	}
	nvtxNameOsThreadA(std::hash<std::thread::id>()(std::this_thread::get_id()), "MainThread");
	nvtxRangePush("Writing Decoded File");
	nvtxMark("Waiting...");
	std::ofstream* stream = new std::ofstream(filename, std::ios::binary | std::ios::trunc);
	stream->write(reinterpret_cast<char *>(memblock), totalSize);
	delete[] memblock;
	delete[] memblockHeader;
	nvtxRangePop();
}

template<class T, class Y>
int IOManager<T, Y>::convertStringToInt(std::string number, char delimiter) 
{
	std::stringstream ss(number);
	std::string numberTaken;

	//Using two getlines to retrieve the second part of the string where the number is stored.
	getline(ss, numberTaken, delimiter);
	getline(ss, numberTaken, delimiter);

	return stoi(numberTaken);
}

/*
* Reads the lut header file, which includes basic information for the LUT engine.
*/
template<class T, class Y>
int* IOManager<T, Y>::loadLUTHeaders()
{
	std::ifstream inputFile;
	int* LUTHeaders = new int[8];
	int i = 0;
	std::string line;
	inputFile.open(_input, std::ifstream::in);

	if (inputFile.is_open())
	{
		while (inputFile.good())
		{
			getline(inputFile, line);
			LUTHeaders[i] = convertStringToInt(line, ';');
			i++;
		}
		inputFile.close();
	}
	else
	{
		std::cout << "Error opening LUT headers file";
	}
	return LUTHeaders;
}

template<class T, class Y>
char* IOManager<T, Y>::concat(const char *s1, char *s2)
{
	char *result = (char*)malloc(strlen(s1) + strlen(s2) + 1);//+1 for the zero-terminator
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

/*
LUT loading taking coding passes and wavelet levels information into account. Because every wavelet level may have an arbitrary amount per bitplane, the information of the files are loaded and for those bitplanes which information
is not contained in the files are loaded with an average default value depending on the number of bits used (7 bits - 64).
The loading of 3CP is done by using the 2CP ones + two extra files with information for that specific coding pass.
Most of the code has been reused from the C version to avoid translation errors to C++.
*/
template<class T, class Y>
void IOManager<T, Y>::loadLUTUpgraded(int LUTNumberOfSubbands, int LUTContextRefinement, int LUTContextSignificance, int LUTContextSign, int codingPasses, int numberOfWaveletLevels, int LUTNumberOfBitplanes, int* hostLUT, int iter, float qStep, int amountOfBitplanes)
{
	int c = -1;
	int maxBitplane = -1;
	char fileName[400];
	int i = 0;
	int bitplane = 0;
	int prevBitplane = -1;
	int c_aux[] = { 0,0,0,0,0,0,0,0,0 };
	int wLevel;
	FILE* file;
	const char* iFolder;
	iFolder = _inputFolder.c_str();
	int subband;
	char* concatF;


	int bitplaneInitialShifting = min((int)ceil(1 / qStep), 15);
	int writtenInfo = 0;
	int *maxLocalBitplane = (int*)malloc((numberOfWaveletLevels*LUTNumberOfSubbands) * sizeof(int) + 4);
	bool in = 0;
	int shifting = 0;
	
	maxBitplane = LUTNumberOfBitplanes;

	int refinementNumber = ((LUTNumberOfSubbands) * (maxBitplane) * (LUTContextRefinement) * (numberOfWaveletLevels)) + (maxBitplane*LUTContextRefinement);
	int significanceNumber = ((LUTNumberOfSubbands) * (maxBitplane) * (LUTContextSignificance) * (numberOfWaveletLevels)) + (maxBitplane*LUTContextSignificance);
	int signNumber = ((LUTNumberOfSubbands) * (maxBitplane) * (LUTContextSign) * (numberOfWaveletLevels)) + (maxBitplane*LUTContextSign);

	switch (iter)
	{
		case 0: concatF = ".txt_";
			break;
		case 1: concatF = "R.txt_";
			break;
		case 2: concatF = "G.txt_";
			break;
		case 3: concatF = "B.txt_";
			break;
		default: concatF = ".txt_";
	}

	char amOfBp[10];
	snprintf(amOfBp, 10, "%d", amountOfBitplanes);
	sprintf(fileName, concat(concat(concat(iFolder, "ref"), concatF), amOfBp));
	file = fopen(fileName, "rb");
	
	if (file != NULL) {

		//Scans the whole refinement file to store the info in local arrays
		while (fscanf(file, "%d %d %d : %d ", &wLevel, &subband, &bitplane, &c) != EOF) {
			if (bitplane <= prevBitplane) {
				//This loop runs to fill the information that does not exist in the file, i.e., bitplanes which weren't evaluated in the LUT generator.
				for (int z = 0; z < (maxBitplane - prevBitplane - 1)*LUTContextRefinement; z++) (hostLUT)[i + (prevBitplane*LUTContextRefinement) + z + 1] = 64;
				i += maxBitplane*LUTContextRefinement;
			}
			if (((wLevel + 1) > numberOfWaveletLevels) && (subband > 0))
				break;
			prevBitplane = bitplane;
			//Adds info from the file. c is read in the while sentence.
			(hostLUT)[i + (bitplane*LUTContextRefinement)] = c;

		}

	}
	else printf("\n		ERROR - Wrong input file %s \n\n", fileName);
	
	i = refinementNumber;
	bitplane = 0;
	prevBitplane = -1;
	int bitpCounter = 0;
	sprintf(fileName, concat(concat(concat(iFolder, "sig"), concatF), amOfBp));
	file = fopen(fileName, "rb");
	
	if (file != NULL) {

		while (fscanf(file, "%d %d %d : %d %d %d %d %d %d %d %d %d ", &wLevel, &subband, &bitplane, &(c_aux[0]), &(c_aux[1]), &(c_aux[2]), &(c_aux[3]), &(c_aux[4]), &(c_aux[5]), &(c_aux[6]), &(c_aux[7]), &(c_aux[8])) != EOF) {
			if (bitplane <= prevBitplane) {
				for (int z = 0; z < (maxBitplane - prevBitplane - 1)*LUTContextSignificance; z++) (hostLUT)[i + (prevBitplane*LUTContextSignificance) + z + 9] = 64;
				i += (maxBitplane*LUTContextSignificance);
				bitpCounter += maxBitplane;
			}
			if (((wLevel + 1) > numberOfWaveletLevels) && (subband > 0))
				break;
			prevBitplane = bitplane;
			(hostLUT)[i + (bitplane*LUTContextSignificance)] = c_aux[0];
			(hostLUT)[i + (bitplane*LUTContextSignificance) + 1] = c_aux[1];
			(hostLUT)[i + (bitplane*LUTContextSignificance) + 2] = c_aux[2];
			(hostLUT)[i + (bitplane*LUTContextSignificance) + 3] = c_aux[3];
			(hostLUT)[i + (bitplane*LUTContextSignificance) + 4] = c_aux[4];
			(hostLUT)[i + (bitplane*LUTContextSignificance) + 5] = c_aux[5];
			(hostLUT)[i + (bitplane*LUTContextSignificance) + 6] = c_aux[6];
			(hostLUT)[i + (bitplane*LUTContextSignificance) + 7] = c_aux[7];
			(hostLUT)[i + (bitplane*LUTContextSignificance) + 8] = c_aux[8];


		}
	}
	else printf("\n		ERROR - Wrong input file %s \n\n", fileName);

 	i = refinementNumber + significanceNumber;
	bitplane = 0;
	prevBitplane = -1;
	bitpCounter = 0;
	sprintf(fileName, concat(concat(concat(iFolder, "sign"), concatF), amOfBp));
	file = fopen(fileName, "rb");

	if (file != NULL) {

		while (fscanf(file, "%d %d %d : %d %d %d %d ", &wLevel, &subband, &bitplane, &(c_aux[0]), &(c_aux[1]), &(c_aux[2]), &(c_aux[3])) != EOF) {
			if (bitplane <= prevBitplane) {
				for (int z = 0; z < (maxBitplane - prevBitplane - 1)*LUTContextSign; z++)
				{
					(hostLUT)[i + (prevBitplane*LUTContextSign) + z + 4] = 64;
				}
				i += (maxBitplane*LUTContextSign);
				bitpCounter += maxBitplane;
			}
			if (((wLevel + 1) > numberOfWaveletLevels) && (subband > 0))
				break;
			prevBitplane = bitplane;
			(hostLUT)[i + (bitplane*LUTContextSign)] = c_aux[0];
			(hostLUT)[i + (bitplane*LUTContextSign) + 1] = c_aux[1];
			(hostLUT)[i + (bitplane*LUTContextSign) + 2] = c_aux[2];
			(hostLUT)[i + (bitplane*LUTContextSign) + 3] = c_aux[3];

		}
		while (fscanf(file, "%d %d %d : %*d %*d %*d %*d ", &wLevel, &subband, &c) != EOF) {
			if (((wLevel + 1) > numberOfWaveletLevels) && (subband > 0))
				break;
			if (c > maxBitplane) maxBitplane = c;
		}
	}
	else printf("\n		ERROR - Wrong input file %s \n\n", fileName);
	
	if (codingPasses == 3)
	{
		i = refinementNumber + significanceNumber + signNumber;
		bitplane = 0;
		prevBitplane = -1;
		
		sprintf(fileName, concat(concat(concat(iFolder, "cp_sig"), concatF), amOfBp));
		file = fopen(fileName, "rb");

		if (file != NULL) {

			while (fscanf(file, "%d %d %d : %d %d %d %d %d %d %d %d %d ", &wLevel, &subband, &bitplane, &(c_aux[0]), &(c_aux[1]), &(c_aux[2]), &(c_aux[3]), &(c_aux[4]), &(c_aux[5]), &(c_aux[6]), &(c_aux[7]), &(c_aux[8])) != EOF) {
				if (bitplane <= prevBitplane) {
					for (int z = 0; z < (maxBitplane - prevBitplane - 1)*LUTContextSignificance; z++) (hostLUT)[i + (prevBitplane*LUTContextSignificance) + z + 9] = 64;
					i += (maxBitplane*LUTContextSignificance);
				}
				if (((wLevel + 1) > numberOfWaveletLevels) && (subband > 0))
					break;
				prevBitplane = bitplane;
				(hostLUT)[i + (bitplane*LUTContextSignificance)] = c_aux[0];
				(hostLUT)[i + (bitplane*LUTContextSignificance) + 1] = c_aux[1];
				(hostLUT)[i + (bitplane*LUTContextSignificance) + 2] = c_aux[2];
				(hostLUT)[i + (bitplane*LUTContextSignificance) + 3] = c_aux[3];
				(hostLUT)[i + (bitplane*LUTContextSignificance) + 4] = c_aux[4];
				(hostLUT)[i + (bitplane*LUTContextSignificance) + 5] = c_aux[5];
				(hostLUT)[i + (bitplane*LUTContextSignificance) + 6] = c_aux[6];
				(hostLUT)[i + (bitplane*LUTContextSignificance) + 7] = c_aux[7];
				(hostLUT)[i + (bitplane*LUTContextSignificance) + 8] = c_aux[8];

			}

		}
		else printf("\n		ERROR - Wrong input file %s \n\n", fileName);


		i = refinementNumber + significanceNumber + signNumber + significanceNumber;
		bitplane = 0;
		prevBitplane = -1;

		sprintf(fileName, concat(concat(concat(iFolder, "cp_sign"), concatF), amOfBp));
		file = fopen(fileName, "rb");

		if (file != NULL) {

			while (fscanf(file, "%d %d %d : %d %d %d %d ", &wLevel, &subband, &bitplane, &(c_aux[0]), &(c_aux[1]), &(c_aux[2]), &(c_aux[3])) != EOF) {
				if (bitplane <= prevBitplane) {
					for (int z = 0; z < (maxBitplane - prevBitplane - 1)*LUTContextSign; z++) (hostLUT)[i + (prevBitplane*LUTContextSign) + z + 4] = 64;
					i += (maxBitplane*LUTContextSign);
				}
				if (((wLevel + 1) > numberOfWaveletLevels) && (subband > 0))
					break;
				prevBitplane = bitplane;
				(hostLUT)[i + (bitplane*LUTContextSign)] = c_aux[0];
				(hostLUT)[i + (bitplane*LUTContextSign) + 1] = c_aux[1];
				(hostLUT)[i + (bitplane*LUTContextSign) + 2] = c_aux[2];
				(hostLUT)[i + (bitplane*LUTContextSign) + 3] = c_aux[3];

			}

			while (fscanf(file, "%d %d %d : %*d %*d %*d %*d ", &wLevel, &subband, &c) != EOF) {
				if (((wLevel + 1) > numberOfWaveletLevels) && (subband > 0))
					break;
				if (c > maxBitplane) maxBitplane = c;
			}

		}
		else printf("\n		ERROR - Wrong input file %s \n\n", fileName);
	}
	_outputData = new int[3];
	_outputData[0] = refinementNumber;
	_outputData[1] = significanceNumber;
	_outputData[2] = signNumber;

}

template<class T, class Y>
void IOManager<T, Y>::writeBitStreamFile(T *data, int sizeOfBitStream, std::string outputFileName)
{
	std::ofstream* stream = new std::ofstream(outputFileName, std::ios::binary | std::ios::trunc);

	stream->write(reinterpret_cast<char *>(data), sizeOfBitStream * sizeof(T));
}

template<class T, class Y>
void IOManager<T, Y>::readBitStreamFile(T* values, int sizeOfBitStream)
{
	std::string data;
	std::ifstream infile;
	nvtxNameOsThreadA(std::hash<std::thread::id>()(std::this_thread::get_id()), "MainThread");
	nvtxRangePush("Reading Coded File");
	nvtxMark("Waiting...");
	infile.open(_input, std::ios::binary | std::ios::in);
	infile.read(reinterpret_cast<char*>(values), sizeOfBitStream * sizeof(T));

	infile.close();
	nvtxRangePop();
}

template<class T, class Y>
void IOManager<T, Y>::setInputFile(std::string iFile)
{
	_input = iFile;
}

template<class T, class Y>
void IOManager<T, Y>::setInputFolder(std::string iFolder)
{
	_inputFolder = iFolder;
}

template<class T, class Y>
int* IOManager<T, Y>::getOutputData() const
{
	return _outputData;
}