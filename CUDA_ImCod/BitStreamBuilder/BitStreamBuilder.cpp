#pragma once
#include "BitStreamBuilder.hpp"
#include "../IO/IOManager.hpp"
#include <iostream>
BitStreamBuilder::BitStreamBuilder(Image *img, int* data, int cbWidth, int cbHeight, int* sArray, int* pArray, int* DTStorageBytes, bool isL, float qSize, int numBlocks, int cp, int wLevels, int* LUTBSTable, int LUTBSTableSteps, float kFactor)
{
	_image = img;
	_DWTCodeBlockWidth = cbWidth;
	_DWTCodeBlockHeight = cbHeight;
	_DCodeStream = data;
	_sizeArray = sArray;
	_prefixedArray = pArray;
	_DTempStorageBytes = DTStorageBytes;
	_isLossy = isL;
	_quantizationSize = qSize;
	_numberOfBlocks = numBlocks;
	_codingPasses = cp;
	_waveletLevels = wLevels;
	_DLUTBSTable = LUTBSTable;
	_HLUTBSTableSteps = LUTBSTableSteps;
	_kFactor = kFactor;
}

BitStreamBuilder::BitStreamBuilder(int size, unsigned short* hData, int cbSize, Image* img)
{
	_sizeOfStream = size;
	_hostData = hData;
	_codeblockSizing = cbSize;
	_image = img;
}

/*
This function stores the information used when coding an image or video in the first bytes of the first coded codeblock. It is then used as side information for the decoding process.
*/
void BitStreamBuilder::setExtraInformation(unsigned short* extraInformation, int numberOfFrames)
{
	
	int localWidth = _image->getWidth();
	int localHeight = _image->getHeight();
	int localComponents = _image->getComponents();
	unsigned short sizeFirstHalf = 0;
	unsigned short sizeSecondHalf = 0;
	unsigned short codingPassesCB1Minus1CB2WLevels1 = 0;
	unsigned short ThreeWLevelsBitDepthWTypeQSize5 = 0;
	unsigned short QSize9ComponentsRGB7 = 0;
	unsigned short TwoComponentsRGBImageHeight = 0;
	unsigned short EndianessBPSSignedOrUnsigned1Frames = 0;
	unsigned short Frames1Wasted = 0;
	unsigned short kFactor = 0;
	int sizeOfImage = localWidth*localHeight*localComponents;


	//Extract the rightmost 16 bits
	sizeFirstHalf = sizeOfImage & ((1 << 16) - 1);
	//Extract the leftmost 16 bits
	sizeSecondHalf = (sizeOfImage & ((1 << 32) - 1) - ((1 << 16) - 1)) >> 16;
	//Extract the CP amount, the CB size and the first bit of the WLevels:
	codingPassesCB1Minus1CB2WLevels1 = (_codingPasses == 2 ? 0 : 1);
	codingPassesCB1Minus1CB2WLevels1 |= _DWTCodeBlockHeight << 1;
	codingPassesCB1Minus1CB2WLevels1 |= _DWTCodeBlockWidth << 8;
	codingPassesCB1Minus1CB2WLevels1 |= (_waveletLevels & 1) << 15;
	//Extract another three bits of the WLevels, bitDepth, wType and the 5 bits of the QSize
	ThreeWLevelsBitDepthWTypeQSize5 = (_waveletLevels & 7) >> 1;
	ThreeWLevelsBitDepthWTypeQSize5 |= _image->getBitDepth() << 3;
	ThreeWLevelsBitDepthWTypeQSize5 |= (_isLossy == true ? 1 : 0) << 10;
	ThreeWLevelsBitDepthWTypeQSize5 |= (((int)(_quantizationSize*10000)) & 31) << 11;
	//Extract last 9 bits of the Qsize and 7 of the Components
	QSize9ComponentsRGB7 = (((int)(_quantizationSize*10000))) >> 5;
	QSize9ComponentsRGB7 |= (_image->getComponents() & ((1 << 7) - 1)) << 9;
	//Extract last 7 bits of the Components and the RGB. Put the rest of the 8 bits to store the Height.
	TwoComponentsRGBImageHeight = (_image->getComponents()) >> 7;
	TwoComponentsRGBImageHeight |= (_image->getIsRGB() == true ? 1 : 0) << 7;
	TwoComponentsRGBImageHeight |= (_image->getHeight() << 8);
	//Extract 8 bits for height, 1 bit for endianess, 5 bits for BPS, 1 bit for signed/unsigned and 1 bit for number of frames
	EndianessBPSSignedOrUnsigned1Frames = (_image->getHeight() >> 8);
	EndianessBPSSignedOrUnsigned1Frames |= (_image->getEndianess() << 8);
	EndianessBPSSignedOrUnsigned1Frames |= (_image->getBitsPerSample() << 9);
	EndianessBPSSignedOrUnsigned1Frames |= ((_image->getSignedOrUnsigned() == true ? 1 : 0) << 14);
	EndianessBPSSignedOrUnsigned1Frames |= ((numberOfFrames & 1) << 15);
	//Extract another 15 bits from the frames variable.
	Frames1Wasted = ((numberOfFrames >> 1)) & ((1 << 16) - 1);
	//Store 16 bits for the _k value. The value is stored with no decimals, and then it is divided by 1000 in the decoder engine.
	kFactor = ((int)(_kFactor * 1000));

	extraInformation[0] = sizeFirstHalf;
	extraInformation[1] = sizeSecondHalf;
	extraInformation[2] = codingPassesCB1Minus1CB2WLevels1;
	extraInformation[3] = ThreeWLevelsBitDepthWTypeQSize5;
	extraInformation[4] = QSize9ComponentsRGB7;
	extraInformation[5] = TwoComponentsRGBImageHeight;
	extraInformation[6] = EndianessBPSSignedOrUnsigned1Frames;
 	extraInformation[7] = Frames1Wasted;
	extraInformation[8] = kFactor;
}


/*
Function responsible of launching the generation of the bitstream and the prefix array used in the codestream generation.
*/
void BitStreamBuilder::createBitStream(int* totalSize, int prefixedArraySize, int cbSize, unsigned short* HExtraInformation, unsigned short* DBitStreamValues, int iter, cudaStream_t mainStream, int numberOfFrames)
{
	//We need 9 shorts to store general information (size of the image, cp, ...) and 2 shorts per codeblock processed by each warp which stores the size and the MSBP of each of those.
	//It has been chosen 4 bytes per codeblock because short structures are 2 bytes long.
	if (iter == 0)
		setExtraInformation(HExtraInformation, numberOfFrames);

	BSEngine *BSEng = new BSEngine(_image, DBitStreamValues, _DCodeStream, _DWTCodeBlockWidth, _DWTCodeBlockHeight, _prefixedArray, _sizeArray, NULL);

	BSEng->launchPrefixArrayGeneration(prefixedArraySize, _DTempStorageBytes, totalSize, mainStream, cbSize);
	BSEng->launchLUTBSGeneration(mainStream, _HLUTBSTableSteps, _DLUTBSTable, prefixedArraySize);
	BSEng->deviceMemoryAllocator(true, totalSize[0], HExtraInformation, NULL, iter, mainStream);
	BSEng->launchKernel(true, prefixedArraySize, cbSize, mainStream);
	delete BSEng;
}

/*
Functions which computes the length of the size array used in the decoding process.
*/
void BitStreamBuilder::retrieveSizeArray(int* sizeArray, unsigned short* hostData, int amountOfCodeblocks)
{
	int headerOffset = 9 + 1;
	int arrayPos = 0;
	int i = headerOffset;
	for (int arrayPos = 0; arrayPos < amountOfCodeblocks; arrayPos ++)
	{
		sizeArray[arrayPos] = hostData[i];
		i = i + 2;
	}
}

/*
Function responsible of managing the launch of the decoding process with its corresponding function to generate the prefix array used in the codestream decoding.
*/
void BitStreamBuilder::createCodeStream(int* DPrefixedArray, int* DSizeArray, int* HBasicInformation, int* DTempStoragePArray, unsigned short* DBitStreamValues, int* DCodeStreamValues, int* HSizeArray, int* HTotalBSSize, cudaStream_t mainStream, int HLUTBSTableSteps, int* DLUTBSTable)
{
	int frameSize = _image->getAdaptedHeight()*_image->getAdaptedWidth();
	_DWTCodeBlockHeight = HBasicInformation[2];
	_DWTCodeBlockWidth = HBasicInformation[3];

	int amountOfCodeblocks = ceil((float)frameSize / (float)_codeblockSizing);
	
	retrieveSizeArray(HSizeArray, _hostData, amountOfCodeblocks);
	
	BSEngine *BSEng = new BSEngine(_image, DBitStreamValues, DCodeStreamValues, _DWTCodeBlockWidth, _DWTCodeBlockHeight, DPrefixedArray, DSizeArray, HSizeArray);
	
	BSEng->deviceMemoryAllocator(false, _sizeOfStream, NULL, amountOfCodeblocks, NULL, mainStream);

	BSEng->launchPrefixArrayGeneration(amountOfCodeblocks, DTempStoragePArray, HTotalBSSize, mainStream, _codeblockSizing);
	BSEng->launchLUTBSGeneration(mainStream, HLUTBSTableSteps, DLUTBSTable, amountOfCodeblocks);
	BSEng->launchKernel(false, amountOfCodeblocks, _codeblockSizing, mainStream);

	delete BSEng;
}
