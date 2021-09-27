#pragma once
#include <string>

#include "Image.hpp"

Image::Image(std::string name, int width, int height, int components, int bitDepth, bool isRGB)
{
	_widthSize = width;
	_heightSize = height;
	_components = components;
	_bitDepth = bitDepth;
	_isRGB = isRGB;
	_filename = name;
}


//Constructor used for more complex images not being used as of today.
Image::Image(std::string name, int width, int height, int components, int bitDepth, bool isRGB, int endianType, int bps, bool signedOrNot)
{
	_widthSize = width;
	_heightSize = height;
	_components = components;
	_bitDepth = bitDepth;
	_isRGB = isRGB;
	_filename = name;
	_endianess = endianType;
	_bitsPerSample = bps;
	_signedOrUnsigned = signedOrNot;
}
Image::Image(){}

Image::~Image()
{
	//No dynamic allocation of memory, no need to put anything here.
}


void Image::setWidth(int w)
{
	_widthSize = w;
}

void Image::setHeight(int h)
{
	_heightSize = h;
}

void Image::setAdaptedWidth(int w)
{
	_adaptedWidthSize = w;
}

void Image::setAdaptedHeight(int h)
{
	_adaptedHeightSize = h;
}


void Image::setComponents(int comp)
{
	_components = comp;
}

void Image::setBitDepth(int bd)
{
	_bitDepth = bd;
}

void Image::setIsRGB(bool iRGB)
{
	_isRGB = iRGB;
}

void Image::setName(std::string fn)
{
	_filename = fn;
}

void Image::setEndianess(int e)
{
	_endianess = e;
}

void Image::setBitsPerSample(int bps)
{
	_bitsPerSample = bps;
}

void Image::setSignedOrUnsigned(bool sou)
{
	_signedOrUnsigned = sou;
}


int Image::getWidth() const
{
	return _widthSize;
}

int Image::getHeight() const
{
	return _heightSize;
}

int Image::getAdaptedWidth() const
{
	return _adaptedWidthSize;
}

int Image::getAdaptedHeight() const
{
	return _adaptedHeightSize;
}

int Image::getComponents() const
{
	return _components;
}

int Image::getBitDepth() const
{
	return _bitDepth;
}

bool Image::getIsRGB() const
{
	return _isRGB;
}

std::string Image::getName() const
{
	return _filename;
}

int Image::getEndianess() const
{
	return _endianess;
}

int Image::getBitsPerSample() const
{
	return _bitsPerSample;
}

bool Image::getSignedOrUnsigned() const
{
	return _signedOrUnsigned;
}