#pragma once
#include <string>

#ifndef IMAGE_HPP
#define IMAGE_HPP


class Image

{

public:

	//Constructor
	Image();
	Image(std::string name, int width, int height, int components, int bitDepth, bool isRGB);
	Image(std::string name, int width, int height, int components, int bitDepth, bool isRGB, int endianType, int bps, bool signedOrNot);
	//Destructor
	~Image();

	//Set Functions

	void setWidth(int w);
	void setAdaptedWidth(int w);
	void setAdaptedHeight(int h);
	void setHeight(int h);
	void setComponents(int comp);
	void setBitDepth(int bd);
	void setIsRGB(bool iRGB);
	void setName(std::string fn);
	void setEndianess(int e);
	void setBitsPerSample(int bps);
	void setSignedOrUnsigned(bool sou);

	//Get Functions
	int getWidth() const;
	int getAdaptedWidth() const;
	int getAdaptedHeight() const;
	int getHeight() const;
	int getComponents() const;
	int getBitDepth() const;
	bool getIsRGB() const;
	std::string getName() const;
	int getEndianess() const;
	int getBitsPerSample() const;
	bool getSignedOrUnsigned() const;



private:

	int _widthSize;
	int _adaptedWidthSize;
	int _adaptedHeightSize;
	int _heightSize;
	int _components;
	int _bitDepth;
	bool _isRGB;
	std::string _filename;
	int _endianess;
	int _bitsPerSample;
	bool _signedOrUnsigned;

	
};

#endif