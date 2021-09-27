#include "AuxiliarFunctions.hpp"

bool SupportFunctions::isInteger(float k)
{
	return std::floor(k) == k;
}

int SupportFunctions::isPowerOfTwo(int x)
{
	int bitsOnOne = 0;

	while (x && bitsOnOne <= 1)
	{
		if ((x & 1) == 1)
			bitsOnOne++;
		x >>= 1;
	}

	return (bitsOnOne == 1);
}

void SupportFunctions::fixImageProportions(Image* img, int cblock_length, int cblock_width)
{
	img->setAdaptedHeight(ceil((float)img->getHeight()/ cblock_length)*cblock_length);
	img->setAdaptedWidth(ceil((float)img->getWidth()/ cblock_width)*cblock_width);
}

int SupportFunctions::refreshPercentageCompleted(int i, int total, int prevPercentage)
{
	int percentage = floor(((float)(i + 1) / (float)total) * 100);
	if (percentage != prevPercentage)
		std::cout << "\rCurrent Progress: " << percentage << "%." << std::flush;
	return prevPercentage;
}




void SupportFunctions::gpuAssert(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
		exit(EXIT_FAILURE);
	}
}


void SupportFunctions::cudaKernelAssert(const char *file, int line)
{
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		std::cout << "cudaKernelAssert() failed at " << file << ":" << line << ":" << cudaGetErrorString(err) << std::endl;
		exit(-1);
	}

}

void SupportFunctions::markInitProfilerCPUSection(const char* threadName, const char* description)
{
	nvtxNameOsThreadA(std::hash<std::thread::id>()(std::this_thread::get_id()), threadName);
	nvtxRangePush(description);
	nvtxMark("Waiting...");
}

void SupportFunctions::markEndProfilerCPUSection()
{
	nvtxRangePop();
}