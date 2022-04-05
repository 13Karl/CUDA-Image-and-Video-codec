#pragma once
#include <string>
#include "../IO/IOManager.hpp"
#include <cuda_runtime.h>
#include <cmath>

#ifndef GPU_HANDLE_ERROR
#define GPU_HANDLE_ERROR(ans) { SupportFunctions::gpuAssert((ans), __FILE__, __LINE__); }
#endif
#ifndef KERNEL_ERROR_HANDLER
#define KERNEL_ERROR_HANDLER { SupportFunctions::cudaKernelAssert( __FILE__, __LINE__);}
#endif

namespace SupportFunctions
{
	
	int refreshPercentageCompleted(int i, int total, int prevPercentage);
	bool isInteger(float k);
	int isPowerOfTwo(int x);
	void fixImageProportions(Image* img, int cblock_length, int cblock_width);
	void gpuAssert(cudaError_t err, const char *file, int line);
	void cudaKernelAssert(const char *file, int line);
	void markInitProfilerCPUSection(const char* threadName, const char* description);
	void markEndProfilerCPUSection();
}