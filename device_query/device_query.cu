#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
/*
struct CDP {
	char name[256];
	size_t totalGlobalMem;
	size_t sharedMemPerBlock;
	int regsPerBlock;
	int warpSize;
	size_t memPitch;
	int maxThreadsPerBlock;
	int maxThreadsDim[3];
	int maxGridSize[3];
	size_t totalConstMem;
	int major;
	int minor;
	int clockRate;
	size_t textureAlignment;
	int deviceOverlap;
	int multiProcessorCount;
	int kernelExecTimeoutEnabled;
	int integrated;
	int canMapHostMemory;
	int computeMode;
	int maxTexture1D;
	int maxTexture2D[2];
	int maxTexture3D[3];
	int maxTexture2DArray[3];
	int concurrentKernels;

};
*/
int main( void ) {
	cudaDeviceProp prop;

	int i;
	int count;
	cudaGetDeviceCount (&count);
	for(i =0; i<count; ++i) {
		cudaGetDeviceProperties( &prop, i );
		printf("   ---- General Information for Device %d ---\n", i);
		printf("Name :                         %s\n", prop.name);
		printf("Compute Capability :           %d.%d/n", prop.major, prop.minor);
		printf("Clock Rate :                   %d\n", prop.clockRate);
		printf("Device Copy Overlap: " );
		if(prop.deviceOverlap)
			printf(              "          Enabled\n");
		else
			printf(              "          Disabled\n");
		printf("Kernel Execution Timeout : ");
		if(prop.kernelExecTimeoutEnabled)
			printf(                    "    Enabled\n");
		else
			printf(                    "    Disabled\n");

		printf("\n  ---- Memory Information for Device %d ---\n", i);
		printf("Total Global Memory:           %ld\n", prop.totalGlobalMem);
		printf("Total Constant Memory:         %ld\n", prop.totalConstMem);
		printf("Max mem pitch:                 %ld\n", prop.memPitch);
		printf("Texture Alignment:             %ld\n", prop.textureAlignment);
		printf("\n    --- MP Information for Device %d ---\n", i);
		printf("Multiprocessor Count:          %d\n", prop.multiProcessorCount);
		printf("Shared mem per mp:             %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp :             %d\n", prop.regsPerBlock);
		printf("Threads in warp:               %d\n", prop.warpSize);
		printf("Max threads per block:         %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions:         (%d %d %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max Grid Dimensions:           (%d %d %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\n");
	}
}
