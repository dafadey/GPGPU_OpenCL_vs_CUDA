#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>
#include "service.h"

extern "C"
void dev_alloc(int device, void** pparr, int sz)
{
	if(cudaSetDevice(device))
		printf("ERROR Initializing device #%d\n",device);
	else
	{
		checkCudaErrors( cudaMalloc( pparr, sz));
		cudaThreadSynchronize();
	}
}

extern "C"
void dev_h2d(int device, real* host_arr, real* dev_arr, int sz)
{
	//printf("\tcopying source of size %d from %p to %p\n", sz, host_arr, dev_arr);

	if(cudaSetDevice(device))
		printf("ERROR Initializing device #%d\n",device);
	else
	{
		checkCudaErrors( cudaMemcpy( dev_arr, host_arr, sz, cudaMemcpyHostToDevice));
		cudaThreadSynchronize();
	}
	//printf("\t\tdone\n");

}

extern "C"
void dev_d2h(int device, const real* dev_arr, real* host_arr, int sz)
{
	if(cudaSetDevice(device))
		printf("ERROR Initializing device #%d\n",device);
	else
	{
		checkCudaErrors( cudaMemcpy( host_arr, dev_arr, sz, cudaMemcpyDeviceToHost));
		cudaThreadSynchronize();
	}
}

