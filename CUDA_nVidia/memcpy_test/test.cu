#include "defaults.h"

__global__ void kernel_simple_copy_add(real* A, real* B, real* C, size_t n)
{
	register const size_t addr = threadIdx.x + WF_SIZE * blockIdx.x;

	for(register size_t j=0; j < n; j += WF_SIZE * WF_COUNT)
		C[addr + j] = A[addr + j] + B[addr + j];
	
}

__global__ void kernel_simple_copy_add_stride(real* A, real* B, real* C, size_t n)
{
	register const size_t addr = threadIdx.x + WF_SIZE * DIMY * blockIdx.x;

	for(register size_t j=0; j < n; j += WF_SIZE * WF_COUNT * DIMY)
	{
		for(register size_t i=0; i != DIMY * WF_SIZE; i+=WF_SIZE)
			C[addr + i + j] = A[addr + i + j] + B[addr + i + j];
	}
}

__global__ void kernel_simple_copy_add_NC(real* A, real* B, real* C, size_t n)
{
	register const size_t block_sz = WF_SIZE * WF_COUNT;
	register const size_t bulk = n / (WF_SIZE * WF_COUNT);
	register const size_t id = threadIdx.x + blockIdx.x * WF_SIZE;

	for(register size_t i=0; i != bulk; i++)
	{
		register const size_t addr = id * bulk + i;
		C[addr] = A[addr] + B[addr];
	}
}

__global__ void kernel_simple_copy_add_NC_fix(real* A, real* B, real* C, size_t n)
{
	register const size_t block_sz = WF_SIZE * WF_COUNT;
	register const size_t bulk = n / (WF_SIZE * WF_COUNT);
	register const size_t id = threadIdx.x + blockIdx.x * WF_SIZE;

	for(register size_t i=0; i != bulk; i++)
	{
		register const size_t addr = id * bulk + (i + id) % bulk;
		C[addr] = A[addr] + B[addr];
	}
}

extern "C"
void simple_copy_add(const real* A, const real* B, real* C, unsigned int n)
{
	dim3 grid(WF_COUNT,1,1);
	dim3 threads(WF_SIZE,1,1);
	kernel_simple_copy_add<<<grid, threads>>>((float*) A, (float*) B, C, (size_t) n);
	cudaThreadSynchronize();
}

extern "C"
void simple_copy_add_stride(const real* A, const real* B, real* C, unsigned int n)
{
	dim3 grid(WF_COUNT,1,1);
	dim3 threads(WF_SIZE,1,1);
	kernel_simple_copy_add_stride<<<grid, threads>>>((float*) A, (float*) B, C, (size_t) n);
	cudaThreadSynchronize();
}

extern "C"
void simple_copy_add_NC(const real* A, const real* B, real* C, unsigned int n)
{
	dim3 grid(WF_COUNT,1,1);
	dim3 threads(WF_SIZE,1,1);
	kernel_simple_copy_add_NC<<<grid, threads>>>((float*) A, (float*) B, C, (size_t) n);
	cudaThreadSynchronize();
}

extern "C"
void simple_copy_add_NC_fix(const real* A, const real* B, real* C, unsigned int n)
{
	dim3 grid(WF_COUNT,1,1);
	dim3 threads(WF_SIZE,1,1);
	kernel_simple_copy_add_NC_fix<<<grid, threads>>>((float*) A, (float*) B, C, (size_t) n);
	cudaThreadSynchronize();
}
