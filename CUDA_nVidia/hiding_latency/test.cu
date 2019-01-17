#include "defaults.h"

__global__ void kernel_simple_copy_add_base(real* A, real* B, real* C, size_t n)
{
	register const size_t addr = threadIdx.x + WF_SIZE * blockIdx.x;

	for(register size_t j=0; j < n; j += WF_SIZE * WF_COUNT)
			C[addr + j] = A[addr + j] + B[addr + j];
}

__global__ void kernel_simple_copy_add(real* A, real* B, real* C, size_t n)
{
	register const size_t addr = threadIdx.x + WF_SIZE * blockIdx.x;

	for(register size_t j=0; j < n; j += WF_SIZE * WF_COUNT)
	{
			register const real a = A[addr + j];
			register const real b = B[addr + j];
			register real bias = (real) 2.2;
			register real res = (real) 0.0;
			for(unsigned int i=0; i!=IMAX; i++)
			{
				bias += (real) 0.1;
				res += a / (bias + b);
			}
			C[addr + j] = res;
	}
}

__global__ void kernel_simple_copy_add_base_stride(real* A, real* B, real* C, size_t n)
{
	register const size_t addr = threadIdx.x + WF_SIZE * DIMY * blockIdx.x;

	for(register size_t j=0; j < n; j += WF_SIZE * WF_COUNT * DIMY)
	{
		for(register size_t i=0; i != DIMY * WF_SIZE; i+=WF_SIZE)
			C[addr + i + j] = A[addr + i + j] + B[addr + i + j];
	}
}

__global__ void kernel_simple_copy_add_stride(real* A, real* B, real* C, size_t n)
{
	register const size_t addr = threadIdx.x + WF_SIZE * DIMY * blockIdx.x;

	for(register size_t j=0; j < n; j += WF_SIZE * WF_COUNT * DIMY)
	{
		for(register size_t i=0; i != DIMY * WF_SIZE; i+=WF_SIZE)
		{
			register const real a = A[addr + i + j];
			register const real b = B[addr + i + j];
			register real bias = (real) 2.2;
			register real res = (real) 0.0;
			for(unsigned int i=0; i!=IMAX; i++)
			{
				bias += (real) 0.1;
				res += a / (bias + b);
			}
			C[addr + i + j] = res;
		}
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
void simple_copy_add_base(const real* A, const real* B, real* C, unsigned int n)
{
	dim3 grid(WF_COUNT,1,1);
	dim3 threads(WF_SIZE,1,1);
	kernel_simple_copy_add_base<<<grid, threads>>>((float*) A, (float*) B, C, (size_t) n);
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
void simple_copy_add_base_stride(const real* A, const real* B, real* C, unsigned int n)
{
	dim3 grid(WF_COUNT,1,1);
	dim3 threads(WF_SIZE,1,1);
	kernel_simple_copy_add_base_stride<<<grid, threads>>>((float*) A, (float*) B, C, (size_t) n);
	cudaThreadSynchronize();
}
