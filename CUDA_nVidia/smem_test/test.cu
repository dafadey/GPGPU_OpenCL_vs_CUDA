#include "defaults.h"

//only first part of algorith is implemented
__global__ void iteg_partial(const real* A, real* B, real* tmp, unsigned int n)
{
	__shared__ real smem[WF_SIZE * (DIMY + 1)];

	const unsigned int bulk_sz = DIMY * WF_SIZE;
	const unsigned int wfId = bulk_sz * blockIdx.x;
	const unsigned int bigstep = WF_SIZE * WF_COUNT * DIMY;
	const unsigned int tid = threadIdx.x;
	for(unsigned int j = 0; j != n; j += bigstep)
	{
		for(unsigned int i = 0; i < bulk_sz; i += WF_SIZE)
		{
			unsigned int la = (((tid + i) / DIMY) * (DIMY + 1)) + ((tid + i) % DIMY);
			smem[la] = A[tid + i + wfId + j];
		}
		
		__syncthreads();

		const unsigned offset = tid * (DIMY + 1);
		for(unsigned int i = 1; i < DIMY; i++)
			smem[i + offset] += smem[i - 1 + offset];
		smem[DIMY + offset] = smem[DIMY -1 + offset];
		__syncthreads();

		if(tid == 0)
		{
			for(unsigned int i=1; i < WF_SIZE; i++)
				smem[DIMY + i * (DIMY + 1)] += smem[DIMY + (i - 1) * (DIMY + 1)];
		}
		__syncthreads();
		
		if(tid != 0)
		{
			for(unsigned int i=0; i < DIMY; i++)
				smem[i + offset] += smem[offset - 1];
		}
		__syncthreads();
		
		if(tid == 0)
			tmp[blockIdx.x + j / (DIMY * WF_SIZE)] = smem[DIMY - 1 + (WF_SIZE - 1) * (DIMY + 1)];
		
		for(unsigned int i = 0; i < bulk_sz; i += WF_SIZE)
		{
			unsigned int la = (((tid + i) / DIMY) * (DIMY + 1)) + ((tid + i) % DIMY);
			B[tid + i + wfId + j] = smem[la];
		}
	}
}

extern "C"
void integrate_partial(const real* A, real* B, real* tmp, unsigned int n)
{
	dim3 grid(WF_COUNT,1,1);
	dim3 threads(WF_SIZE,1,1);
	iteg_partial<<<grid, threads>>>(A, B, tmp, n);
	cudaThreadSynchronize();
}
