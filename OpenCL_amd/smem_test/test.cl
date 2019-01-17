#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// extension to use double values. you can change 'real' typedef in data_type.h.
// extension name can be found in device info which is printed out bu these examples.
#include "data_type.h"

//void kernel partial_integ(global const real* A, global real* B, unsigned int n, local real* smem, unsigned int DIMY, global real* tmp)

//#define blockDim WF_SIZE

//void kernel partial_integ(global const real* A, global real* B, unsigned int n, global real* tmp)
void kernel partial_integ(global const real* A, global real* B, unsigned int n, local real* smem, global real* tmp)
{
  //local smem[WF_SIZE * (DIMY+1) * sizeof(real)]; // another way to ask for local mem
  
  const unsigned int tid = get_local_id(0);
  #ifndef blockDim
    const unsigned int blockDim = get_local_size(0);
  #endif
  const unsigned int bulk_sz = blockDim * DIMY;
  const unsigned int wfId = get_group_id(0) * bulk_sz;
  const unsigned int bigstep = get_global_size(0) * DIMY;
  
  
  for(unsigned int j = 0; j != n; j += bigstep)
  {
    // put DIMY * WF_SIZE = 1024 items to localmemory of device
    // we do coalesced reads and place data in local memory using
    // some specific pattern:
    // 16 values,17-the empty, more 16 values, 34-th empty and so on.
    // see below why.
    unsigned int i = 0;
    for(unsigned int _i = 0; _i < DIMY; _i++)
//    for(unsigned int i = 0; i != bulk_sz; i += blockDim) // here the above approach makes faster code
    {
      unsigned int la = (((tid+i) / DIMY) * (DIMY + 1)) + ((tid + i) % DIMY);
      smem[la] = A[tid + i + wfId + j];
      i+=blockDim;
    }

    // small partial integ
    // small means we calculate partial sums of size DIMY (16)
    // we have no bank conflicts here because of our specific pattern.
    // try tool bank_calc.cpp to see how memory is accessed
    barrier(CLK_LOCAL_MEM_FENCE); // we will use data written by another
                                  // thread so wait until all threads
                                  // will reach this point
    const unsigned int offset = tid * (DIMY + 1);
    i=1;
    for(;i < DIMY; i++)
      smem[i + offset] += smem[i - 1 + offset];
    smem[DIMY + offset] = smem[DIMY - 1 + offset]; // make some profit out of 17-th empty spaces
    
    //integ biases
    barrier(CLK_LOCAL_MEM_FENCE); // do not forget to sycronize threads
    
    if(tid == 0) // all others threads are waiting
    {
      i=1;
      for(; i < WF_SIZE; i++) // !!! HERE COMPILER NEEDS TO KNOW THEVALUE OF blockDim TO MAKE FASTER CODE (try get_local_size() and watch the performance drop, i have no clear explanation for that, guess to understand this trick intermediate (GPU assembler) code should be studied)
        smem[DIMY + i * (DIMY + 1)] += smem[DIMY + (i - 1) * (DIMY + 1)];
    }
    // now we collected offsets for out small parts in 17-th empty spaces
    
    //bigger portion intergation
    // just add offsets to samll partial results
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tid != 0)
    {
      i=0;
      for(; i < DIMY; i++)
        smem[i + offset] += smem[offset - 1];// explanation : smem[i + tid * (DIMY + 1)] += smem[DIMY + (tid - 1) * (DIMY + 1)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // great! we have big partial sum of size DIMY * WF_SIZE
    // since we have no way to communicate CU-s dump data to global and do the rest of job in different kernel
    
    //dump data to temporary array
    if(tid == 0)
      tmp[get_group_id(0) + (j / bigstep) * get_num_groups(0)] = smem[DIMY - 1 + (blockDim - 1) * (DIMY + 1)];

    i = 0;
    for(unsigned int _i = 0; _i < DIMY; _i++)
    //for(unsigned int i = 0; i != bulk_sz; i += blockDim) //gives equal performance even if compiler does not know blockDim
    {
      unsigned int la = (((tid+i) / DIMY) * (DIMY + 1)) + ((tid+i) % DIMY);
      B[tid + i + wfId + j] = smem[la];
      i+=blockDim;
    }
  }
}

// quite trivial kernel nothing to comment here
//void kernel integ_merge(global real* B, global const real* tmp, unsigned int n, unsigned int DIMY)
void kernel integ_merge(global real* B, global const real* tmp, unsigned int n)
{
  const unsigned int tid = get_local_id(0);
  #ifndef blockDim
    const unsigned int blockDim = get_local_size(0);
  #endif
  const unsigned int bulk_sz = blockDim * DIMY;
  const unsigned int wfId = get_group_id(0) * bulk_sz;
  const unsigned int bigstep = get_global_size(0) * DIMY;

  for(unsigned int j = 0; j != n; j += bigstep)
  {
    unsigned int addr = get_group_id(0) + (j / bigstep) * get_num_groups(0);
    const real bias = addr > 0 ? tmp[addr - 1] : (real) 0;
    unsigned int i = 0;
    for(unsigned int _i = 0; _i < DIMY; _i++)
    {
      B[tid + i + wfId + j] += bias;
      //B[tid + ((i+blockDim*get_group_id(0)) % (bulk_sz)) + wfId + j] += bias;
      i+=blockDim;
    }
    /*explanation:
    * actually the following loop
    *  for(unsigned int i = 0; i < block_sz; i+=blockDim)
    * gives the same performance
    * even if compiler does not know bockDim value
    * but it does know DIMY so could possibly analyze and unwrap and/or vectorize
    */
  }
}
