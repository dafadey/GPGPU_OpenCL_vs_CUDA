#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// extension to use double values. you can change 'real' typedef in data_type.h.
// extension name can be found in device info which is printed out bu these examples.

#include "data_type.h"

/*  access pattern:
 *               i=0                                i=block_sz.....
 *   CU0        CU1       ...    CU256           CU0    ...........
 * 0     63  0      63          0      63      0      63...........
 * ||||...|  ||||....|    ...   ||||....|      ||||....|...........
 * 0     63 64      127       16320   16383  16384 ................
 */
void kernel simple_add(global const real* A, global const real* B, global real* C, size_t n)
{
  const size_t li = get_local_id(0) + get_group_id(0) * get_local_size(0);//get_global_id(0);
  const size_t block_sz = get_global_size(0);
  for(size_t i=0; i < n; i += block_sz)
  {
    const size_t id = li + i;
    C[id] = A[id] + B[id];
  }
}

/*
 * same as above, last parameter sent as cl::Buffer
 */
void kernel simple_add2(global const real* A, global const real* B, global real* C, global const size_t* N)
{
  const size_t n = N[0];
  const size_t li = get_global_id(0);
  const size_t block_sz = get_global_size(0);
  for(size_t i=0; i < n; i += block_sz)
  {
    const size_t id = li + i;
    C[id] = A[id] + B[id];
  }
}

/*
 * worst memory access pattern
 * _____________________CU0_____________________     ____CU1______   ...
 * |                                            |   |             |
 *     th0            th1         ....    th63
 * ||||||......|   ||||||.....|         ||||....|   ||||.....|....   ...
 * 0          b-1  b         2b-1      63b   64b-1  64b ..........   ...
 * b = n / 64(threads) * 256(blocks)
 * NOTE: 1. all threads use just one float while reading a line of floats
 *       2. due to pow of 2 ALL in every CU takes the same bank and same memory channel (8th,9th,... etc bits of address)
 * this is actually the analog of uniform random access in CPU codes
 */
void kernel simple_add_NC(global const real* A, global const real* B, global real* C, size_t n)
{
  size_t li = get_global_id(0);
  size_t block_sz = get_global_size(0);
  size_t bulk = n / block_sz;
  for(size_t i=0; i < bulk; i++)
  {
    size_t id = li * bulk + i;
    C[id] = A[id] + B[id];
  }
}

/*
 * same case but let us try to save something on division
 * actually silly idea. does not save anything in the end
 */
void kernel simple_add_NC_try(global const real* A, global const real* B, global real* C, size_t n, size_t bulk)
{
  size_t li = get_global_id(0);
  size_t block_sz = get_global_size(0);
  for(size_t i=0; i < bulk; i++)
  {
    size_t id = li * bulk + i;
    C[id] = A[id] + B[id];
  }
}

/*
 * let us try to resolve at least channel and bank conflicts
 * each thread still reads values one by one but each thread has individual shift
 * new_address = (old_address + (thread_index + block_index * number_of_threads(64))) % bulk
 * at first cycle over i
 * CU0 th0 reads   [0]   channel0
 * CU0 th1 reads   [1]   channel0
 * ...
 * CU0 th63 reads  [63]  channel0
 * CU1 th0  reads  [64]  channel0
 * ...
 * CU4 th0 reads   [256] channel1 !!! ok CU4 works with new channel!
 * ...
 * 
 * having 256 CU-s running simulataneously we will loop over 12 channels 256/4/121 ~ 5 times
 * never do this. such access pattern is still very bad.
 * even if each of your next thread will use new differen channel you will still miss data that you read
 * and this loss is cruicial for banwidth performance
 */
void kernel simple_add_stride_fix(global const real* A, global const real* B, global real* C, size_t n)
{
  // here we try to do a channed and bank fix
  const size_t li = get_global_id(0);
  const size_t block_sz = get_global_size(0);
  const size_t bulk = n / block_sz;
  for(size_t i=0; i < bulk; i++)
  {
    size_t id = li * bulk + (i + li) % bulk;
    C[id] = A[id] + B[id];
  }
}

// with next kernels we are trying to partition our vector one level deep.
// we will use DIMY as a parameter of additional partitioning
#define DIMY 16

/*
 * in this pattern we will read basically the same way as in first one
 * here tid+wfId = thread_index + block_index * thread_number(64)
 * other part of address 'i+j' is advanced by get_global_size(0) = thread_number(64) * block_number(256)
 * for some reason doing this in two nested loops is slower this is still a question
 * note we have coalesced read without channel or bank conflicts
 */
void kernel simple_add_pattern1(global const real* A, global const real* B, global real* C, size_t n)
{
  size_t tid = get_local_id(0);
  size_t wfId = get_group_id(0) * WF_SIZE;
  const size_t bigstep = DIMY * get_global_size(0);
  
  for(size_t j = 0; j != n; j += bigstep)
  {
    for(size_t i = 0; i != bigstep; i+=get_global_size(0))
    {
      const size_t addr = tid + wfId + i + j;
      C[addr] = A[addr] + B[addr];
    }
  }
}

/*
 * this time we will make a strided acces
 * to stay coalesced we will read thread_number(64) aligned continous bulks of values DIMY times one by one
 * so for example th0:CU0 and th0:CU1 will access locations 0 and (WF_SIZE*DIMY = 64*16 = 1024*4(bytes for float) = 4096)
 * for Radeon 7970 channel coded with address bits 10,9,8. So forget about th0, all threads within CUx
 * will access same channel sine 64 * f(size of float) = 256 (8lower bits bits). Just keep this in mind.
 * Ok, for sure CU0 will be with channel 0.
 * to calculate CU1's channel you should refer to Figure 2.1 of AMD_OpenCL_Programming_Developer_Optimization_Guide2.pdf
 * from there it follows that CU1 will be with channel 0 again.
 * does it always happen like that?
 * almost every time - yes. just take a brief look at table 2.1 and you will see that you can use just channels 0 and 1
 * with corresponding probabilities 0.6[6] and 0.3[3] with this access pattern. for example channel 1 will be used bu CU2. 
 */
void kernel simple_add_pattern2(global const real* A, global const real* B, global real* C, size_t n)
{
  const size_t tid = get_local_id(0);
  const size_t bigstep = get_global_size(0) * DIMY;
  const size_t bulk_sz = WF_SIZE * DIMY;
  const size_t wfId = get_group_id(0) * bulk_sz;
  
  for(size_t j = 0; j < n; j += bigstep)
  {
    for(size_t _i = 0; _i < DIMY; _i++)
    {
      const size_t addr = tid + wfId + _i * WF_SIZE + j; 
      C[addr] = A[addr] + B[addr];
    }
  }
}

/*
 * Can we fix channel problem?
 * Yes! compare the kernels below and above. They differ in just one line.
 * we calculate individual CUx offset not as trivial as i * WF_SIZE(64)
 * but with the following rule (_i + get_group_id(0)) % DIMY) * WF_SIZE.
 * at first iteration of inner loop CU0 takes channel 0, CU1 takes NEXT channel 2
 * (yes 2! do not be distracted by this if you still did not looked into Figure 2.1
 * AMD has some non trivial rule to calculate channel index)
 * then CU2 will take channel 3 and so on. DIMY is 16 number of channels is 12
 * we will take each of them at least once, while some channels will be hit twice.
 * For some reason it works not so well. Possibly this approach also hits bank conflict.
 * The rule to calculate bank index is not evident so all this improvements are not straightforward.
 * 
 * Nethertheless % operation allows simple solution to sove bank/channel conflicts
 * while all data is still accessible and we stay within desired memory block. 
 */
void kernel simple_add_pattern2_bank_channel_fix(global const real* A, global const real* B, global real* C, size_t n)
{
  const size_t tid = get_local_id(0);
  const size_t bigstep = get_global_size(0) * DIMY;
  const size_t bulk_sz = WF_SIZE * DIMY;
  const size_t wfId = get_group_id(0) * bulk_sz;
  
  for(size_t j = 0; j < n; j += bigstep)
  {
    for(size_t _i = 0; _i < DIMY; _i++)
    {
      const size_t i = ((_i + get_group_id(0)) % DIMY) * WF_SIZE;
      const size_t addr = tid + wfId + i + j; 
      C[addr] = A[addr] + B[addr];
    }
  }
}

