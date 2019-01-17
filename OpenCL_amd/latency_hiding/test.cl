#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// extension to use double values. you can change 'real' typedef in data_type.h.
// extension name can be found in device info which is printed out bu these examples.
#include "data_type.h"
// kernel is trivial
// it does some crazy math just to force GPU to use division as many times as we want
// since input data in sin and cos 2.2> + data is > 1.2 and will not cauze div by zero
void kernel simple_add(global const real* A, global const real* B, global real* C, size_t n)
{
  for(size_t i=0; i < n; i += get_global_size(0))
  {
    const size_t id = get_global_id(0) + i;
    real res = (real) 0;
    real bias = (real) 2.2;
    const real a = A[id];
    const real b = B[id];
    for(int i = 0; i != IMAX; i++)
    {
      bias += (real) 0.1;
      res += a / (bias + b);
    }
    C[id] = res;
  }
}
