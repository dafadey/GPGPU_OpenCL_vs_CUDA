#include "../helpers.h"
#include "data_type.h"
#include "cmath"

// global memory access takes 400-600 GPU cycles
// can we do something while waiting for data
// not during the whole global i/o flow but we have free place to do some math!
// in this simple test we do quite slow operation division in the inner loop
// inside kernel
// see data.dat (0 means code was rebuild with IMAX=1 and '/' vhanged to '*' which is cheap,
// you can plot data from file or get your results and see that some of first
// divisions do not influence much on kernel run time 

bool check_resutls(const std::vector<real>& C, const std::vector<real>& Cref)
{
  bool passed(true);
  int bads(0);
  int goods(0);
  for(size_t i(0); i != C.size(); i++)
  {
    if(std::abs(C[i] - Cref[i]) > 0.001 * (std::abs(C[i])+std::abs(Cref[i])))
    {
      bads++;
      passed = false;
    }
    else
      goods++;
  }
  std::cout << (passed ? "test passed" : "test failed, bads:" + std::to_string(bads) + ", goods:" + std::to_string(goods)) << '\n';
  return passed;
}

void zero(std::vector<real>& C)
{
  for(auto& v : C)
    v = (real) 0;
}

int main()
{
  int wf_size=64;
  int wf_count=256;
  
  opencl ocl;
  if(!ocl.buildOCLsources("test.cl"))
    return -1;
  if(!ocl.add_kernel("simple_add"))
    return -1;
  
  opencl_data<real> ocl_data(ocl);

  size_t N = 1024 * 1024 * 64;
  std::vector<real> A(N);
  std::vector<real> B(N);
  std::vector<real> C(N);
  std::vector<real> Cref(N);
  for(size_t j(0); j != N; j++)
  {
    A[j] = sin((real) 0.1 * (real) j);
    B[j] = cos((real) 0.1 * (real) j);
    
    real res = (real) 0;
    real b = (real) 2.2;
    for(int i = 0; i != IMAX; i++)
    {
      b += (real) 0.1;
      res+= A[j] / (b + B[j]);
    }
    Cref[j] = res;
  }
  
  cl::Buffer dev_A(ocl_data.alloc(N*sizeof(real), opencl_data<real>::memType::RW));
  cl::Buffer dev_B(ocl_data.alloc(N*sizeof(real), opencl_data<real>::memType::RW));
  cl::Buffer dev_C(ocl_data.alloc(N*sizeof(real), opencl_data<real>::memType::RW));
  
  ocl_data.h2d(A, dev_A);
  ocl_data.h2d(B, dev_B);
  
// MC 1
  zero(C);
  ocl_data.h2d(C, dev_C);
  ocl_data.select_kernel("simple_add");
  ocl_data.set_arg<cl::Buffer>(0, dev_A);
  ocl_data.set_arg<cl::Buffer>(1, dev_B);
  ocl_data.set_arg<cl::Buffer>(2, dev_C);
  ocl_data.set_arg<size_t>(3, N);
  
  std::cout << "running kernel " << ocl_data.current_kernel_name << '\n';
  for(int i=-7; i!=300; i++)
  {
    if(!ocl_data.run(wf_count, wf_size, i > 0))
      return -1;
  }

  std::cout << "getting result data\n";
  ocl_data.d2h(dev_C, C.data(), C.size());
  check_resutls(C, Cref);

  ocl_data.print_timings();


  return 0;
}
