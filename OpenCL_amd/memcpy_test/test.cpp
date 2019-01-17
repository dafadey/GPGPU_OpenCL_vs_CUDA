#include "../helpers.h"
#include "data_type.h"

// tests for global memory access patterns in GPU
// coalesced r/w, channel and bank conflicts

bool check_resutls(const std::vector<real>& C)
{
  bool passed(true);
  int bads(0);
  int goods(0);
  for(size_t i(0); i != C.size(); i++)
  {
    if(C[i] != (real) i)
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

bool check_resutls(const std::vector<real>& C, const std::vector<real>& Cref)
{
  bool passed(true);
  int bads(0);
  int goods(0);
  for(size_t i(0); i != C.size(); i++)
  {
    if(C[i] != Cref[i])
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
  int wf_size=WF_SIZE;  // defined in data_types.h
  
  int wf_count=256;     // this is the number of wavefronts(OpenCL), thread blocks(CUDA)
                        // try other values to see how it works.
                        // 256, 512 are good choices for my Radeon 7970
                        // nVidia devices will typically require bigger numbers 512,1024,2048
                        // you can try non pow2 values too but it will be hard
                        // to control memory access patters in that case
  
  opencl ocl;
  if(!ocl.buildOCLsources("test.cl"))
    return -1;
  if(!ocl.add_kernel("simple_add"))
    return -1;
  if(!ocl.add_kernel("simple_add2"))
    return -1;
  if(!ocl.add_kernel("simple_add_NC"))
    return -1;
  if(!ocl.add_kernel("simple_add_NC_try"))
    return -1;
  if(!ocl.add_kernel("simple_add_stride_fix"))
    return -1;
  if(!ocl.add_kernel("simple_add_pattern1"))
    return -1;
  if(!ocl.add_kernel("simple_add_pattern2"))
    return -1;
  if(!ocl.add_kernel("simple_add_pattern2_bank_channel_fix"))
    return -1;
  
  opencl_data<real> ocl_data(ocl); // template wrapper works with data buffers
                                   // so has 

  size_t N = 1024 * 1024 * 64;
  // i use big pow2 number for 2 reasons
  // typically you cannot reach good performance due to overheads
  // with pow2 you do need to control address in the kernel function
  // it is not big overhead to add one if there but for testing puposes
  // i just took big perfectly aligned vector 
  std::vector<real> A;
  std::vector<real> B;
  std::vector<real> C;
  for(size_t i(0); i != N; i++)
  {
    A.push_back((real) 2.0 * (real) i);
    B.push_back(- (real) i);
  }
  C.resize(N);
  
  cl::Buffer dev_A(ocl_data.alloc(N*sizeof(real), opencl_data<real>::memType::RW));
  cl::Buffer dev_B(ocl_data.alloc(N*sizeof(real), opencl_data<real>::memType::RW));
  cl::Buffer dev_C(ocl_data.alloc(N*sizeof(real), opencl_data<real>::memType::RW));
  cl::Buffer dev_N(ocl_data.alloc(sizeof(size_t), opencl_data<real>::memType::R));
  
  ocl_data.h2d(A, dev_A);
  ocl_data.h2d(B, dev_B);
  ocl_data.h2d_manual((const void*) &N, sizeof(size_t), dev_N);
  
/* MC 1
* this is a simple aligned coalesced global i/o example
* pattern is trivial see coments in test.cl.
*/ 
  zero(C);
  ocl_data.h2d(C, dev_C);
  ocl_data.select_kernel("simple_add");
  ocl_data.set_arg<cl::Buffer>(0, dev_A);
  ocl_data.set_arg<cl::Buffer>(1, dev_B);
  ocl_data.set_arg<cl::Buffer>(2, dev_C);
  ocl_data.set_arg<size_t>(3, N);
  
  std::cout << "running kernel " << ocl_data.current_kernel_name << '\n';
  for(int i=-30; i!=300; i++) // -30 is for warmup
  {
    if(!ocl_data.run(wf_count, wf_size, i > 0)) // last argument enables timing count
      return -1;
  }

  std::cout << "getting result data\n";
  ocl_data.d2h(dev_C, C.data(), C.size());
  check_resutls(C);

/* MC 2
* this is a simple aligned coalesced global i/o example
* the only special thing about it is that it uses cl::Buffer dev_N to pass vector length
*/ 
  zero(C);
  ocl_data.h2d(C, dev_C);
  ocl_data.select_kernel("simple_add2");
  ocl_data.set_arg<cl::Buffer>(0, dev_A);
  ocl_data.set_arg<cl::Buffer>(1, dev_B);
  ocl_data.set_arg<cl::Buffer>(2, dev_C);
  ocl_data.set_arg<cl::Buffer>(3, dev_N);

  std::cout << "running kernel " << ocl_data.current_kernel_name << '\n';
  for(int i=-30; i!=300; i++)
  {
    if(!ocl_data.run(wf_count, wf_size, i > 0))
      return -1;
  }
  
  std::cout << "getting result data\n";
  ocl_data.d2h(dev_C, C.data(), C.size());
  check_resutls(C);

/* MC 3
 * the worst way to read/write global memory
 * see explanation in .cl file
 */
  zero(C);
  ocl_data.h2d(C, dev_C);
  ocl_data.select_kernel("simple_add_NC");
  ocl_data.set_arg<cl::Buffer>(0, dev_A);
  ocl_data.set_arg<cl::Buffer>(1, dev_B);
  ocl_data.set_arg<cl::Buffer>(2, dev_C);
  ocl_data.set_arg<size_t>(3, N);

  std::cout << "running kernel " << ocl_data.current_kernel_name << '\n';
  for(int i=-7; i!=10; i++)
  {
    if(!ocl_data.run(wf_count, wf_size, true))
      return -1;
  }

  std::cout << "getting result data\n";
  ocl_data.d2h(dev_C, C.data(), C.size());
  check_resutls(C);

/* MC 4
 * same thing with minor efforts to speed it up
 */
  zero(C);
  ocl_data.h2d(C, dev_C);
  ocl_data.select_kernel("simple_add_NC_try");
  ocl_data.set_arg<cl::Buffer>(0, dev_A);
  ocl_data.set_arg<cl::Buffer>(1, dev_B);
  ocl_data.set_arg<cl::Buffer>(2, dev_C);
  ocl_data.set_arg<size_t>(3, N);
  ocl_data.set_arg<size_t>(4, N / (wf_count * wf_size));

  std::cout << "running kernel " << ocl_data.current_kernel_name << '\n';
  for(int i=-7; i!=10; i++)
  {
    if(!ocl_data.run(wf_count, wf_size, true))
      return -1;
  }

  std::cout << "getting result data\n";
  ocl_data.d2h(dev_C, C.data(), C.size());
  check_resutls(C);

  
/* MC 5
 * simple copy with bad continous pattern but with a fix
 * that reduces channel and bank conflicts
 */
  zero(C);
  ocl_data.h2d(C, dev_C);
  ocl_data.select_kernel("simple_add_stride_fix");
  ocl_data.set_arg<cl::Buffer>(0, dev_A);
  ocl_data.set_arg<cl::Buffer>(1, dev_B);
  ocl_data.set_arg<cl::Buffer>(2, dev_C);
  ocl_data.set_arg<size_t>(3, N);

  std::cout << "running kernel " << ocl_data.current_kernel_name << '\n';
  for(int i=-7; i!=10; i++)
  {
    if(!ocl_data.run(wf_count, wf_size, true))
      return -1;
  }

  std::cout << "getting result data\n";
  ocl_data.d2h(dev_C, C.data(), C.size());
  check_resutls(C);

// MC 6 pattern1
  zero(C);
  ocl_data.h2d(C, dev_C);
  ocl_data.select_kernel("simple_add_pattern1");
  ocl_data.set_arg<cl::Buffer>(0, dev_A);
  ocl_data.set_arg<cl::Buffer>(1, dev_B);
  ocl_data.set_arg<cl::Buffer>(2, dev_C);
  ocl_data.set_arg<size_t>(3, N);

  std::cout << "running kernel " << ocl_data.current_kernel_name << '\n';
  for(int i=-30; i!=300; i++)
  {
    if(!ocl_data.run(wf_count, wf_size, true))
      return -1;
  }

  std::cout << "getting result data\n";
  ocl_data.d2h(dev_C, C.data(), C.size());
  check_resutls(C);


// MC 7 pattern2

  zero(C);
  ocl_data.h2d(C, dev_C);
  ocl_data.select_kernel("simple_add_pattern2");
  ocl_data.set_arg<cl::Buffer>(0, dev_A);
  ocl_data.set_arg<cl::Buffer>(1, dev_B);
  ocl_data.set_arg<cl::Buffer>(2, dev_C);
  ocl_data.set_arg<size_t>(3, N);

  std::cout << "running kernel " << ocl_data.current_kernel_name << '\n';
  for(int i=-30; i!=300; i++)
  {
    if(!ocl_data.run(wf_count, wf_size, true))
      return -1;
  }

  std::cout << "getting result data\n";
  ocl_data.d2h(dev_C, C.data(), C.size());
  check_resutls(C);

// MC 8 pattern2 with fix

  zero(C);
  ocl_data.h2d(C, dev_C);
  ocl_data.select_kernel("simple_add_pattern2_bank_channel_fix");
  ocl_data.set_arg<cl::Buffer>(0, dev_A);
  ocl_data.set_arg<cl::Buffer>(1, dev_B);
  ocl_data.set_arg<cl::Buffer>(2, dev_C);
  ocl_data.set_arg<size_t>(3, N);

  std::cout << "running kernel " << ocl_data.current_kernel_name << '\n';
  for(int i=-30; i!=300; i++)
  {
    if(!ocl_data.run(wf_count, wf_size, true))
      return -1;
  }

  std::cout << "getting result data\n";
  ocl_data.d2h(dev_C, C.data(), C.size());
  check_resutls(C);

  ocl_data.print_timings();


  return 0;
}
