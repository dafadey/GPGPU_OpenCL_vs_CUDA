#include "../helpers.h"
#include "data_type.h"
#include "fstream"
#include "cmath"

// integral calculation for vector with GPU
// local/chared memory demonstration

bool check_resutls(const std::vector<real>& C, const std::vector<real>& Cref)
{
  bool passed(true);
  int bads(0);
  int goods(0);
  for(size_t i(0); i != C.size(); i++)
  {
    if(std::abs(C[i] - Cref[i]) > (real) 0.1)
    {
      //std::cerr << "bad at " << i << " " << C[i] << "!=" << Cref[i] << '\n';
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
  unsigned int wf_size=WF_SIZE; // defined in data_types.h
  
  unsigned int wf_count=256;    // this is the number of wavefronts(OpenCL), thread blocks(CUDA)
                                // try other values to see how it works.
                                // 256, 512 are good choices for my Radeon 7970
                                // nVidia devices will typically require bigger numbers 512,1024,2048
                                // you can try non pow2 values too but it will be hard
                                // to control memory access patters in that case

  opencl ocl;
  if(!ocl.buildOCLsources("test.cl"))
    return -1;
  if(!ocl.add_kernel("partial_integ"))
    return -1;
  if(!ocl.add_kernel("integ_merge"))
    return -1;
  
  opencl_data<real> ocl_data(ocl);

  size_t N = 1024 * 1024 * 64;
  std::vector<real> A;
  std::vector<real> B;
  B.resize(N);
  std::vector<real> Bref;
  Bref.resize(N);
  std::vector<real> Bref_partial;
  Bref_partial.resize(N);
  for(size_t i(0); i != N; i++)
  {
    //A.push_back((real) 0.25 * (real) i);
    A.push_back(sin((real) 0.01 * (real) i));
  }


  for(int j=-1;j!=3;j++)
  {
    for(size_t i(0); i != N; i++)
      Bref[i]= (real) 0;
    if(j>0)
      ocl_data.timer_start();
    Bref[0] = A[0];
    for(size_t i(1); i != N; i++)
      Bref[i] = Bref[i-1] + A[i];
    if(j>0)
      ocl_data.timer_stop("PURE_CPU_integ");
  }

  for(size_t i(0); i != N; i++)
    Bref_partial[i]=A[i];
  for(size_t i(0); i != N; i += wf_size * DIMY)
  {
    for(size_t j(1); j != wf_size * DIMY; j++)
      Bref_partial[j + i] += Bref_partial[i + j - 1];
  }
  
  
  cl::Buffer dev_A(ocl_data.alloc(N*sizeof(real), opencl_data<real>::memType::RW));
  cl::Buffer dev_B(ocl_data.alloc(N*sizeof(real), opencl_data<real>::memType::RW));
  cl::Buffer dev_tmp(ocl_data.alloc((N / (wf_size * DIMY)) * sizeof(real), opencl_data<real>::memType::RW));
  std::vector<real> tmp;
  tmp.resize(N / (wf_size * DIMY));
  ocl_data.h2d(A, dev_A);

  zero(B);
  ocl_data.h2d(B, dev_B);
  ocl_data.select_kernel("partial_integ");
  ocl_data.set_arg<cl::Buffer>(0, dev_A);
  ocl_data.set_arg<cl::Buffer>(1, dev_B);
  ocl_data.set_arg<unsigned int>(2, N);
  ocl_data.set_arg<cl::LocalSpaceArg>(3, cl::Local((DIMY + 1) * wf_size * sizeof(real)));
  // this is tricky line. it hard to find in documentation C++ wrapper or google. posible but hard.
  // this is how local/shared memory size is announced to kernel
  
  //ocl_data.set_arg<unsigned int>(4, DIMY); // we can pass this parameter as variable but it not good practice since it is very inner value which can be used for optimization by compiler. so i keep it #defined contant
  ocl_data.set_arg<cl::Buffer>(4, dev_tmp);

  ocl_data.select_kernel("integ_merge");
  ocl_data.set_arg<cl::Buffer>(0, dev_B);
  ocl_data.set_arg<cl::Buffer>(1, dev_tmp);
  ocl_data.set_arg<unsigned int>(2, N);
  //ocl_data.set_arg<unsigned int>(3, DIMY);

  std::cout << "running kernel " << ocl_data.current_kernel_name << '\n';
  for(int i=-7; i!=100; i++)
  {
    //===========STAGE=1=============
    // partial integration of bulks with size = DIMY * WF_SIZE
    ocl_data.select_kernel("partial_integ");
    if(!ocl_data.run(wf_count, wf_size, i > 0))
      return -1;
    
    //===========STAGE=2=============
    // interation of array of size n / (DIMY * WF_SIZE)
    // input array consists of small bulks endpoints
    ocl_data.timer_start();
    ocl_data.d2h(dev_tmp, tmp.data(), tmp.size());
    for(unsigned int j(1); j != tmp.size(); j++)
      tmp[j] += tmp[j - 1];
    ocl_data.h2d(tmp, dev_tmp);
    ocl_data.timer_stop("integ_cpu_part");

    //===========STAGE=3=============
    // merge both results toether
    ocl_data.select_kernel("integ_merge");
    if(!ocl_data.run(wf_count, wf_size, i > 0))
      return -1;
  }
  
  std::cout << "getting result data\n";
  ocl_data.d2h(dev_B, B.data(), B.size());

  ocl_data.print_timings();

  check_resutls(B, Bref);
  
  std::ofstream diff("diff.txt");
  for(int i=0; i<N; i+=wf_size*DIMY)
    diff << "B[" << i << "]=" << B[i] << ", Bref[" << i << "]=" << Bref[i] << '\n';
  diff.close();
  return 0;
}
