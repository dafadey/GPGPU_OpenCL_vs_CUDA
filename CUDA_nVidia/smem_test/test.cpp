#include "../service.h"
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>

bool check_resutls(const std::vector<real>& C, const std::vector<real>& Cref)
{
  bool passed(true);
  int bads(0);
  int goods(0);
  for(size_t i(0); i != C.size(); i++)
  {
    if(std::abs(C[i] - Cref[i]) > (real) 0.001)
    {
			//std::cerr << '[' << i << "]=" << C[i] << " <-> ref[" << i << "]=" << Cref[i] << '\n';
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

struct timer
{
	timer() : timings(), ts(), te() {}
  std::map<std::string, std::pair<double, unsigned long>> timings;
  std::chrono::time_point<std::chrono::high_resolution_clock> ts;
  std::chrono::time_point<std::chrono::high_resolution_clock> te;
  
  void timer_start()
  {
    ts = std::chrono::high_resolution_clock::now();
  }

  void timer_stop(const std::string& current_kernel_name)
  {
    te = std::chrono::high_resolution_clock::now();
    auto it = timings.find(current_kernel_name);
    if(it == timings.end())
      timings[current_kernel_name] = std::make_pair((double) std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count(), 1);
    else
      it->second = std::make_pair(it->second.first + (double) std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count(), it->second.second + 1);
  }
  
  void print_timings(std::ostream& os = std::cout)
  {
    for(auto& it : timings)
      os << it.first << " : " << it.second.first / (double) it.second.second << "us\n";
  }
};

extern "C"
void integrate_partial(const real* A, real* B, real* tmp, unsigned int n);

int main()
{
	if(cudaSetDevice(0))
	{
		std::cerr << "device could not be initialized\n";
		return -1;
  }
  unsigned int wf_size=WF_SIZE;
  unsigned int wf_count=WF_COUNT;
	std::cout << "wavefront count is " << wf_count << ", wavefromt size is " << wf_size << '\n';
  size_t N = 1024 * 1024 * 64;
  std::vector<real> A(N);
  std::vector<real> B(N);
  std::vector<real> Bref(N);
  for(size_t i(0); i != N; i++)
    A[i] = sin((real) i);
    //A[i] = ((real) 0.01 * (real) i);

  for(size_t i(0); i != N; i++)
		Bref[i] = A[i];
  for(size_t i(0); i < N; i += DIMY * WF_SIZE)
  {
		for(size_t j(1); j != DIMY * WF_SIZE; j++)
			Bref[i + j] += Bref[i + j - 1];
	}
  real* dev_A(nullptr);
  real* dev_B(nullptr);
  real* dev_tmp(nullptr);
  dev_alloc(0, (void**) &dev_A, N*sizeof(real));
  if(!dev_A)
	{
		std::cerr << "failed allocating A\n";
		return -1;
	}
	dev_h2d(0, A.data(), dev_A, N*sizeof(real));
  dev_alloc(0, (void**) &dev_B, N*sizeof(real));
  if(!dev_B)
	{
		std::cerr << "failed allocating B\n";
		return -1;
	}
  dev_alloc(0, (void**) &dev_tmp, (N / wf_count)*sizeof(real));
  if(!dev_tmp)
	{
		std::cerr << "failed allocating dev_tmp\n";
		return -1;
	}
	std::vector<real> tmp(N / wf_count);

	timer t;
  std::cout << "running kernel \n";
  for(int i=-30; i!=300; i++)
  {
		if(i>0)
			t.timer_start();
		integrate_partial(dev_A, dev_B, dev_tmp, N);
		if(i>0)
			t.timer_stop("integrate_partial");
  }
  std::cout << "getting result data\n";
  dev_d2h(0, dev_B, B.data(), N*sizeof(real));
  check_resutls(B, Bref);
  
  t.print_timings();
  
  //for(int i=0; i!=64; i++)
	//	std::cout << "B[" << i << "]=" << B[i] << " <-> Bref[" << i << "]=" << Bref[i] << '\n';
  return 0;
}
