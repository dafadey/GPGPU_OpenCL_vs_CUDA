#include "../service.h"
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>

bool check_resutls(const std::vector<real>& C, const std::vector<real>& Cref, const std::string& prefix="")
{
  bool passed(true);
  int bads(0);
  int goods(0);
  for(size_t i(0); i != C.size(); i++)
  {
    if(std::abs(C[i] - Cref[i]) != 0)
    {
			//std::cerr << '[' << i << "]=" << C[i] << " <-> ref[" << i << "]=" << Cref[i] << '\n';
      bads++;
      passed = false;
    }
    else
      goods++;
  }
  std::cout << prefix << (passed ? "test passed" : "test FAILED, bads:" + std::to_string(bads) + ", goods:" + std::to_string(goods)) << '\n';
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
  
  double get_timing(const std::string& kern_name)
  {
		auto it = timings.find(kern_name);
		if(it != timings.end())
			return it->second.first / (double) it->second.second;
		else
			return -1.0;
	}
};

extern "C"
void simple_copy_add(const real*, const real*, real*, unsigned int n);
extern "C"
void simple_copy_add_stride(const real*, const real*, real*, unsigned int n);
extern "C"
void simple_copy_add_NC(const real*, const real*, real*, unsigned int n);
extern "C"
void simple_copy_add_NC_fix(const real*, const real*, real*, unsigned int n);

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
  std::vector<real> C(N);
  std::vector<real> Cref(N);
  for(size_t i(0); i != N; i++)
  {
		A[i] = (real) 2.0 * (real) i;
    B[i] = - (real) i;
		Cref[i] = A[i]+B[i];
	}

  real* dev_A(nullptr);
  real* dev_B(nullptr);
  real* dev_C(nullptr);
  dev_alloc(0, (void**) &dev_A, N*sizeof(real));
  dev_alloc(0, (void**) &dev_B, N*sizeof(real));
  dev_alloc(0, (void**) &dev_C, N*sizeof(real));
  if(!(dev_A && dev_B && dev_C))
	{
		std::cerr << "failed allocating arrays\n";
		return -1;
	}
	timer t;

	bool pass=true;
	
	#define TEST(test_func, ntimes)              \
	dev_h2d(0, A.data(), dev_A, N*sizeof(real)); \
	dev_h2d(0, B.data(), dev_B, N*sizeof(real)); \
	zero(C); 																		 \
	dev_h2d(0, C.data(), dev_C, N*sizeof(real)); \
  std::cout << "running kernel "               \
            << #test_func << '\n';             \
  for(int i=-30; i!=ntimes; i++)               \
  {                                            \
		if(i>0)	t.timer_start();                   \
		test_func(dev_A, dev_B, dev_C, N);    		 \
		if(i>0)	t.timer_stop(#test_func);          \
  }                     											 \
  std::cout << "\tgetting result data\n";			 \
  dev_d2h(0, dev_C, C.data(), N*sizeof(real)); \
  pass &= check_resutls(C, Cref, "\t");				 \
  std::cout << '\t' << #test_func << " bw is " << sizeof(real)*(A.size()+B.size()+C.size())/t.get_timing(#test_func)*1e6/1024.0/1024.0/1024.0 << " Gb/sec\n";
  
  TEST(simple_copy_add, 300)
  TEST(simple_copy_add_stride, 300)
  TEST(simple_copy_add_NC, 10)
  TEST(simple_copy_add_NC_fix, 10)
  
  std::cout << ">>>=====timings=are======>>>\n";
  t.print_timings();
  if(pass)
  {
		std::cout << "ALL GOOD!\n";
		return 0;
	}
	else
	{
		std::cerr << "SOME TESTS FAILED!\n";
		return -1;
	}
}
