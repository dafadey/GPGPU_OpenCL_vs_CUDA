#pragma once
#include <CL/cl.hpp>
#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <iostream>

struct opencl
{
  opencl();

  bool buildOCLsources(const std::string&); // ' ' separated list of files

  bool add_kernel(const std::string&);

  cl::Device dev;
  cl::Context ctx;
  std::vector<std::string> srcs;
  cl::Program::Sources cl_srcs;
  cl::Program prg;
  cl::CommandQueue cqueue;
  std::map<std::string, cl::Kernel> kernels;
};

template<typename T>
struct opencl_data
{
  opencl_data(opencl& ocl) : ctx_ptr(&(ocl.ctx)), kernel_map_ptr(&(ocl.kernels)), cqueue_ptr(&(ocl.cqueue)), ev(), current_kernel_ptr(nullptr), current_kernel_name() {}
  
  cl::Context* ctx_ptr;
  std::map<std::string, cl::Kernel>* kernel_map_ptr;
  cl::CommandQueue* cqueue_ptr;
  cl::Event ev;
  cl::Kernel* current_kernel_ptr;
  std::string current_kernel_name;
  std::map<std::string, std::pair<double, unsigned long>> timings;
  std::chrono::time_point<std::chrono::high_resolution_clock> ts;
  std::chrono::time_point<std::chrono::high_resolution_clock> te;
  
  bool h2d_manual(const void* buff_h, size_t n_bytes, cl::Buffer& buff_d)
  {
    int err_code = cqueue_ptr->enqueueWriteBuffer(buff_d, CL_TRUE, 0, n_bytes, buff_h);
    if(err_code != CL_SUCCESS)
    {
      std::cerr << "error writing data to device, error code is " << err_code << '\n';
      return false;
    }
    return true;
  }
  
  
  bool h2d(const T* buff_h, size_t n, cl::Buffer& buff_d)
  {
    int err_code = cqueue_ptr->enqueueWriteBuffer(buff_d, CL_TRUE, 0, sizeof(T) * n, buff_h);
    if(err_code != CL_SUCCESS)
    {
      std::cerr << "error writing data to device, error code is " << err_code << '\n';
      return false;
    }
    return true;
  }

  bool h2d(const std::vector<T>& buff_h, cl::Buffer& buff_d)
  {
    return h2d(buff_h.data(), buff_h.size(), buff_d);
  }
  
  bool d2h(const cl::Buffer& buff_d, T* buff_h, size_t n)
  {
    int err_code = cqueue_ptr->enqueueReadBuffer(buff_d, CL_TRUE, 0, sizeof(T) * n, buff_h);
    if(err_code != CL_SUCCESS)
    {
      std::cerr << "error reading data from device, error code is " << err_code << '\n';
      return false;
    }
    return true;
  }

  bool select_kernel(const std::string& kernel_name)
  {
    auto it = kernel_map_ptr->find(kernel_name);
    if(it == kernel_map_ptr->end())
      return false;
    current_kernel_ptr = &(it->second);
    current_kernel_name = kernel_name;
    return true;
  }
  
  template<typename TT>
  bool set_arg(int count, const TT& arg)
  {
    if(!current_kernel_ptr)
      return false;
    int err_code = current_kernel_ptr->setArg(count, arg); 
    if(err_code != CL_SUCCESS)
    {
      std::cerr << "error setting argument " << count << ", error code is " << err_code << '\n';
      return false;
    }
    return true;
  }
  
  
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

  
  bool run(int work_group_count, int work_group_size, bool profile = false)
  {
    if(!current_kernel_ptr)
      return false;
    if(profile)
      timer_start();
    int err_code = cqueue_ptr->enqueueNDRangeKernel(*current_kernel_ptr, cl::NullRange, cl::NDRange(work_group_count * work_group_size), cl::NDRange(work_group_size), NULL, &ev); 
    ev.wait(); // DO NOT FORGET THIS SMALL BUT IMPORTANT THING
    if(err_code != CL_SUCCESS)
    {
      std::cerr << "Failed running kernel, err code is " << err_code << '\n';
      return false;
    }
    if(!profile)
      return true;
    timer_stop(current_kernel_name);
    return true;
  }

  enum memType {RW, R};
  
  cl::Buffer alloc(size_t num_bytes, memType mt)
  {
    return cl::Buffer(*ctx_ptr, mt == memType::RW ? CL_MEM_READ_WRITE : CL_MEM_READ_ONLY, num_bytes);
  }
  
  void print_timings(std::ostream& os = std::cout)
  {
    for(auto& it : timings)
      os << it.first << " : " << it.second.first / (double) it.second.second << "us\n";
  }

};
