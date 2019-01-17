#include <fstream>
#include <sstream>
#include "helpers.h"

opencl::opencl()
{
  cl::Platform _platform;
  std::vector<cl::Platform> platforms;
    if(cl::Platform::get(&platforms)!= CL_SUCCESS)
    {
      std::cerr << "ERROR: failed to get any openCL platforms\n";
      return;
    }
    _platform = platforms[0];
  std::cout << "found " << platforms.size() << " platforms\n";
  for(const auto& p : platforms)
  {
    std::string name = p.getInfo<CL_PLATFORM_NAME>();
    std::string ext = p.getInfo<CL_PLATFORM_EXTENSIONS>();
    std::string profile = p.getInfo<CL_PLATFORM_PROFILE>();
    std::string ven = p.getInfo<CL_PLATFORM_VENDOR>();
    std::string ver = p.getInfo<CL_PLATFORM_VERSION>();
    
    std::cout << "\tplatform name is " << name << "\n"
              << "\tplatform extensions are " << ext << "\n"
              << "\tplatform profile is " << profile << "\n"
              << "\tplatform vendor is " << ven << "\n"
              << "\tplatform version is " << ver << "\n";
      
    std::vector<cl::Device> devices;
    if(p.getDevices(CL_DEVICE_TYPE_ALL, &devices) != CL_SUCCESS)
    {
      std::cerr << "ERROR: failed to get any GPU (CL_DEVICE_TYPE_GPU) devices for this platform\n";
      continue;
    }
    std::cout << "\t\tfound " << devices.size() << " devices\n";
    dev = devices[0];
    for(const auto& d : devices)
    {
      std::string kernels = d.getInfo<CL_DEVICE_BUILT_IN_KERNELS>();
      std::cout << "\t\t\tCL_DEVICE_BUILT_IN_KERNELS=" << kernels << "\n";
      std::string exts = d.getInfo<CL_DEVICE_EXTENSIONS>();
      std::cout << "\t\t\tCL_DEVICE_EXTENSIONS=" << exts << "\n";
      std::string name = d.getInfo<CL_DEVICE_NAME>();
      std::cout << "\t\t\tCL_DEVICE_NAME=" << name << "\n";

      cl_ulong memsz = d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
      std::cout << "\t\t\tCL_DEVICE_GLOBAL_MEM_SIZE=" << memsz << "\n";
      #define SHOW(PARAM)	std::cout << "\t\t\t" << #PARAM << "=" << d.getInfo<PARAM>() << "\n";
      SHOW(CL_DEVICE_GLOBAL_MEM_SIZE);
      SHOW(CL_DEVICE_LOCAL_MEM_SIZE);
      SHOW(CL_DEVICE_MAX_CLOCK_FREQUENCY);
      SHOW(CL_DEVICE_MAX_COMPUTE_UNITS);
      #undef SHOW
    }
  }
  ctx = cl::Context(dev);
  cqueue = cl::CommandQueue(ctx, dev);
}

bool opencl::buildOCLsources(const std::string& src_files)
{
  std::string file_name;
  std::istringstream is(src_files);
  while(std::getline(is, file_name, ' '))
  {
    std::cout << "reading cl source " << file_name << '\n';
    std::ifstream src_file(file_name.c_str());
    if(!src_file.is_open())
    {
      std::cerr << "WARNING: no " << file_name << " CL source file found\n";
      continue;    
    }
    std::string src;
    std::string line;
    while(std::getline(src_file, line))
      src += line + '\n';
    src_file.close();
    if(!src.size())
       continue;
    srcs.push_back(src);
  }
  for(auto& s : srcs)
    cl_srcs.push_back({s.c_str(), s.length()});

  prg = cl::Program(ctx, cl_srcs);

  std::cout << "building kernel...\n";
  int err_code = prg.build({dev},"-w"); 
  if(err_code != CL_SUCCESS)
  {
      std::cerr << "ERROR #" << err_code << ": while building CL source: " << prg.getBuildInfo<CL_PROGRAM_BUILD_LOG>({dev}) << '\n';
      return false;
  }
  std::cout << "\tCL program is built\n";
  return true;
}

bool opencl::add_kernel(const std::string& kernel_name)
{
  cl_int err;
  cl::Kernel _kernel = cl::Kernel(prg, kernel_name.c_str(), &err);
  if(err != CL_SUCCESS)
  {
    std::cerr << "Kernel constructor failed " << err << "\n";
    return false;
  }
  kernels[kernel_name] = _kernel;
  std::cout << "Kernel object is created\n";
  return true;
}
