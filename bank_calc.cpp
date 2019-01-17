#include <iostream>
#include <vector>

/*
 * this is a simple tool demonstrating that it is quite easy to check
 * your access patterns that you use on GPU devices
 */
int main()
{
  int bsz=64; // OpenCL's wavefront_size = CUDA's blockDim.x 
  int DIMY = 32; // some stride/ number of blocks that you plan
                 // to process sequentially 
  
  //local/shared memory example:
  for(int i=0;i!=DIMY;i++)
  {
    std::vector<int> banks(32,0); // 32 is just an example.
                                  // make sure you use proper values here
    for(int tid=0;tid!=bsz;tid++)
    {
      // NOTE: we iterate over GPU threads in the inner loop
      int adr = i + tid * (DIMY + 1);
      banks[adr % banks.size()]++;
    }
    // ok we did one cycle of local memory read/write
    // now it is time to check how many conflicts do we have
    // ideally numbers in banks vector should be equal
    // if your number of threas exceed real number of physical threads per CU/SM
    // it is better to test if banks is fille uniformly over time.
    // in other words try to break upper loop to bunches of sizes equal to
    // the number of physical threads in CU/SM.
    for(auto b : banks)
      std::cout << '[' << b << "] ";
    std::cout << '\n';
  }

  //this is the same local mem conflicts for another access pattern
  for(int i=0;i!=DIMY;i++)
  {
    std::vector<int> banks(32,0);
    for(int tid=0;tid!=bsz;tid++)
    {
      int adr = ((i*bsz + tid) / DIMY) * (DIMY + 1) + ((i*bsz+tid) % DIMY);
      banks[adr % 32]++;
    }
    for(auto b : banks)
      std::cout << '[' << b << "] ";
    std::cout << '\n';
  }


  //the below code is for estimation of channel/bank conflicts in global memory
  int j=0;
  //for(; j!=1024*1024*64; j+=64*256*16)
  {
    int i=0;
    for(; i!=16*64; i+=64)
    {
      std::vector<int> channels(12,0);
      for(int bi=0; bi!=256; bi++)
      {
        int wfId=bi*64*16;
        int addr = ((i+64*bi)%(16*64)+wfId+j)*4;
        channels[(addr / 256) % 12]++;
      }
      for(auto c : channels)
        std::cout << '[' << c << "] ";
      std::cout << '\n';
    }
  }

  return -1;
}
