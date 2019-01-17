#pragma once
#include "defaults.h"

#ifndef nullptr
	#define nullptr 0
#endif

extern "C"
void dev_alloc(int device, void** pparr, int sz);

extern "C"
void dev_h2d(int device, real* host_arr, real* dev_arr, int sz);

extern "C"
void dev_d2h(int device, const real* dev_arr, real* host_arr, int sz);
