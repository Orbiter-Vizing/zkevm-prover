#ifndef __PINNED_MEMORY_MANAGER_CUH__
#define __PINNED_MEMORY_MANAGER_CUH__
#include "alt_bn128.hpp"

AltBn128::Engine::FrElement *alloc_pinned_memory(u_int64_t totalLength);
void free_pinned_memory(AltBn128::Engine::FrElement *pinned_memory);

#endif // __PINNED_MEMORY_MANAGER_CUH__