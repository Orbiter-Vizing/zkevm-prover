#ifdef __USE_CUDA__
  #include "api/bn254.h"
#endif

#include "pinned_memory_manager.cuh"
// #include "curves/params/bn254.cuh"

// template <typename T> T *alloc_pinned_memory(u_int64_t totalLength) {
//   T *pinned_memory;
//   CHK_STICKY(cudaMallocHost((void **)&pinned_memory, totalLength *
//   sizeof(T))); return pinned_memory;
// }

// template <typename T> void free_pinned_memory(T *pinned_memory) {
//   CHK_STICKY(cudaFreeHost(pinned_memory));
// }

AltBn128::Engine::FrElement* alloc_pinned_memory(u_int64_t totalLength)
{
  AltBn128::Engine::FrElement* pinned_memory = nullptr;
#ifdef __USE_CUDA__
  CHK_STICKY(cudaMallocHost((void**)&pinned_memory, totalLength * sizeof(AltBn128::Engine::FrElement)));
#else
  pinned_memory = new AltBn128::Engine::FrElement[totalLength];
#endif
  return pinned_memory;
}

void free_pinned_memory(AltBn128::Engine::FrElement* pinned_memory)
{
#ifdef __USE_CUDA__
  CHK_STICKY(cudaFreeHost(pinned_memory));
#else
  if (pinned_memory != nullptr) {
    delete[] pinned_memory;
    pinned_memory = nullptr;
  }

#endif
}