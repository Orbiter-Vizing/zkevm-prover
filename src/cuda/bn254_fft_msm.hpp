// #include <nvtx3/nvtx3.hpp>
#ifndef __BN254_FFT_MSM_HPP__
#define __BN254_FFT_MSM_HPP__
#ifdef __USE_CUDA__
  #include "alt_bn128.hpp"
  #include "timer.hpp"

void icicle_bn254_ntt_cuda(AltBn128::Engine::FrElement* a, u_int64_t n);
void icicle_bn254_intt_cuda(AltBn128::Engine::FrElement* a, u_int64_t n);
void icicle_bn254_msm_cuda(
  AltBn128::Engine::G1Point& r, AltBn128::Engine::G1PointAffine* bases, uint8_t* scalars, unsigned int n);
#endif

#endif //__BN254_FFT_MSM_HPP__