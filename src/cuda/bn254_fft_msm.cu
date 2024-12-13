#ifdef __USE_CUDA__

#include "api/bn254.h"
#include "bn254_fft_msm.hpp"
#include "curves/params/bn254.cuh"
#include "curves/projective.cuh"
#include "thread_channel_t.hpp"
#include <cstring>
#include <iostream>
#include <mutex>

static const unsigned int msm_batch_num = 5;
static const unsigned log_ntt_size = 26;
static const size_t flip_flop = 1;
static std::vector<device_context::DeviceContext> m_device_contexts;
static std::vector<cudaStream_t> m_device_streams;
static std::vector<msm::MSMConfig> m_msm_configs;
static std::vector<ntt::NTTConfig<bn254::scalar_t>> m_ntt_configs;

static bn254::scalar_t NTT_BASIC_ROOT;

struct resource_t {
  // int dev;
  int stream;
  resource_t(/*int _dev ,*/ int _stream) {
    // dev = _dev;
    stream = _stream;
  }
};
channel_t<resource_t *> resources;

static std::once_flag init_flag;

static void init_ntt_msm_config() {
  std::call_once(init_flag, []() {
    NTT_BASIC_ROOT = bn254::scalar_t::omega(log_ntt_size);
    m_device_streams.resize(flip_flop, nullptr);
    for (int dev_id = 0; dev_id < flip_flop; ++dev_id) {
      CHK_STICKY(cudaStreamCreate(&m_device_streams[dev_id]));
      device_context::DeviceContext context = {m_device_streams[dev_id],
                                               (size_t)0, 0x0};
      bn254_initialize_domain(&NTT_BASIC_ROOT, context, true);
      m_device_contexts.push_back(context);
      ntt::NTTConfig<bn254::scalar_t> ntt_config = {
          context,                 // ctx
          bn254::scalar_t::one(),  // coset_gen
          1,                       // batch_size
          false,                   // columns_batch
          ntt::Ordering::kNN,      // ordering
          false,                   // are_inputs_on_device
          false,                   // are_outputs_on_device
          false,                   // is_async
          ntt::NttAlgorithm::Auto, // ntt_algorithm
      };
      m_ntt_configs.push_back(ntt_config);
      msm::MSMConfig msm_config = {
          context, // ctx
          0,       // points_size
          1,       // precompute_factor
          0,       // c
          0,       // bitsize
          10,      // large_bucket_factor
          1,       // batch_size
          false,   // are_scalars_on_device
          true,    // are_scalars_montgomery_form
          false,   // are_points_on_device
          true,    // are_points_montgomery_form
          false,   // are_results_on_device
          false,   // is_big_triangle
          false,   // is_async
      };
      m_msm_configs.push_back(msm_config);
    }

    for (size_t j = 0; j < flip_flop; ++j) {
      resources.send(new resource_t(j));
    }
  });
}

void icicle_bn254_intt_cuda(AltBn128::Engine::FrElement *a, u_int64_t n) {
  // nvtx3::scoped_range nvtx_r{"icicle_bn254_intt_cuda"};
  TimerStart(ICICLE_BN254_INTT_CUDA);

  // std::cout << "------------------------intt size: " << n <<
  // "---------------------------------\n";
  // TODO: fix n * 4, fflonk maybe n*4 and groth16 is n
  init_ntt_msm_config();
  auto icicle_a = reinterpret_cast<bn254::scalar_t *>(a);
  resource_t *resource = resources.recv();
  bn254_ntt_cuda(icicle_a, n, ntt::NTTDir::kInverse,
                 m_ntt_configs[resource->stream], icicle_a);
  resources.send(resource);
  TimerStopAndLog(ICICLE_BN254_INTT_CUDA);
}

void icicle_bn254_ntt_cuda(AltBn128::Engine::FrElement *a, u_int64_t n) {
  // nvtx3::scoped_range nvtx_r{"icicle_bn254_ntt_cuda"};
  TimerStart(ICICLE_BN254_NTT_CUDA);
  // std::cout << "------------------------ntt size: " << n <<
  // "---------------------------------\n";
  // TODO: fix n * 4, fflonk maybe n*4 and groth16 is n
  init_ntt_msm_config();
  auto icicle_a = reinterpret_cast<bn254::scalar_t *>(a);
  resource_t *resource = resources.recv();
  bn254_ntt_cuda(icicle_a, n, ntt::NTTDir::kForward,
                 m_ntt_configs[resource->stream], icicle_a);
  resources.send(resource);
  TimerStopAndLog(ICICLE_BN254_NTT_CUDA);
}

// TODO: fix multi GPU
void icicle_bn254_msm_cuda(AltBn128::Engine::G1Point &r,
                           AltBn128::Engine::G1PointAffine *bases,
                           uint8_t *scalars, unsigned int n) {
  TimerStart(ICICLE_BN254_MSM_CUDA);
  // std::cout << "-------------msm size: " << n <<
  // "----------------------------\n"; nvtx3::scoped_range
  // nvtx_r{"icicle_bn254_msm_cuda"};

  auto icicle_bases = reinterpret_cast<bn254::affine_t *>(bases);
  auto icicle_scalars = reinterpret_cast<bn254::scalar_t *>(scalars);
  static bn254::projective_t msm_results[msm_batch_num];

  unsigned int ave_len = (n + msm_batch_num - 1) / msm_batch_num;
  for (int i = 0; i < msm_batch_num; ++i) {
    // nvtx3::scoped_range for_loop{"partial msm"};
    unsigned int batch_len = (i == msm_batch_num - 1)
                                 ? (n - (msm_batch_num - 1) * ave_len)
                                 : ave_len;
    // resource_t *resource = resources.recv();
    CHK_STICKY(bn254_msm_cuda(&icicle_scalars[ave_len * i],
                              &icicle_bases[ave_len * i], batch_len,
                              m_msm_configs[0], &msm_results[i]));

    // resources.send(resource);
    // res = (i == 0) ? msm_results[0] : res + msm_results[i];
  }
  bn254::projective_t res = msm_results[0];
  for (int i = 1; i < msm_batch_num; ++i) {
    res = res + msm_results[i];
  }
  // assert(bn254::projective_t::is_on_curve(res));
  bn254::affine_t affine_res = bn254::projective_t::to_affine(res);
  bn254::affine_t affine_res_mont = bn254::affine_t::to_montgomery(affine_res);
  AltBn128::Engine::G1PointAffine affine_res2;
  std::memcpy((void *)&affine_res2, (void *)&affine_res_mont,
              sizeof(affine_res_mont));
  AltBn128::Engine::engine.g1.copy(r, affine_res2);

  TimerStopAndLog(ICICLE_BN254_MSM_CUDA);
}

#endif