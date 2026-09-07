#include "common.cuh"
#include <cublas_v2.h>

__global__ void fp32_to_fp16(const float* __restrict__ src, __half* __restrict__ dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}

#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARPS_PER_BLOCK 4
#define TC_BLOCK_ROWS   (WARPS_PER_BLOCK * WMMA_M)  
#define TC_BLOCK_COLS   WMMA_N                        

__global__ void sgemm_tc_wmma(int M, int N, int K, float alpha, const __half* __restrict__ A, const __half* __restrict__ B, float beta, float* __restrict__ C) {
  const int warpId = threadIdx.x / warpSize;
  const int laneId = threadIdx.x % warpSize;

  const int warpRow = blockIdx.y * TC_BLOCK_ROWS + warpId * WMMA_M;
  const int warpCol = blockIdx.x * TC_BLOCK_COLS;

  if (warpRow >= M || warpCol >= N) return;

  wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> aFrag;
  wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> bFrag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

  if (beta != 0.0f) {
    wmma::load_matrix_sync(cFrag, C + warpRow * N + warpCol, N, wmma::mem_row_major);
    for (int i = 0; i < cFrag.num_elements; ++i)
      cFrag.x[i] *= beta;
  } else {
    wmma::fill_fragment(cFrag, 0.0f);
  }

  for (int k = 0; k + WMMA_K <= K; k += WMMA_K) {
    wmma::load_matrix_sync(aFrag, A + warpRow * K + k,       K);
    wmma::load_matrix_sync(bFrag, B + k * N       + warpCol, N);
    wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
  }

  for (int i = 0; i < cFrag.num_elements; ++i)
    cFrag.x[i] *= alpha;

  wmma::store_matrix_sync(C + warpRow * N + warpCol, cFrag, N, wmma::mem_row_major);
}

void launch_sgemm_tensor_core(int M, int N, int K, float alpha, const float* dA, const float* dB, float beta, float* dC) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  if (prop.major < 7 || (prop.major == 7 && prop.minor < 5)) {
    fprintf(stderr,
            "sgemm_tc_wmma: requires sm_75+ (RTX 2080 Ti or newer).\n"
            "This GPU is %s (sm_%d%d) — no Tensor Cores.\n",
            prop.name, prop.major, prop.minor);
    return;  
  }

  int Mpad = (M + 15) / 16 * 16;
  int Kpad = (K + 15) / 16 * 16;
  int Npad = (N + 15) / 16 * 16;

  __half *dAh, *dBh;
  CUDA_CHECK(cudaMalloc(&dAh, (size_t)Mpad * Kpad * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dBh, (size_t)Kpad * Npad * sizeof(__half)));
  CUDA_CHECK(cudaMemset(dAh, 0, (size_t)Mpad * Kpad * sizeof(__half)));
  CUDA_CHECK(cudaMemset(dBh, 0, (size_t)Kpad * Npad * sizeof(__half)));

  {
    int nA = M * K, nB = K * N;
    int blockSz = 256;
    fp32_to_fp16<<<(nA + blockSz - 1) / blockSz, blockSz>>>(dA, dAh, nA);
    fp32_to_fp16<<<(nB + blockSz - 1) / blockSz, blockSz>>>(dB, dBh, nB);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  dim3 block(WARPS_PER_BLOCK * 32);  
  dim3 grid((Npad + TC_BLOCK_COLS - 1) / TC_BLOCK_COLS,
            (Mpad + TC_BLOCK_ROWS  - 1) / TC_BLOCK_ROWS);
  sgemm_tc_wmma<<<grid, block>>>(M, N, K, alpha, dAh, dBh, beta, dC);

  cudaFree(dAh);
  cudaFree(dBh);
}

void launch_cublas_fp16_tc(cublasHandle_t handle, int M, int N, int K, float alpha, const float* dA, const float* dB, float beta, float* dC) {
  __half *dAh, *dBh;
  CUDA_CHECK(cudaMalloc(&dAh, (size_t)M * K * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dBh, (size_t)K * N * sizeof(__half)));

  int nA = M * K, nB = K * N;
  fp32_to_fp16<<<(nA + 255) / 256, 256>>>(dA, dAh, nA);
  fp32_to_fp16<<<(nB + 255) / 256, 256>>>(dB, dBh, nB);
  CUDA_CHECK(cudaDeviceSynchronize());

  const __half alpha_h = __float2half(alpha);
  const __half beta_h  = __float2half(beta);

  cublasStatus_t st = cublasGemmEx(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha,   
      dBh, CUDA_R_16F, N,
      dAh, CUDA_R_16F, K,
      &beta,    
      dC,  CUDA_R_32F, N,
      CUDA_R_32F,  
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  if (st != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cublasGemmEx FP16 TC failed: %d\n", (int)st);
  }

  cudaFree(dAh);
  cudaFree(dBh);
}
