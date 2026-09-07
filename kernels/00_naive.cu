#include "common.cuh"

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C) {      
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M || col >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}


void launch_sgemm_naive(int M, int N, int K, float alpha, const float* dA, const float* dB, float beta, float* dC) {
  
  dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32);
  sgemm_naive<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}
