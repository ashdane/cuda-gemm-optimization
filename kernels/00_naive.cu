// 00_naive.cu — Kernel 0: Naive SGEMM
//
// Each thread computes one output element C[row,col] = sum_k A[row,k]*B[k,col].
// Access pattern: A is row-strided (stride=K), B is column-strided (stride=N).
// Both accesses are uncoalesced for the K dimension traversal.
// Expected bottleneck: extreme global memory bandwidth waste; L2/DRAM bound.
// Predicted profiler observation: very low global load efficiency (~12.5% for
// 32-wide warps doing strided B access), near-zero occupancy on large tiles.

#include "common.cuh"

__global__ void sgemm_naive(int M, int N, int K,
                             float alpha,
                             const float* __restrict__ A,  // M×K row-major
                             const float* __restrict__ B,  // K×N row-major
                             float beta,
                             float* __restrict__ C) {      // M×N row-major
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M || col >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// Launcher used by bench_runner
void launch_sgemm_naive(int M, int N, int K,
                         float alpha, const float* dA, const float* dB,
                         float beta, float* dC) {
  // 32×32 thread block = 1024 threads per block
  dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32);
  sgemm_naive<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}
