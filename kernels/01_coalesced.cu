// 01_coalesced.cu — Kernel 1: Global Memory Coalescing
//
// Fix: swap threadIdx.x/y assignment so threads in the SAME warp access
// consecutive columns of C, which means consecutive elements of B row.
// A-access (row of A) is now broadcast-friendly across a warp.
//
// Bottleneck targeted: uncoalesced global loads from B in the naive kernel.
// Prediction: Global load efficiency improves from ~3% to ~100% for B;
// A access becomes a broadcast (1 transaction per warp). Should show
// immediate 2–4× speedup over naive despite no other changes.
//
// Failure mode: on hardware with wide L2 cache lines, naive may partially
// hide uncoalescing via L2 hits at small sizes — but at 4096³ it will be
// fully DRAM-bound so coalescing matters a great deal.

#include "common.cuh"

__global__ void sgemm_coalesced(int M, int N, int K,
                                 float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float beta,
                                 float* __restrict__ C) {
  // KEY CHANGE vs naive: col = threadIdx.x (fast dimension), row = threadIdx.y
  // In a warp, threadIdx.x varies 0..31; consecutive threads → consecutive cols
  // → consecutive B and C memory addresses → fully coalesced 128-byte transactions
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= M || col >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    // A[row, k]: all threads in warp have same row → broadcast (1 transaction)
    // B[k, col]: consecutive cols → perfectly coalesced (1 transaction/warp)
    acc += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

void launch_sgemm_coalesced(int M, int N, int K,
                              float alpha, const float* dA, const float* dB,
                              float beta, float* dC) {
  dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32);
  sgemm_coalesced<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}
