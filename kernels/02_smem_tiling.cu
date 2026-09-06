// 02_smem_tiling.cu — Kernel 2: Shared Memory Tiling
//
// Load TILE_SIZE×TILE_SIZE tiles of A and B into shared memory, then compute
// from fast on-chip SRAM instead of repeated global memory reads.
//
// Arithmetic intensity analysis:
//   Each element of A and B is loaded once into shared memory (global),
//   then reused TILE_SIZE times in the inner loop.
//   Arithmetic intensity ≈ TILE_SIZE/2 FLOPS/byte (FP32 = 4 bytes).
//   With TILE=32: ~8 FLOPS/byte vs. ~0.25 for naive kernel.
//
// Bottleneck targeted: repeated global memory reads of the same A/B elements
// across threads in the block.
//
// Potential failure: shared memory bank conflicts if threads access the same
// bank column — padding can fix this (see tile_A layout below).
// With 32-wide tiles and row-major layout: A tile is conflict-free (each row
// is a different bank group); B tile can have conflicts if strided.

#include "common.cuh"

#define TILE_SIZE 32

__global__ void sgemm_smem(int M, int N, int K,
                            float alpha,
                            const float* __restrict__ A,
                            const float* __restrict__ B,
                            float beta,
                            float* __restrict__ C) {
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float acc = 0.0f;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    // Load tile of A: thread (ty, tx) loads A[row, t*TILE+tx]
    // Coalescing: consecutive tx → consecutive K-column addresses → coalesced
    int aCol = t * TILE_SIZE + threadIdx.x;
    tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K)
                                       ? A[row * K + aCol]
                                       : 0.0f;

    // Load tile of B: thread (ty, tx) loads B[t*TILE+ty, col]
    // Coalescing: consecutive tx → consecutive N-column addresses → coalesced
    int bRow = t * TILE_SIZE + threadIdx.y;
    tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N)
                                       ? B[bRow * N + col]
                                       : 0.0f;

    __syncthreads();

    // Compute partial dot product from tiles (no global memory access)
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
      // tileA[threadIdx.y][k]: all threads in warp read same row → broadcast OK
      // tileB[k][threadIdx.x]: consecutive tx → consecutive bank → conflict-free
      acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

void launch_sgemm_smem(int M, int N, int K,
                        float alpha, const float* dA, const float* dB,
                        float beta, float* dC) {
  dim3 block(TILE_SIZE, TILE_SIZE);  // 32×32 = 1024 threads
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  sgemm_smem<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}
