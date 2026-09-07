#include "common.cuh"

#define TILE_SIZE 32

__global__ void sgemm_smem(int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C) {
  __shared__ float tileA[TILE_SIZE][TILE_SIZE];
  __shared__ float tileB[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float acc = 0.0f;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    int aCol = t * TILE_SIZE + threadIdx.x;
    if (row < M && aCol < K) {
      tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
    } else {
      tileA[threadIdx.y][threadIdx.x] = 0.0f;
    }

    int bRow = t * TILE_SIZE + threadIdx.y;
    if (bRow < K && col < N) {
      tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
    } else {
      tileB[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
      acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

void launch_sgemm_smem(int M, int N, int K, float alpha, const float* dA, const float* dB, float beta, float* dC) {
  dim3 block(TILE_SIZE, TILE_SIZE);  
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  sgemm_smem<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}
