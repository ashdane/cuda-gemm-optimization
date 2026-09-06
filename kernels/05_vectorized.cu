// 05_vectorized.cu — Kernel 5: Vectorized Memory Access
//
// Use float4 (128-bit) loads/stores for global→shared memory transfers.
// A single float4 load moves 4 floats in ONE instruction instead of 4,
// reducing the instruction count and pressure on the load/store units.
//
// Why this matters: on Pascal/Turing, L2 cache line = 128 bytes (32 floats).
// A float4 load at an aligned address is served by a single cache transaction.
// Scalar float loads at 4 different addresses could serialize.
//
// Constraint: BK must be divisible by 4 (for float4 load of As rows).
//             BN must be divisible by 4 (for float4 load of Bs rows).
//             Alignment: base pointer + offset must be 16-byte aligned.
//
// Adjustment from k4: BK=16 (was 8) to allow float4 loads from both A and B.
//   As: BM×BK=128×16, Bs: BK×BN=16×128
//   smem = (128*16 + 16*128)*4 = 16384 bytes = 16 KB ✓
//
// Turing advantage: RTX 2080 Ti has a 32-bank 32-bit shared memory
// with 32 B per bank per cycle. float4 loads from global can be more
// efficiently pipelined in Turing's LD unit (L1/shared memory bandwidth
// is higher: 121 TB/s vs 96 TB/s on Pascal per spec).

#include "common.cuh"

#define BM 128
#define BN 128
#define BK 16   // Increased from 8 to accommodate float4 loads
#define TM 8
#define TN 8

__global__ void sgemm_vectorized(int M, int N, int K,
                                  float alpha,
                                  const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float beta,
                                  float* __restrict__ C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const int cRow = blockIdx.y * BM;
  const int cCol = blockIdx.x * BN;

  const int threadRow = threadIdx.x / (BN / TN);
  const int threadCol = threadIdx.x % (BN / TN);

  // Vectorized loading: each thread loads one float4 per tile load step
  // As: BM*BK/4 float4s total; blockDim=256 threads → each loads BM*BK/(4*256) float4s
  // BM*BK = 128*16 = 2048; 2048/4 = 512 float4s; 512/256 = 2 per thread
  const int innerRowA = threadIdx.x / (BK / 4);
  const int innerColA = threadIdx.x % (BK / 4);  // float4 column index
  const int innerRowB = threadIdx.x / (BN / 4);
  const int innerColB = threadIdx.x % (BN / 4);  // float4 column index

  float threadResults[TM * TN] = {0.0f};
  float regA[TM];
  float regB[TN];

  for (int bkIdx = 0; bkIdx < (K + BK - 1) / BK; ++bkIdx) {
    // Load As via float4 (2 float4s per thread)
    for (int loadOffset = 0; loadOffset < BM; loadOffset += blockDim.x / (BK / 4)) {
      const int globalRow = cRow + innerRowA + loadOffset;
      const int globalCol = bkIdx * BK + innerColA * 4;
      if (globalRow < M && globalCol + 3 < K) {
        // Fully in-bounds: use float4 load
        const float4 tmp = *reinterpret_cast<const float4*>(
            &A[globalRow * K + globalCol]);
        As[(innerRowA + loadOffset) * BK + innerColA * 4 + 0] = tmp.x;
        As[(innerRowA + loadOffset) * BK + innerColA * 4 + 1] = tmp.y;
        As[(innerRowA + loadOffset) * BK + innerColA * 4 + 2] = tmp.z;
        As[(innerRowA + loadOffset) * BK + innerColA * 4 + 3] = tmp.w;
      } else if (globalRow < M) {
        // Partial row (K boundary)
        for (int i = 0; i < 4; ++i) {
          As[(innerRowA + loadOffset) * BK + innerColA * 4 + i] =
              (globalCol + i < K) ? A[globalRow * K + globalCol + i] : 0.0f;
        }
      } else {
        for (int i = 0; i < 4; ++i)
          As[(innerRowA + loadOffset) * BK + innerColA * 4 + i] = 0.0f;
      }
    }

    // Load Bs via float4
    for (int loadOffset = 0; loadOffset < BK; loadOffset += blockDim.x / (BN / 4)) {
      const int globalRow = bkIdx * BK + innerRowB + loadOffset;
      const int globalCol = cCol + innerColB * 4;
      if (globalRow < K && globalCol + 3 < N) {
        const float4 tmp = *reinterpret_cast<const float4*>(
            &B[globalRow * N + globalCol]);
        Bs[(innerRowB + loadOffset) * BN + innerColB * 4 + 0] = tmp.x;
        Bs[(innerRowB + loadOffset) * BN + innerColB * 4 + 1] = tmp.y;
        Bs[(innerRowB + loadOffset) * BN + innerColB * 4 + 2] = tmp.z;
        Bs[(innerRowB + loadOffset) * BN + innerColB * 4 + 3] = tmp.w;
      } else if (globalRow < K) {
        for (int i = 0; i < 4; ++i)
          Bs[(innerRowB + loadOffset) * BN + innerColB * 4 + i] =
              (globalCol + i < N) ? B[globalRow * N + globalCol + i] : 0.0f;
      } else {
        for (int i = 0; i < 4; ++i)
          Bs[(innerRowB + loadOffset) * BN + innerColB * 4 + i] = 0.0f;
      }
    }

    __syncthreads();

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (int i = 0; i < TM; ++i)
        regA[i] = As[(threadRow * TM + i) * BK + dotIdx];
      for (int j = 0; j < TN; ++j)
        regB[j] = Bs[dotIdx * BN + threadCol * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
          threadResults[i * TN + j] += regA[i] * regB[j];
    }

    __syncthreads();
  }

  // Vectorized writeback using float4
  for (int i = 0; i < TM; ++i) {
    const int globalRow = cRow + threadRow * TM + i;
    if (globalRow >= M) continue;
    for (int j = 0; j < TN; j += 4) {
      const int globalCol = cCol + threadCol * TN + j;
      if (globalCol + 3 < N) {
        float4 result;
        result.x = alpha * threadResults[i * TN + j + 0] + beta * C[globalRow * N + globalCol + 0];
        result.y = alpha * threadResults[i * TN + j + 1] + beta * C[globalRow * N + globalCol + 1];
        result.z = alpha * threadResults[i * TN + j + 2] + beta * C[globalRow * N + globalCol + 2];
        result.w = alpha * threadResults[i * TN + j + 3] + beta * C[globalRow * N + globalCol + 3];
        *reinterpret_cast<float4*>(&C[globalRow * N + globalCol]) = result;
      } else {
        for (int jj = 0; jj < 4 && globalCol + jj < N; ++jj)
          C[globalRow * N + globalCol + jj] =
              alpha * threadResults[i * TN + j + jj] + beta * C[globalRow * N + globalCol + jj];
      }
    }
  }
}

void launch_sgemm_vectorized(int M, int N, int K,
                               float alpha, const float* dA, const float* dB,
                               float beta, float* dC) {
  dim3 block((BM / TM) * (BN / TN));  // 256 threads
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_vectorized<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}
