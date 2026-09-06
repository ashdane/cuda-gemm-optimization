// 04_2d_blocktile.cu — Kernel 4: 2D Block Tiling
//
// Each thread now computes a TM×TN tile of output (not just a strip).
// This is the most important optimization step: it dramatically increases
// the FMA-to-load ratio from the shared memory tiles.
//
// Register reuse analysis:
//   Each thread loads TM elements of As and TN elements of Bs (per BK step),
//   then computes TM*TN FMAs from those loads — O(TM*TN) work from O(TM+TN) loads.
//   With TM=TN=8: 64 FMAs from 16 register loads. Arithmetic intensity ≈ 4 FLOPS/byte.
//
// Shared memory layout:
//   As: BM×BK, Bs: BK×BN (same as before, just more work per thread).
//   Total smem = (BM*BK + BK*BN)*4 bytes.
//   With BM=BN=128, BK=8: (128*8 + 8*128)*4 = 8192 bytes ✓ (well within 48KB)
//
// Bottleneck targeted: insufficient FMA-to-load ratio in 1D tiling.
// Expected: close to FP32 compute-bound regime (roofline near TFLOPS peak).
// PTX register count: TM*TN (accumulators) + TM (regA) + TN (regB) + addressing
//                    = 64 + 8 + 8 + ~10 ≈ ~90 registers → may limit occupancy.

#include "common.cuh"

#define BM 128   // Block rows
#define BN 128   // Block cols
#define BK 8     // Block depth
#define TM 8     // Thread tile rows
#define TN 8     // Thread tile cols

__global__ void sgemm_2d_blocktile(int M, int N, int K,
                                    float alpha,
                                    const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float beta,
                                    float* __restrict__ C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Block origin
  const int cRow = blockIdx.y * BM;
  const int cCol = blockIdx.x * BN;

  // Total threads = (BM/TM) * (BN/TN) = 16 * 16 = 256
  const int threadRow = threadIdx.x / (BN / TN);
  const int threadCol = threadIdx.x % (BN / TN);

  // Cooperative loading indices
  // We have 256 threads, As has BM*BK=1024 elements → each thread loads 4
  // Bs has BK*BN=1024 elements → each thread loads 4
  const int strideA = blockDim.x / BK;    // rows per load step for As
  const int strideB = blockDim.x / BN;    // rows per load step for Bs
  const int innerRowA = threadIdx.x / BK;
  const int innerColA = threadIdx.x % BK;
  const int innerRowB = threadIdx.x / BN;
  const int innerColB = threadIdx.x % BN;

  // Per-thread accumulators
  float threadResults[TM * TN] = {0.0f};
  // Registers for holding a column of As and a row of Bs
  float regA[TM];
  float regB[TN];

  for (int bkIdx = 0; bkIdx < (K + BK - 1) / BK; ++bkIdx) {
    // Load As tile (BM×BK): strided across rows, each thread loads 4 elements
    for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      const int globalRow = cRow + innerRowA + loadOffset;
      const int globalCol = bkIdx * BK + innerColA;
      As[(innerRowA + loadOffset) * BK + innerColA] =
          (globalRow < M && globalCol < K) ? A[globalRow * K + globalCol] : 0.0f;
    }

    // Load Bs tile (BK×BN): strided across rows, each thread loads 4 elements
    for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      const int globalRow = bkIdx * BK + innerRowB + loadOffset;
      const int globalCol = cCol + innerColB;
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          (globalRow < K && globalCol < N) ? B[globalRow * N + globalCol] : 0.0f;
    }

    __syncthreads();

    // Compute: for each BK step, load TM from As and TN from Bs into registers,
    // then do TM*TN FMAs entirely from registers
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // Load column fragment from As into registers
      for (int i = 0; i < TM; ++i) {
        regA[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      // Load row fragment from Bs into registers
      for (int j = 0; j < TN; ++j) {
        regB[j] = Bs[dotIdx * BN + threadCol * TN + j];
      }
      // Outer product — TM*TN FMAs from register-resident data
      for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
          threadResults[i * TN + j] += regA[i] * regB[j];
        }
      }
    }

    __syncthreads();
  }

  // Write results
  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      const int globalRow = cRow + threadRow * TM + i;
      const int globalCol = cCol + threadCol * TN + j;
      if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] =
            alpha * threadResults[i * TN + j] + beta * C[globalRow * N + globalCol];
      }
    }
  }
}

void launch_sgemm_2d_blocktile(int M, int N, int K,
                                 float alpha, const float* dA, const float* dB,
                                 float beta, float* dC) {
  dim3 block((BM / TM) * (BN / TN));  // 256 threads
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_2d_blocktile<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}
