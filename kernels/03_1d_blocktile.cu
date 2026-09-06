// 03_1d_blocktile.cu — Kernel 3: 1D Block Tiling
//
// Each thread now computes TM output elements (a 1×TM strip of output C).
// Thread count per block stays the same (BM*BN/TM), but each thread does
// more FMA work before needing to re-issue global/shared loads.
//
// Why this helps: more work per thread → better instruction-level parallelism
// (ILP), deeper pipeline utilization, fewer synchronization barriers per
// unit of output. Reduces kernel launch overhead amortization for large M.
//
// Bottleneck targeted: low compute utilization in smem kernel — threads
// do 1 FMA per loaded element, leaving FP32 pipelines mostly idle.
//
// Expected: step function improvement in GFLOPS; PTX register count increases.
// Registers: ~TM*2 + BK + overhead ≈ ~30-40 regs for TM=8 — check with ptxas.
//
// Constraint: BM*BK*4 + BK*BN*4 bytes shared memory must fit in 48 KB (Pascal)
// or 64 KB (Turing). With BM=BN=64, BK=8: (64*8 + 8*64)*4 = 4096 bytes ✓

#include "common.cuh"

// Tunable parameters — compile-time constants for best register allocation
#define BM 64   // Block tile rows (output rows covered by one block)
#define BN 64   // Block tile cols (output cols covered by one block)
#define BK 8    // Block tile depth (K-dimension tile)
#define TM 8    // Thread tile rows (output rows per thread)

__global__ void sgemm_1d_blocktile(int M, int N, int K,
                                    float alpha,
                                    const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float beta,
                                    float* __restrict__ C) {
  // Shared memory tiles
  __shared__ float As[BM * BK];  // BM×BK tile of A
  __shared__ float Bs[BK * BN];  // BK×BN tile of B

  // Block origin in C
  const int cRow = blockIdx.y * BM;
  const int cCol = blockIdx.x * BN;

  // Thread index within block (linearized: BN/TM threads per row of output)
  // Total threads per block = (BM / TM) * BN = 8 * 64 = 512
  const int threadRow = threadIdx.x / BN;  // which output-row strip this thread owns
  const int threadCol = threadIdx.x % BN;  // which output column

  // Accumulator registers: TM values per thread
  float threadResults[TM] = {0.0f};

  // Strided loading indices (for loading As and Bs tiles cooperatively)
  // BM*BK = 512 elements, BK*BN = 512 elements, blockDim.x = BM/TM * BN = 512
  const int innerRowA = threadIdx.x / BK;  // row within As tile to load
  const int innerColA = threadIdx.x % BK;
  const int innerRowB = threadIdx.x / BN;  // row within Bs tile to load
  const int innerColB = threadIdx.x % BN;

  for (int bkIdx = 0; bkIdx < (K + BK - 1) / BK; ++bkIdx) {
    // Load As: each thread loads one element
    const int aRow = cRow + innerRowA;
    const int aCol = bkIdx * BK + innerColA;
    As[innerRowA * BK + innerColA] =
        (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;

    // Load Bs: each thread loads one element  
    const int bRow = bkIdx * BK + innerRowB;
    const int bCol = cCol + innerColB;
    Bs[innerRowB * BN + innerColB] =
        (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;

    __syncthreads();

    // Compute: each thread accumulates TM results from this tile
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float bVal = Bs[dotIdx * BN + threadCol];  // shared, same for all TM iterations
      for (int resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * bVal;
      }
    }

    __syncthreads();
  }

  // Write results back to global memory
  for (int resIdx = 0; resIdx < TM; ++resIdx) {
    const int globalRow = cRow + threadRow * TM + resIdx;
    const int globalCol = cCol + threadCol;
    if (globalRow < M && globalCol < N) {
      C[globalRow * N + globalCol] =
          alpha * threadResults[resIdx] + beta * C[globalRow * N + globalCol];
    }
  }
}

void launch_sgemm_1d_blocktile(int M, int N, int K,
                                 float alpha, const float* dA, const float* dB,
                                 float beta, float* dC) {
  // (BM/TM) * BN = 8 * 64 = 512 threads per block
  dim3 block((BM / TM) * BN);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_1d_blocktile<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}
