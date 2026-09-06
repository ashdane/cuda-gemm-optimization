// 06_warptiling.cu — Kernel 6: Warp Tiling
//
// Adds an explicit warp-level decomposition layer between block-tile and
// thread-tile. Each warp owns a WARP_M×WARP_N sub-region of the block tile
// and all 32 threads in the warp cooperate on it. This improves:
//
// 1. Warp-level shared memory access locality: all 32 threads access
//    a contiguous region of As/Bs → better cache line reuse across the warp.
// 2. Instruction scheduling: the warp scheduler can hide latency better
//    when work is warp-local (avoids inter-warp shared memory conflicts).
// 3. Register file usage: warp-level tiling reduces addressing arithmetic
//    (fewer index computations per FMA cycle).
//
// Hierarchy: Block → Warp → Thread
//   Block covers BM×BN of C (e.g. 128×128)
//   Each warp covers WM×WN (e.g. 64×32)
//   Each thread covers TM×TN (e.g. 8×4)
//
// Turing-specific hypothesis: Turing has independent concurrent FP32+INT32
// execution paths. Integer address arithmetic (index calculations) can
// overlap with FP32 FMAs. This kernel generates more interleaved INT/FP
// work than earlier kernels, which may show up as a measurable Turing advantage.
// We predict a small but nonzero Turing uplift (2-5%) beyond what Pascal sees.
//
// smem: (128*16 + 16*128)*4 = 16 KB ✓ with BM=BN=128, BK=16

#include "common.cuh"

#define BM 128
#define BN 128
#define BK 16
#define WM 64     // Warp tile rows (must divide BM)
#define WN 32     // Warp tile cols (must divide BN and be ≤ 32 threads wide)
#define TM 8      // Thread tile rows within warp tile
#define TN 4      // Thread tile cols within warp tile
#define WARP_SIZE 32

// Derived: warps per block = (BM/WM) * (BN/WN) = 2 * 4 = 8
// Threads per block = 8 warps * 32 = 256
// Thread tile per thread: TM*TN = 32 accumulators (healthy for register file)

__global__ void sgemm_warptile(int M, int N, int K,
                                float alpha,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float beta,
                                float* __restrict__ C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const int cRow = blockIdx.y * BM;
  const int cCol = blockIdx.x * BN;

  const int warpIdx  = threadIdx.x / WARP_SIZE;
  const int laneIdx  = threadIdx.x % WARP_SIZE;
  const int warpRow  = warpIdx / (BN / WN);
  const int warpCol  = warpIdx % (BN / WN);

  // Thread's position within the warp tile
  const int threadRowInWarp = laneIdx / (WN / TN);
  const int threadColInWarp = laneIdx % (WN / TN);

  // Accumulators
  float threadResults[TM * TN] = {0.0f};
  float regA[TM];
  float regB[TN];

  // Cooperative loading (same as vectorized kernel)
  const int innerRowA = threadIdx.x / BK;
  const int innerColA = threadIdx.x % BK;
  const int innerRowB = threadIdx.x / (BN / 4);
  const int innerColB = threadIdx.x % (BN / 4);

  for (int bkIdx = 0; bkIdx < (K + BK - 1) / BK; ++bkIdx) {
    // Load As
    for (int lo = 0; lo < BM; lo += blockDim.x / BK) {
      int gr = cRow + innerRowA + lo, gc = bkIdx * BK + innerColA;
      As[(innerRowA + lo) * BK + innerColA] =
          (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
    // Load Bs via float4
    for (int lo = 0; lo < BK; lo += blockDim.x / (BN / 4)) {
      int gr = bkIdx * BK + innerRowB + lo;
      int gc = cCol + innerColB * 4;
      if (gr < K && gc + 3 < N) {
        float4 tmp = *reinterpret_cast<const float4*>(&B[gr * N + gc]);
        Bs[(innerRowB + lo) * BN + innerColB * 4 + 0] = tmp.x;
        Bs[(innerRowB + lo) * BN + innerColB * 4 + 1] = tmp.y;
        Bs[(innerRowB + lo) * BN + innerColB * 4 + 2] = tmp.z;
        Bs[(innerRowB + lo) * BN + innerColB * 4 + 3] = tmp.w;
      } else if (gr < K) {
        for (int i = 0; i < 4; ++i)
          Bs[(innerRowB + lo) * BN + innerColB * 4 + i] =
              (gc + i < N) ? B[gr * N + gc + i] : 0.0f;
      } else {
        for (int i = 0; i < 4; ++i)
          Bs[(innerRowB + lo) * BN + innerColB * 4 + i] = 0.0f;
      }
    }

    __syncthreads();

    // Compute warp-tiled outer product
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (int i = 0; i < TM; ++i)
        regA[i] = As[(warpRow * WM + threadRowInWarp * TM + i) * BK + dotIdx];
      for (int j = 0; j < TN; ++j)
        regB[j] = Bs[dotIdx * BN + warpCol * WN + threadColInWarp * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
          threadResults[i * TN + j] += regA[i] * regB[j];
    }

    __syncthreads();
  }

  // Write results
  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      int gr = cRow + warpRow * WM + threadRowInWarp * TM + i;
      int gc = cCol + warpCol * WN + threadColInWarp * TN + j;
      if (gr < M && gc < N)
        C[gr * N + gc] = alpha * threadResults[i * TN + j] + beta * C[gr * N + gc];
    }
  }
}

void launch_sgemm_warptile(int M, int N, int K,
                            float alpha, const float* dA, const float* dB,
                            float beta, float* dC) {
  // 8 warps × 32 threads = 256 threads per block
  dim3 block(256);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_warptile<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}
