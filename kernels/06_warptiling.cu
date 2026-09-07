#include "common.cuh"

#define BM 128
#define BN 128
#define BK 16
#define WM 64     
#define WN 32     
#define TM 8      
#define TN 4      
#define WARP_SIZE 32

__global__ void sgemm_warptile(int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const int cRow = blockIdx.y * BM;
  const int cCol = blockIdx.x * BN;

  const int warpIdx  = threadIdx.x / WARP_SIZE;
  const int laneIdx  = threadIdx.x % WARP_SIZE;
  const int warpRow  = warpIdx / (BN / WN);
  const int warpCol  = warpIdx % (BN / WN);

  const int threadRowInWarp = laneIdx / (WN / TN);
  const int threadColInWarp = laneIdx % (WN / TN);

  float threadResults[TM * TN] = {0.0f};
  float regA[TM];
  float regB[TN];

  const int innerRowA = threadIdx.x / BK;
  const int innerColA = threadIdx.x % BK;
  const int innerRowB = threadIdx.x / (BN / 4);
  const int innerColB = threadIdx.x % (BN / 4);

  for (int bkIdx = 0; bkIdx < (K + BK - 1) / BK; ++bkIdx) {
    for (int lo = 0; lo < BM; lo += blockDim.x / BK) {
      int gr = cRow + innerRowA + lo, gc = bkIdx * BK + innerColA;
      if (gr < M && gc < K) {
        As[(innerRowA + lo) * BK + innerColA] = A[gr * K + gc];
      } else {
        As[(innerRowA + lo) * BK + innerColA] = 0.0f;
      }
    }
    
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
        for (int i = 0; i < 4; ++i) {
          if (gc + i < N) {
            Bs[(innerRowB + lo) * BN + innerColB * 4 + i] = B[gr * N + gc + i];
          } else {
            Bs[(innerRowB + lo) * BN + innerColB * 4 + i] = 0.0f;
          }
        }
      } else {
        for (int i = 0; i < 4; ++i)
          Bs[(innerRowB + lo) * BN + innerColB * 4 + i] = 0.0f;
      }
    }

    __syncthreads();

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

  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      int gr = cRow + warpRow * WM + threadRowInWarp * TM + i;
      int gc = cCol + warpCol * WN + threadColInWarp * TN + j;
      if (gr < M && gc < N)
        C[gr * N + gc] = alpha * threadResults[i * TN + j] + beta * C[gr * N + gc];
    }
  }
}

void launch_sgemm_warptile(int M, int N, int K, float alpha, const float* dA, const float* dB, float beta, float* dC) {
  dim3 block(256);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_warptile<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}
