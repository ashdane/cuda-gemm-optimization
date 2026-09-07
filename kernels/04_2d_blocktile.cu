#include "common.cuh"

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 8
#endif
#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 8
#endif     

__global__ void sgemm_2d_blocktile(int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C) {
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const int cRow = blockIdx.y * BM;
  const int cCol = blockIdx.x * BN;

  const int threadRow = threadIdx.x / (BN / TN);
  const int threadCol = threadIdx.x % (BN / TN);

  const int strideA = blockDim.x / BK;    
  const int strideB = blockDim.x / BN;    
  const int innerRowA = threadIdx.x / BK;
  const int innerColA = threadIdx.x % BK;
  const int innerRowB = threadIdx.x / BN;
  const int innerColB = threadIdx.x % BN;

  float threadResults[TM * TN] = {0.0f};
  
  float regA[TM];
  float regB[TN];

  for (int bkIdx = 0; bkIdx < (K + BK - 1) / BK; ++bkIdx) {
    for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      const int globalRow = cRow + innerRowA + loadOffset;
      const int globalCol = bkIdx * BK + innerColA;
      if (globalRow < M && globalCol < K) {
        As[(innerRowA + loadOffset) * BK + innerColA] = A[globalRow * K + globalCol];
      } else {
        As[(innerRowA + loadOffset) * BK + innerColA] = 0.0f;
      }
    }

    for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      const int globalRow = bkIdx * BK + innerRowB + loadOffset;
      const int globalCol = cCol + innerColB;
      if (globalRow < K && globalCol < N) {
        Bs[(innerRowB + loadOffset) * BN + innerColB] = B[globalRow * N + globalCol];
      } else {
        Bs[(innerRowB + loadOffset) * BN + innerColB] = 0.0f;
      }
    }

    __syncthreads();

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (int i = 0; i < TM; ++i) {
        regA[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      
      for (int j = 0; j < TN; ++j) {
        regB[j] = Bs[dotIdx * BN + threadCol * TN + j];
      }
      
      for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
          threadResults[i * TN + j] += regA[i] * regB[j];
        }
      }
    }

    __syncthreads();
  }

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

void launch_sgemm_2d_blocktile(int M, int N, int K, float alpha, const float* dA, const float* dB, float beta, float* dC) {
  dim3 block((BM / TM) * (BN / TN));  
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_2d_blocktile<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}
