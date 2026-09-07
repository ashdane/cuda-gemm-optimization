#include "common.cuh"

#define BM 64   
#define BN 64   
#define BK 8    
#define TM 8    

__global__ void sgemm_1d_blocktile(int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C)
{  
  __shared__ float As[BM * BK];  
  __shared__ float Bs[BK * BN];  
  const int cRow = blockIdx.y * BM;
  const int cCol = blockIdx.x * BN;
  const int threadRow = threadIdx.x / BN;  
  const int threadCol = threadIdx.x % BN;  

  float threadResults[TM] = {0.0f};

  const int innerRowA = threadIdx.x / BK;  
  const int innerColA = threadIdx.x % BK;
  const int innerRowB = threadIdx.x / BN;  
  const int innerColB = threadIdx.x % BN;

  for (int bkIdx = 0; bkIdx < (K + BK - 1) / BK; ++bkIdx)
  {
    const int aRow = cRow + innerRowA;
    const int aCol = bkIdx * BK + innerColA;
    if (aRow < M && aCol < K) {
      As[innerRowA * BK + innerColA] = A[aRow * K + aCol];
    } else {
      As[innerRowA * BK + innerColA] = 0.0f;
    }

    const int bRow = bkIdx * BK + innerRowB;
    const int bCol = cCol + innerColB;
    if (bRow < K && bCol < N) {
      Bs[innerRowB * BN + innerColB] = B[bRow * N + bCol];
    } else {
      Bs[innerRowB * BN + innerColB] = 0.0f;
    }

    __syncthreads();

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float bVal = Bs[dotIdx * BN + threadCol];  
      for (int resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * bVal;
      }
    }

    __syncthreads();
  }

  for (int resIdx = 0; resIdx < TM; ++resIdx) {
    const int globalRow = cRow + threadRow * TM + resIdx;
    const int globalCol = cCol + threadCol;
    if (globalRow < M && globalCol < N) {
      C[globalRow * N + globalCol] =
          alpha * threadResults[resIdx] + beta * C[globalRow * N + globalCol];
    }
  }
}

void launch_sgemm_1d_blocktile(int M, int N, int K, float alpha, const float* dA, const float* dB, float beta, float* dC) {
  dim3 block((BM / TM) * BN);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_1d_blocktile<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}
