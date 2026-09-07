#include "common.cuh"
#include <stdint.h>

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

__global__ void sgemm_double_buf_pascal(int M, int N, int K, float alpha, const float* __restrict__ A, const float* __restrict__ B, float beta, float* __restrict__ C) {
  __shared__ float As[2][BM * BK];
  __shared__ float Bs[2][BK * BN];

  const int cRow = blockIdx.y * BM;
  const int cCol = blockIdx.x * BN;

  const int threadRow = threadIdx.x / (BN / TN);
  const int threadCol = threadIdx.x % (BN / TN);

  const int innerRowA = threadIdx.x / BK;
  const int innerColA = threadIdx.x % BK;
  const int innerRowB = threadIdx.x / (BN / 4);
  const int innerColB = threadIdx.x % (BN / 4);
  const int strideA   = blockDim.x / BK;
  const int strideB   = blockDim.x / (BN / 4);

  float threadResults[TM * TN] = {0.0f};
  float regA[TM], regB[TN];

  int numTiles = (K + BK - 1) / BK;

  {
    for (int lo = 0; lo < BM; lo += strideA) {
      int gr = cRow + innerRowA + lo, gc = 0 * BK + innerColA;
      if (gr < M && gc < K) {
        As[0][(innerRowA + lo) * BK + innerColA] = A[gr * K + gc];
      } else {
        As[0][(innerRowA + lo) * BK + innerColA] = 0.0f;
      }
    }
    for (int lo = 0; lo < BK; lo += strideB) {
      int gr = 0 * BK + innerRowB + lo;
      int gc = cCol + innerColB * 4;
      if (gr < K && gc + 3 < N) {
        float4 tmp = *reinterpret_cast<const float4*>(&B[gr * N + gc]);
        Bs[0][(innerRowB + lo) * BN + innerColB * 4 + 0] = tmp.x;
        Bs[0][(innerRowB + lo) * BN + innerColB * 4 + 1] = tmp.y;
        Bs[0][(innerRowB + lo) * BN + innerColB * 4 + 2] = tmp.z;
        Bs[0][(innerRowB + lo) * BN + innerColB * 4 + 3] = tmp.w;
      } else if (gr < K) {
        for (int i = 0; i < 4; ++i) {
          if (gc + i < N) {
            Bs[0][(innerRowB + lo) * BN + innerColB * 4 + i] = B[gr * N + gc + i];
          } else {
            Bs[0][(innerRowB + lo) * BN + innerColB * 4 + i] = 0.0f;
          }
        }
      } else {
        for (int i = 0; i < 4; ++i)
          Bs[0][(innerRowB + lo) * BN + innerColB * 4 + i] = 0.0f;
      }
    }
    __syncthreads();
  }

  int buf = 0;  

  for (int bkIdx = 0; bkIdx < numTiles; ++bkIdx) {
    int nextBuf = 1 - buf;
    int nextTile = bkIdx + 1;

    float prefA[8];   
    float prefB[8];   
    if (nextTile < numTiles) {
      for (int lo = 0; lo < BM; lo += strideA) {
        int gr = cRow + innerRowA + lo, gc = nextTile * BK + innerColA;
        if (gr < M && gc < K) {
          prefA[lo / strideA] = A[gr * K + gc];
        } else {
          prefA[lo / strideA] = 0.0f;
        }
      }
      for (int lo = 0; lo < BK; lo += strideB) {
        int gr = nextTile * BK + innerRowB + lo;
        int gc = cCol + innerColB * 4;
        if (gr < K && gc + 3 < N) {
          float4 tmp = *reinterpret_cast<const float4*>(&B[gr * N + gc]);
          prefB[(lo / strideB) * 4 + 0] = tmp.x;
          prefB[(lo / strideB) * 4 + 1] = tmp.y;
          prefB[(lo / strideB) * 4 + 2] = tmp.z;
          prefB[(lo / strideB) * 4 + 3] = tmp.w;
        } else {
          for (int i = 0; i < 4; ++i) {
            if (gr < K && gc + i < N) {
              prefB[(lo / strideB) * 4 + i] = B[gr * N + gc + i];
            } else {
              prefB[(lo / strideB) * 4 + i] = 0.0f;
            }
          }
        }
      }
    }

    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (int i = 0; i < TM; ++i)
        regA[i] = As[buf][(threadRow * TM + i) * BK + dotIdx];
      for (int j = 0; j < TN; ++j)
        regB[j] = Bs[buf][dotIdx * BN + threadCol * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
          threadResults[i * TN + j] += regA[i] * regB[j];
    }

    if (nextTile < numTiles) {
      __syncthreads();
      for (int lo = 0; lo < BM; lo += strideA)
        As[nextBuf][(innerRowA + lo) * BK + innerColA] = prefA[lo / strideA];
      for (int lo = 0; lo < BK; lo += strideB)
        for (int i = 0; i < 4; ++i)
          Bs[nextBuf][(innerRowB + lo) * BN + innerColB * 4 + i] = prefB[(lo / strideB) * 4 + i];
      __syncthreads();
    }

    buf = nextBuf;
  }

  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; j += 4) {
      int gr = cRow + threadRow * TM + i;
      int gc = cCol + threadCol * TN + j;
      if (gr < M && gc + 3 < N) {
        float4 r;
        r.x = alpha*threadResults[i*TN+j+0] + beta*C[gr*N+gc+0];
        r.y = alpha*threadResults[i*TN+j+1] + beta*C[gr*N+gc+1];
        r.z = alpha*threadResults[i*TN+j+2] + beta*C[gr*N+gc+2];
        r.w = alpha*threadResults[i*TN+j+3] + beta*C[gr*N+gc+3];
        *reinterpret_cast<float4*>(&C[gr*N+gc]) = r;
      } else if (gr < M) {
        for (int jj = 0; jj < 4 && gc+jj < N; ++jj)
          C[gr*N+gc+jj] = alpha*threadResults[i*TN+j+jj] + beta*C[gr*N+gc+jj];
      }
    }
  }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda/pipeline>
#endif

void launch_sgemm_double_buf(int M, int N, int K, float alpha, const float* dA, const float* dB, float beta, float* dC) {
  dim3 block((BM / TM) * (BN / TN));  
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_double_buf_pascal<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}
