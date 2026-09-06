// common.cuh — Shared utilities for all GEMM kernels
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ─── Error-checking macros ─────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t _e = (call);                                                     \
    if (_e != cudaSuccess) {                                                     \
      fprintf(stderr, "CUDA error %s:%d — %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(_e));                                            \
      exit(EXIT_FAILURE);                                                         \
    }                                                                            \
  } while (0)

#define CUBLAS_CHECK(call)                                                       \
  do {                                                                           \
    cublasStatus_t _s = (call);                                                  \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                           \
      fprintf(stderr, "cuBLAS error %s:%d — code %d\n", __FILE__, __LINE__,     \
              (int)_s);                                                           \
      exit(EXIT_FAILURE);                                                         \
    }                                                                            \
  } while (0)

// ─── Timing helpers ────────────────────────────────────────────────────────
struct GpuTimer {
  cudaEvent_t start, stop;
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void Start() { CUDA_CHECK(cudaEventRecord(start)); }
  void Stop()  { CUDA_CHECK(cudaEventRecord(stop)); }
  float ElapsedMs() {
    float ms;
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// ─── GFLOPS calculation ────────────────────────────────────────────────────
// Standard 2*M*N*K flops for a dense SGEMM
inline double gflops(long M, long N, long K, double ms) {
  return 2.0 * M * N * K / (ms * 1e6);  // ms → GFLOPS
}

// ─── Correctness check ─────────────────────────────────────────────────────
// Computes max absolute error between two float arrays of length n
inline float maxAbsError(const float* ref, const float* test, size_t n) {
  float maxErr = 0.0f;
  for (size_t i = 0; i < n; i++) {
    float e = fabsf(ref[i] - test[i]);
    if (e > maxErr) maxErr = e;
  }
  return maxErr;
}
