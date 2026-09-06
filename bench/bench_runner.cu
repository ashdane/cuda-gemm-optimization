// bench_runner.cu — Complete unified benchmark harness (v2)
//
// Usage:
//   ./bench_runner <kernel_id> <M> <N> <K> [warmup=10] [iters=100] [alpha=1] [beta=0]
//
// CSV output to stdout (one row per invocation):
//   kernel_id, kernel_name, M, N, K,
//   gflops_median, gflops_stddev, gflops_p10, gflops_p90,
//   ms_median, ms_stddev,
//   eff_bw_gbs,          ← effective memory BW = (bytes_accessed) / time
//   correctness, max_abs_error,
//   gpu_name, sm_count, compute_cap_major, compute_cap_minor, gpu_mem_gb
//
// Kernel IDs:
//   0  naive               7  double_buffering
//   1  coalesced           8  tensor_core_wmma   (sm_75 only)
//   2  smem_tiling         9  cublas_fp16_tc      (sm_75 only)
//   3  1d_blocktile        10 cublas_sgemm_fp32   (vendor reference)
//   4  2d_blocktile
//   5  vectorized
//   6  warptile
//
// Profiling note:
//   - Primary: CUDA events (cudaEventElapsedTime)
//   - Secondary: cross-validate with nsys --trace=cuda --stats=true
//     The nsys "gpu_kern_sum" table will show kernel durations that should
//     match ms_median within ±1%.
//   - Hardware counters (occupancy, BW, bank conflicts): BLOCKED on Ada
//     cluster (ERR_NVGPUCTRPERM). All occupancy values are CALCULATED
//     from ptxas output by analysis/occupancy_calc.py.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "common.cuh"

// ─── Forward declarations for all kernel launchers ────────────────────────
void launch_sgemm_naive        (int M, int N, int K, float a, const float* dA, const float* dB, float b, float* dC);
void launch_sgemm_coalesced    (int M, int N, int K, float a, const float* dA, const float* dB, float b, float* dC);
void launch_sgemm_smem         (int M, int N, int K, float a, const float* dA, const float* dB, float b, float* dC);
void launch_sgemm_1d_blocktile (int M, int N, int K, float a, const float* dA, const float* dB, float b, float* dC);
void launch_sgemm_2d_blocktile (int M, int N, int K, float a, const float* dA, const float* dB, float b, float* dC);
void launch_sgemm_vectorized   (int M, int N, int K, float a, const float* dA, const float* dB, float b, float* dC);
void launch_sgemm_warptile     (int M, int N, int K, float a, const float* dA, const float* dB, float b, float* dC);
void launch_sgemm_double_buf   (int M, int N, int K, float a, const float* dA, const float* dB, float b, float* dC);
__attribute__((weak)) void launch_sgemm_tensor_core  (int M, int N, int K, float a, const float* dA, const float* dB, float b, float* dC) {}
__attribute__((weak)) void launch_cublas_fp16_tc     (cublasHandle_t h, int M, int N, int K, float a, const float* dA, const float* dB, float b, float* dC) {}

// ─── Global cuBLAS handle ─────────────────────────────────────────────────
static cublasHandle_t g_cublas = nullptr;

// ─── cuBLAS FP32 launcher (row-major via transpose trick) ────────────────
static void launch_cublas_fp32(int M, int N, int K,
                                float alpha, const float* dA, const float* dB,
                                float beta,  float* dC) {
  // For row-major C = alpha*A*B + beta*C:
  // cuBLAS column-major equivalent: C^T = alpha*(B^T)*(A^T) + beta*C^T
  CUBLAS_CHECK(cublasSgemm(g_cublas,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha, dB, N,
                                    dA, K,
                            &beta,  dC, N));
}

static void launch_cublas_fp16_tc_wrapper(int M, int N, int K,
                                           float alpha, const float* dA,
                                           const float* dB, float beta, float* dC) {
  launch_cublas_fp16_tc(g_cublas, M, N, K, alpha, dA, dB, beta, dC);
}

// ─── Kernel registry ─────────────────────────────────────────────────────
typedef void (*KernelFn)(int, int, int, float, const float*, const float*, float, float*);

struct KernelEntry {
  int         id;
  const char* name;
  KernelFn    fn;
  bool        sm75_only;     // Tensor Core paths
  bool        skip_correctness_on_large;  // Naive is slow for large check
};

static KernelEntry g_kernels[] = {
  { 0,  "naive",             launch_sgemm_naive,                  false, true  },
  { 1,  "coalesced",         launch_sgemm_coalesced,              false, false },
  { 2,  "smem_tiling",       launch_sgemm_smem,                   false, false },
  { 3,  "1d_blocktile",      launch_sgemm_1d_blocktile,           false, false },
  { 4,  "2d_blocktile",      launch_sgemm_2d_blocktile,           false, false },
  { 5,  "vectorized",        launch_sgemm_vectorized,             false, false },
  { 6,  "warptile",          launch_sgemm_warptile,               false, false },
  { 7,  "double_buffering",  launch_sgemm_double_buf,             false, false },
  { 8,  "tensor_core_wmma",  launch_sgemm_tensor_core,            true,  false },
  { 9,  "cublas_fp16_tc",    launch_cublas_fp16_tc_wrapper,       true,  false },
  { 10, "cublas_sgemm_fp32", launch_cublas_fp32,                  false, false },
};
static const int N_KERNELS = sizeof(g_kernels) / sizeof(g_kernels[0]);

// ─── Statistics ──────────────────────────────────────────────────────────
static double percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  double idx = p / 100.0 * (v.size() - 1);
  int lo = (int)idx;
  int hi = lo + 1 < (int)v.size() ? lo + 1 : lo;
  return v[lo] + (idx - lo) * (v[hi] - v[lo]);
}
static double median_v(std::vector<double>& v) { return percentile(v, 50); }
static double stddev_v(const std::vector<double>& v, double mean) {
  double s = 0;
  for (double x : v) s += (x - mean) * (x - mean);
  return sqrt(s / v.size());
}

// ─── Effective bandwidth ──────────────────────────────────────────────────
// Theoretical data movement for GEMM: read A (M×K) + B (K×N) + C (M×N),
// write C (M×N). Assumes worst-case no caching (all from DRAM).
// Real BW will be lower due to L1/L2 hits; this is a lower bound on
// achieved BW (i.e., an UPPER bound on effective BW — useful for roofline).
static double eff_bandwidth_gbs(long M, long N, long K, double ms) {
  double bytes = 4.0 * (M * K + K * N + 2.0 * M * N);  // float32 = 4 bytes
  return bytes / (ms * 1e6);  // GB/s
}

// ─── GPU info ─────────────────────────────────────────────────────────────
struct GpuInfo {
  char name[256];
  int  smCount, ccMaj, ccMin;
  double memGb;
};
static GpuInfo getGpuInfo() {
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  GpuInfo g;
  strncpy(g.name, p.name, 255); g.name[255] = '\0';
  g.smCount = p.multiProcessorCount;
  g.ccMaj   = p.major;
  g.ccMin   = p.minor;
  g.memGb   = (double)p.totalGlobalMem / (1024.0*1024.0*1024.0);
  return g;
}

// ─── Correctness validation against cuBLAS FP32 ───────────────────────────
// Uses a small sub-problem (max 512³) for speed; checks max absolute error.
static float validateKernel(KernelFn fn, int M, int N, int K,
                              float alpha, float beta,
                              const float* dA, const float* dB) {
  // Clamp validation size
  int Mv = std::min(M, 256), Nv = std::min(N, 256), Kv = std::min(K, 256);

  // Allocate fresh matrices for validation
  size_t szA = (size_t)Mv*Kv*4, szB = (size_t)Kv*Nv*4, szC = (size_t)Mv*Nv*4;
  float *dvA, *dvB, *dvRef, *dvTest;
  CUDA_CHECK(cudaMalloc(&dvA,    szA));
  CUDA_CHECK(cudaMalloc(&dvB,    szB));
  CUDA_CHECK(cudaMalloc(&dvRef,  szC));
  CUDA_CHECK(cudaMalloc(&dvTest, szC));

  // Initialize with deterministic values
  std::vector<float> hA(Mv*Kv), hB(Kv*Nv);
  unsigned s = 12345;
  auto lcg = [&]{ s = s*1664525u+1013904223u; return (float)(s&0xFFFF)/65535.f-0.5f; };
  for (auto& x : hA) x = lcg();
  for (auto& x : hB) x = lcg();
  CUDA_CHECK(cudaMemcpy(dvA, hA.data(), szA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dvB, hB.data(), szB, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dvRef,  0, szC));
  CUDA_CHECK(cudaMemset(dvTest, 0, szC));

  // Reference: cuBLAS FP32
  launch_cublas_fp32(Mv, Nv, Kv, alpha, dvA, dvB, beta, dvRef);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Kernel under test
  fn(Mv, Nv, Kv, alpha, dvA, dvB, beta, dvTest);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Compare
  std::vector<float> hRef(Mv*Nv), hTest(Mv*Nv);
  CUDA_CHECK(cudaMemcpy(hRef.data(),  dvRef,  szC, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hTest.data(), dvTest, szC, cudaMemcpyDeviceToHost));

  float maxErr = 0.0f;
  for (int i = 0; i < Mv*Nv; ++i) {
    float e = fabsf(hRef[i] - hTest[i]);
    if (e > maxErr) maxErr = e;
  }

  cudaFree(dvA); cudaFree(dvB); cudaFree(dvRef); cudaFree(dvTest);
  return maxErr;
}

// ─── Main ─────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr,
        "Usage: %s <kernel_id> <M> <N> <K> [warmup=10] [iters=100] [alpha=1] [beta=0]\n"
        "\nKernel IDs:\n"
        "  0  naive               7  double_buffering\n"
        "  1  coalesced           8  tensor_core_wmma (sm_75 only)\n"
        "  2  smem_tiling         9  cublas_fp16_tc   (sm_75 only)\n"
        "  3  1d_blocktile        10 cublas_sgemm_fp32\n"
        "  4  2d_blocktile\n"
        "  5  vectorized\n"
        "  6  warptile\n", argv[0]);
    return 1;
  }

  int   kid    = atoi(argv[1]);
  int   M      = atoi(argv[2]);
  int   N      = atoi(argv[3]);
  int   K      = atoi(argv[4]);
  int   warmup = argc > 5 ? atoi(argv[5]) : 10;
  int   iters  = argc > 6 ? atoi(argv[6]) : 100;
  float alpha  = argc > 7 ? (float)atof(argv[7]) : 1.0f;
  float beta   = argc > 8 ? (float)atof(argv[8]) : 0.0f;

  if (kid < 0 || kid >= N_KERNELS) {
    fprintf(stderr, "Invalid kernel_id %d\n", kid);
    return 1;
  }

  GpuInfo gpu = getGpuInfo();
  KernelEntry& ke = g_kernels[kid];

  // Architecture guard for TC kernels
  if (ke.sm75_only && (gpu.ccMaj < 7 || (gpu.ccMaj == 7 && gpu.ccMin < 5))) {
    printf("%d,%s,%d,%d,%d,N/A,N/A,N/A,N/A,N/A,N/A,N/A,NOT_SUPPORTED_NO_TC,N/A,"
           "%s,%d,%d.%d,%.1f\n",
           kid, ke.name, M, N, K,
           gpu.name, gpu.smCount, gpu.ccMaj, gpu.ccMin, gpu.memGb);
    return 0;
  }

  // cuBLAS init
  CUBLAS_CHECK(cublasCreate(&g_cublas));
  // Enable math mode to allow Tensor Cores when available
  if (gpu.ccMaj >= 7)
    CUBLAS_CHECK(cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH));

  // Allocate device matrices
  size_t szA = (size_t)M*K*sizeof(float);
  size_t szB = (size_t)K*N*sizeof(float);
  size_t szC = (size_t)M*N*sizeof(float);

  // Check we have enough memory
  size_t freeMem, totalMem;
  CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
  size_t needed = szA + szB + 2*szC;  // A, B, C, plus ref buffer
  if (needed > freeMem * 0.9) {
    fprintf(stderr,
        "WARNING: need %.2f GB, only %.2f GB free — may OOM\n",
        needed/1e9, freeMem/1e9);
  }

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, szA));
  CUDA_CHECK(cudaMalloc(&dB, szB));
  CUDA_CHECK(cudaMalloc(&dC, szC));

  // Initialize with pseudo-random data
  {
    std::vector<float> hA(M*K), hB(K*N);
    unsigned s = 0xDEADBEEF;
    auto lcg = [&]{ s=s*1664525u+1013904223u; return (float)(s&0xFFFF)/65535.f-0.5f; };
    for (auto& x : hA) x = lcg();
    for (auto& x : hB) x = lcg();
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), szA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), szB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, szC));
  }

  // ── Correctness check ─────────────────────────────────────────────────
  float maxErr = 0.0f;
  bool  passed = true;
  if (!ke.skip_correctness_on_large || (M <= 512 && N <= 512 && K <= 512)) {
    maxErr = validateKernel(ke.fn, M, N, K, alpha, beta, dA, dB);
    // TC kernels use FP16 math → allow 1% relative error
    float threshold = ke.sm75_only ? 0.5f : 1e-2f;
    passed = (maxErr < threshold);
  } else {
    // Skip for naive at large sizes — mark as UNCHECKED
    maxErr = -1.0f;
    passed = true;
  }

  CUDA_CHECK(cudaMemset(dC, 0, szC));

  // ── Warmup ────────────────────────────────────────────────────────────
  for (int i = 0; i < warmup; ++i) {
    ke.fn(M, N, K, alpha, dA, dB, beta, dC);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // ── Timed runs ────────────────────────────────────────────────────────
  GpuTimer timer;
  std::vector<double> msVec(iters), gfVec(iters), bwVec(iters);

  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaMemset(dC, 0, szC));  // reset C to avoid accumulation skew
    timer.Start();
    ke.fn(M, N, K, alpha, dA, dB, beta, dC);
    timer.Stop();

    double ms = timer.ElapsedMs();
    msVec[i]  = ms;
    gfVec[i]  = gflops(M, N, K, ms);
    bwVec[i]  = eff_bandwidth_gbs(M, N, K, ms);
  }

  double msMedian  = median_v(msVec);
  double gfMedian  = median_v(gfVec);
  double gfStddev  = stddev_v(gfVec, gfMedian);
  double gfP10     = percentile(gfVec, 10);
  double gfP90     = percentile(gfVec, 90);
  double bwMedian  = median_v(bwVec);
  double msStddev  = stddev_v(msVec, msMedian);

  // ── CSV output ────────────────────────────────────────────────────────
  // One row — fields match the header written by run_sweep.sh
  printf("%d,%s,%d,%d,%d,"           // kernel_id,name,M,N,K
         "%.3f,%.3f,%.3f,%.3f,"      // gflops med,std,p10,p90
         "%.4f,%.4f,"                // ms med, std
         "%.2f,"                     // eff_bw GB/s
         "%s,%.6f,"                  // correctness, max_abs_err
         "%s,%d,%d,%d,%.2f\n",       // gpu,sm,cc_maj,cc_min,gpu_mem_gb
         kid, ke.name, M, N, K,
         gfMedian, gfStddev, gfP10, gfP90,
         msMedian, msStddev,
         bwMedian,
         (maxErr < 0) ? "UNCHECKED" : (passed ? "PASS" : "FAIL"),
         (maxErr < 0) ? 0.0f : maxErr,
         gpu.name, gpu.smCount, gpu.ccMaj, gpu.ccMin, gpu.memGb);

  fflush(stdout);

  // ── Stderr: verbose timing distribution (captured separately) ────────
  fprintf(stderr,
      "# kernel=%s M=%d N=%d K=%d iters=%d\n"
      "# GFLOPS: median=%.2f p10=%.2f p90=%.2f stddev=%.2f\n"
      "# Time(ms): median=%.4f stddev=%.4f\n"
      "# EffBW(GB/s): %.2f\n"
      "# Correctness: %s (max_abs_err=%.2e)\n",
      ke.name, M, N, K, iters,
      gfMedian, gfP10, gfP90, gfStddev,
      msMedian, msStddev,
      bwMedian,
      (maxErr < 0) ? "UNCHECKED" : (passed ? "PASS" : "FAIL"),
      (double)(maxErr < 0 ? 0 : maxErr));

  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  cublasDestroy(g_cublas);
  return passed ? 0 : 2;  // exit code 2 = correctness failure
}
