#!/usr/bin/env bash
# run_sweep.sh — Complete benchmark sweep (v2, 11 kernels)
#
# §2.1: All kernels at 4096³ (fixed headline)
# §2.3: Dimension sweep (square, non-square, awkward) for best kernels + cuBLAS
# §2.6: Alternative approach comparisons (cuBLAS FP32, cuBLAS FP16 TC, WMMA TC)
#
# Noise control: --warmup=10 --iters=100 (configurable via env vars)
# Output: CSV per node/GPU in $OUTPUT_DIR

set -euo pipefail

module load u18/cuda/11.6 2>/dev/null || true

OUTPUT_DIR="${1:-/scratch/cuda_gemm_results}"
mkdir -p "$OUTPUT_DIR"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1 | tr ' ' '_')
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1)
SM_MAJOR=$(echo $COMPUTE_CAP | cut -d. -f1)
SM_MINOR=$(echo $COMPUTE_CAP | cut -d. -f2)
HOSTNAME_SHORT=$(hostname -s)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

OUTFILE="${OUTPUT_DIR}/results_${HOSTNAME_SHORT}_${GPU_NAME}_${TIMESTAMP}.csv"

WARMUP="${BENCH_WARMUP:-10}"
ITERS="${BENCH_ITERS:-100}"

echo "================================================================"
echo "CUDA GEMM Benchmark Sweep v2"
echo "Host:         $HOSTNAME_SHORT"
echo "GPU:          $GPU_NAME (sm_${SM_MAJOR}.${SM_MINOR})"
echo "Warmup/Iters: ${WARMUP}/${ITERS}"
echo "Output:       $OUTFILE"
echo "================================================================"

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
if (( SM_MAJOR >= 7 )) && (( (SM_MAJOR*10 + SM_MINOR) >= 75 )); then
    BENCH="$BENCH_DIR/bench_runner_sm75"
    echo "Using sm_75 binary (Tensor Cores available)"
else
    BENCH="$BENCH_DIR/bench_runner_sm61"
    echo "Using sm_61 binary (no Tensor Cores)"
fi

if [ ! -f "$BENCH" ]; then
    echo "ERROR: $BENCH not found. Run 'make' first."
    exit 1
fi

# CSV header
echo "kernel_id,kernel_name,M,N,K,gflops_median,gflops_stddev,gflops_p10,gflops_p90,ms_median,ms_stddev,eff_bw_gbs,correctness,max_abs_error,gpu_name,sm_count,compute_cap_major,compute_cap_minor,gpu_mem_gb" \
    > "$OUTFILE"

# ─── Helper: run one (kernel, size) pair, append CSV row ──────────────────
run_kernel() {
    local kid=$1 m=$2 n=$3 k=$4
    local kname
    kname=$(echo "0:naive,1:coalesced,2:smem_tiling,3:1d_blocktile,4:2d_blocktile,5:vectorized,6:warptile,7:double_buffering,8:tensor_core_wmma,9:cublas_fp16_tc,10:cublas_sgemm_fp32" \
            | tr ',' '\n' | grep "^${kid}:" | cut -d: -f2)
    printf "  K%-2d %-22s M=%-5d N=%-5d K=%-5d  " "$kid" "($kname)" "$m" "$n" "$k"

    local result
    if result=$("$BENCH" "$kid" "$m" "$n" "$k" "$WARMUP" "$ITERS" 1.0 0.0 2>/dev/null); then
        echo "$result" >> "$OUTFILE"
        local gf; gf=$(echo "$result" | cut -d, -f6)
        local ok; ok=$(echo "$result" | cut -d, -f13)
        printf "→  %8.1f GFLOPS  [%s]\n" "$gf" "$ok"
    else
        printf "→  FAILED\n"
    fi
}

# ─── §2.1 Fixed size (4096³) — ALL kernel versions ───────────────────────
echo ""
echo "=== §2.1: Fixed size progression (M=N=K=4096) ==="
FIXED=4096

# Always: kernels 0–7 + cuBLAS FP32
for kid in 0 1 2 3 4 5 6 7 10; do
    run_kernel "$kid" "$FIXED" "$FIXED" "$FIXED"
done

# Turing only: TC kernels
if (( SM_MAJOR >= 7 )) && (( (SM_MAJOR*10 + SM_MINOR) >= 75 )); then
    echo ""
    echo "  [Turing-only Tensor Core kernels]"
    run_kernel 8 "$FIXED" "$FIXED" "$FIXED"   # WMMA FP16
    run_kernel 9 "$FIXED" "$FIXED" "$FIXED"   # cuBLAS FP16 TC
fi

# ─── §2.3: Dimension sweep — best kernels + cuBLAS ───────────────────────
# Best hand-tuned kernels: vectorized(5), double_buf(7), warptile(6)
# Plus cuBLAS FP32(10) as reference ceiling
# On Turing: also cuBLAS FP16 TC(9)
if (( SM_MAJOR >= 7 )) && (( (SM_MAJOR*10 + SM_MINOR) >= 75 )); then
    SWEEP_KIDS=(5 6 7 9 10)
else
    SWEEP_KIDS=(5 6 7 10)
fi

echo ""
echo "=== §2.3a: Square power-of-two sizes ==="
SQUARE_SIZES=(256 512 1024 2048 4096 8192)
for sz in "${SQUARE_SIZES[@]}"; do
    echo ""
    echo "  --- M=N=K=${sz} ---"
    for kid in "${SWEEP_KIDS[@]}"; do
        run_kernel "$kid" "$sz" "$sz" "$sz"
    done
done

echo ""
echo "=== §2.3b: Non-square shapes (tall-skinny, wide-flat, asymmetric) ==="
# Format: "M N K  description"
NON_SQUARE_SHAPES=(
    "8192  512  512   tall_skinny_M_dominant"
    "512   8192 512   wide_flat_N_dominant"
    "512   512  8192  long_K_reduction"
    "2048  512  4096  moderate_asymmetric"
    "1024  4096 2048  transpose_like"
    "256   256  16384 very_long_K"
    "4096  128  4096  skinny_batch"
)
for entry in "${NON_SQUARE_SHAPES[@]}"; do
    read m n k desc <<< "$entry"
    echo ""
    echo "  --- ${desc} M=${m} N=${n} K=${k} ---"
    for kid in "${SWEEP_KIDS[@]}"; do
        run_kernel "$kid" "$m" "$n" "$k"
    done
done

echo ""
echo "=== §2.3c: Awkward/non-power-of-two sizes ==="
AWKWARD_SHAPES=(
    "3000 1500 1500  spec_example"
    "3001 3001 3001  prime_adjacent_cube"
    "4097 4097 4097  just_over_pow2"
    "1000 1000 1000  round_number_non_pow2"
    "768  768  768   divisible_by_256_not_512"
    "511  513  511   prime_adjacent_asymm"
    "1024 3000 1024  non_pow2_N"
    "3333 3333 3333  large_prime_adjacent"
)
for entry in "${AWKWARD_SHAPES[@]}"; do
    read m n k desc <<< "$entry"
    echo ""
    echo "  --- ${desc} M=${m} N=${n} K=${k} ---"
    for kid in "${SWEEP_KIDS[@]}"; do
        run_kernel "$kid" "$m" "$n" "$k"
    done
done

# ─── §2.6: Small-size overhead study ─────────────────────────────────────
echo ""
echo "=== §2.6: Small-size kernel launch overhead study ==="
echo "     (where launch overhead dominates — naive can win on tiny M)"
SMALL_SIZES=(16 32 64 128 256)
for sz in "${SMALL_SIZES[@]}"; do
    echo ""
    echo "  --- M=N=K=${sz} (tiny) ---"
    for kid in 0 2 7 10; do  # naive, smem, double_buf, cublas
        run_kernel "$kid" "$sz" "$sz" "$sz"
    done
done

echo ""
echo "================================================================"
echo "Sweep complete: $(wc -l < "$OUTFILE") rows (incl. header)"
echo "Output: $OUTFILE"
echo "================================================================"

# Copy to home immediately (scratch is purged after 7 days)
HOME_RESULTS="/home2/${USER}/cuda_gemm/results"
mkdir -p "$HOME_RESULTS"
cp "$OUTFILE" "$HOME_RESULTS/"
echo "Copied to: $HOME_RESULTS/$(basename "$OUTFILE")"
