#!/usr/bin/env bash
#SBATCH --job-name=gemm_turing_sweep
#SBATCH --partition=long
#SBATCH --nodelist=gnode50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH -A cvit
#SBATCH --qos=medium
#SBATCH --time=08:00:00
#SBATCH --output=/home2/%u/cuda_gemm/logs/turing_sweep_%j.log
#SBATCH --error=/home2/%u/cuda_gemm/logs/turing_sweep_%j.err

# turing_sweep.slurm — Full GEMM benchmark sweep on Turing (RTX 2080 Ti, sm_75)
#
# IMPORTANT: If gnode50 is busy, check 'sinfo' for idle Turing nodes (gnode43-92)
# and update --nodelist accordingly. Do not run on Pascal nodes for this script.
#
# Additional experiments vs. Pascal:
#   - Tensor Core (FP16 WMMA) kernel (kernel 7)
#   - Turing FP32+INT32 concurrent execution hypothesis test

set -euo pipefail

echo "========================================"
echo "CUDA GEMM Turing Sweep — Job $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "========================================"

module load u18/cuda/11.6

echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
    --format=csv,noheader

# Verify we actually got a Turing GPU (paranoia check)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1)
SM_MAJOR=$(echo $COMPUTE_CAP | cut -d. -f1)
if [ "$SM_MAJOR" -lt 7 ]; then
    echo "ERROR: Expected Turing (sm_7x) but got sm_${COMPUTE_CAP}. Wrong node!"
    exit 1
fi
echo "✓ Confirmed Turing GPU (sm_${COMPUTE_CAP})"

SCRATCH="/scratch/${USER}/cuda_gemm_turing_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"

SRC_DIR="/home2/${USER}/cuda_gemm"
cp -r "$SRC_DIR/kernels" "$SCRATCH/"
cp -r "$SRC_DIR/bench"   "$SCRATCH/"
cd "$SCRATCH/bench"

echo ""
echo "=== Building Turing binary (sm_75) ==="
make turing 2>&1 | tee "$SCRATCH/build_turing.log"

echo ""
echo "=== ptxas register/smem info (sm_75) ==="
grep "ptxas info" "$SCRATCH/bench/ptxas_sm75.log" | tee "$SCRATCH/ptxas_summary_sm75.txt"

# ─── nsys CUDA API traces ──────────────────────────────────────────────────
echo ""
echo "=== nsys CUDA API traces (GPU metric counters are blocked on this cluster) ==="

for kid in 5 6 7 8; do
    NSYS_OUT="$SCRATCH/nsys_turing_k${kid}_4096"
    nsys profile \
        --trace=cuda \
        --stats=true \
        --output="$NSYS_OUT" \
        ./bench_runner_sm75 "$kid" 4096 4096 4096 5 20 1.0 0.0 \
        > "$SCRATCH/nsys_k${kid}_stdout.txt" 2>&1 || true
    echo "  Kernel $kid trace: ${NSYS_OUT}.nsys-rep"
done

# ─── Main benchmark sweep ──────────────────────────────────────────────────
echo ""
echo "=== Running main benchmark sweep (includes Tensor Cores) ==="
bash run_sweep.sh "$SCRATCH/results"

# ─── Turing-specific: Tensor Core at multiple sizes ───────────────────────
echo ""
echo "=== Tensor Core size sweep ==="
TC_OUTFILE="$SCRATCH/results/tc_sweep.csv"
echo "kernel_id,kernel_name,M,N,K,gflops_median,gflops_stddev,gflops_min,gflops_max,ms_median,ms_half_range,correctness,max_abs_error,gpu_name,sm_count,compute_cap" \
    > "$TC_OUTFILE"

for sz in 1024 2048 4096 8192; do
    # Tensor Core requires sizes divisible by WMMA_M=16
    if (( sz % 16 == 0 )); then
        echo "  TC M=$sz N=$sz K=$sz"
        ./bench_runner_sm75 7 "$sz" "$sz" "$sz" 5 50 1.0 0.0 >> "$TC_OUTFILE" 2>/dev/null || true
        # cuBLAS FP16 (cublasGemmEx) for comparison
        ./bench_runner_sm75 8 "$sz" "$sz" "$sz" 5 50 1.0 0.0 >> "$TC_OUTFILE" 2>/dev/null || true
    fi
done

# ─── Parameter sensitivity sweep (Turing) ─────────────────────────────────
echo ""
echo "=== Parameter sensitivity sweep (Turing) ==="
bash param_sensitivity.sh "$SCRATCH/param_results"

# ─── Copy to home ─────────────────────────────────────────────────────────
echo ""
echo "=== Copying results to home ==="
HOME_DEST="/home2/${USER}/cuda_gemm/results"
mkdir -p "$HOME_DEST"

cp "$SCRATCH"/results/*.csv      "$HOME_DEST/" 2>/dev/null || true
cp "$SCRATCH"/param_results/*.csv "$HOME_DEST/" 2>/dev/null || true
cp "$SCRATCH/ptxas_summary_sm75.txt" "$HOME_DEST/"
cp "$SCRATCH/build_turing.log"       "$HOME_DEST/"

# Copy nsys stdout (text stats — no large binary .nsys-rep files)
for kid in 5 6 7 8; do
    cp "$SCRATCH/nsys_k${kid}_stdout.txt" "$HOME_DEST/" 2>/dev/null || true
done

echo ""
echo "========================================"
echo "Turing sweep complete — $(date)"
echo "Results in: $HOME_DEST"
echo "========================================"
