#!/usr/bin/env bash
#SBATCH --job-name=gemm_pascal_sweep
#SBATCH --output=../logs/pascal_sweep_%j.log
#SBATCH --error=../logs/pascal_sweep_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=u22
#SBATCH --account=research
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# pascal_sweep.slurm — Full GEMM benchmark sweep on Pascal (GTX 1080 Ti, sm_61)
#
# Submit with: sbatch pascal_sweep.slurm
# Monitor with: squeue -u $USER
# After completion: results in /home2/$USER/cuda_gemm/results/

set -euo pipefail

echo "========================================"
echo "CUDA GEMM Pascal Sweep — Job $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "========================================"

# ─── Environment ──────────────────────────────────────────────────────────
module load u18/cuda/11.6

# Confirm GPU (non-negotiable rule: verify what you got)
echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
    --format=csv,noheader
echo ""

# Work directory on local scratch
SCRATCH="/scratch/${USER}/cuda_gemm_pascal_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"
echo "Scratch dir: $SCRATCH"

# ─── Copy source to scratch ────────────────────────────────────────────────
SRC_DIR="/home2/${USER}/cuda_gemm"
cp -r "$SRC_DIR/kernels" "$SCRATCH/"
cp -r "$SRC_DIR/bench"   "$SCRATCH/"
cd "$SCRATCH/bench"

# ─── Build ────────────────────────────────────────────────────────────────
echo ""
echo "=== Building Pascal binary (sm_61) ==="
make pascal 2>&1 | tee "$SCRATCH/build_pascal.log"

# Save ptxas register/smem info (always works, no counters needed)
echo ""
echo "=== ptxas register/smem info ==="
grep "ptxas info" "$SCRATCH/bench/ptxas_sm61.log" | tee "$SCRATCH/ptxas_summary_sm61.txt"

# ─── §0.2 Profiling: nsys CUDA trace (what we CAN do) ─────────────────────
echo ""
echo "=== nsys CUDA API trace (§0.2 fallback — hardware counters blocked) ==="
NSYS_OUT="$SCRATCH/nsys_pascal_warptile_4096"
nsys profile \
    --trace=cuda \
    --stats=true \
    --output="$NSYS_OUT" \
    ./bench_runner_sm61 6 4096 4096 4096 5 20 1.0 0.0 \
    > "$SCRATCH/nsys_warptile_stdout.txt" 2>&1 || true
echo "nsys output: ${NSYS_OUT}.nsys-rep"
echo "nsys stats:  $SCRATCH/nsys_warptile_stdout.txt"

# Also run nsys on cuBLAS for comparison
NSYS_CUBLAS="$SCRATCH/nsys_pascal_cublas_4096"
nsys profile \
    --trace=cuda \
    --stats=true \
    --output="$NSYS_CUBLAS" \
    ./bench_runner_sm61 8 4096 4096 4096 5 20 1.0 0.0 \
    > "$SCRATCH/nsys_cublas_stdout.txt" 2>&1 || true

# ─── §2.1 + §2.3: Main benchmark sweep ────────────────────────────────────
echo ""
echo "=== Running main benchmark sweep ==="
bash run_sweep.sh "$SCRATCH/results"

# ─── §2.5: Parameter sensitivity sweep ────────────────────────────────────
echo ""
echo "=== Running parameter sensitivity sweep ==="
bash param_sensitivity.sh "$SCRATCH/param_results"

# ─── Copy all results to home ──────────────────────────────────────────────
echo ""
echo "=== Copying results to home directory ==="
HOME_DEST="/home2/${USER}/cuda_gemm/results"
mkdir -p "$HOME_DEST"

# Copy condensed results (CSV files, nsys text exports, ptxas log)
cp "$SCRATCH"/results/*.csv   "$HOME_DEST/" 2>/dev/null || true
cp "$SCRATCH"/param_results/*.csv "$HOME_DEST/" 2>/dev/null || true
cp "$SCRATCH/ptxas_summary_sm61.txt" "$HOME_DEST/"
cp "$SCRATCH/build_pascal.log"       "$HOME_DEST/"
cp "$SCRATCH/nsys_warptile_stdout.txt"  "$HOME_DEST/"
cp "$SCRATCH/nsys_cublas_stdout.txt"    "$HOME_DEST/"

# Export nsys report to text (if nsys2text is available)
if command -v nsys &> /dev/null; then
    nsys export --type=text "$NSYS_OUT".nsys-rep \
        > "$HOME_DEST/nsys_pascal_warptile_4096.txt" 2>/dev/null || true
fi

echo ""
echo "========================================"
echo "Pascal sweep complete — $(date)"
echo "Results in: $HOME_DEST"
echo "========================================"
