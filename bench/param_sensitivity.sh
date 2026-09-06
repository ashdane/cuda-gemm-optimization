#!/usr/bin/env bash
# param_sensitivity.sh — §2.5 Parameter sensitivity sweep
#
# Systematically varies BM, BN, BK, TM, TN for the 2D blocktile and
# warptile kernels to find performance cliffs and optimal configurations.
#
# This requires recompiling with different #define values.
# We use compile-time constants (not runtime parameters) for optimal codegen.
#
# Output: param_sweep_<arch>_<timestamp>.csv
#
# Note: hardware counter metrics (occupancy, etc.) are unavailable due to
# ERR_NVGPUCTRPERM on this cluster. Occupancy is CALCULATED from ptxas output.

set -euo pipefail

module load u18/cuda/11.6 2>/dev/null || true

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1 | tr ' ' '_')
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1)
SM_MAJOR=$(echo $COMPUTE_CAP | cut -d. -f1)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRATCH_DIR="/scratch/cuda_gemm_param"
mkdir -p "$SCRATCH_DIR"

OUTFILE="${SCRATCH_DIR}/param_sweep_${GPU_NAME}_${TIMESTAMP}.csv"

echo "kernel_name,BM,BN,BK,TM,TN,threads_per_block,gflops_median,gflops_stddev,regs_per_thread,smem_bytes,theoretical_occupancy_pct,M,N,K" \
    > "$OUTFILE"

# Fixed benchmark size for parameter sweep
M=4096; N=4096; K=4096
WARMUP=5; ITERS=50

# ─── Occupancy calculator ─────────────────────────────────────────────────
# GPU hardware limits (look these up from spec; also in occupancy_calc.py)
if [ "$SM_MAJOR" -ge 7 ]; then
    MAX_REGS_PER_SM=65536
    MAX_THREADS_PER_SM=1024
    MAX_BLOCKS_PER_SM=16
    SMEM_PER_SM=65536       # 64 KB configurable on Turing
    WARP_SIZE=32
else
    MAX_REGS_PER_SM=65536
    MAX_THREADS_PER_SM=2048
    MAX_BLOCKS_PER_SM=32
    SMEM_PER_SM=49152       # 48 KB on Pascal (fixed)
    WARP_SIZE=32
fi

calc_occupancy() {
    local threads_per_block=$1
    local regs_per_thread=$2
    local smem_bytes=$3
    python3 -c "
import math
max_regs_sm   = $MAX_REGS_PER_SM
max_threads_sm= $MAX_THREADS_PER_SM
max_blocks_sm = $MAX_BLOCKS_PER_SM
smem_sm       = $SMEM_PER_SM
warp_size     = $WARP_SIZE

tpb = $threads_per_block
regs= $regs_per_thread
smem= $smem_bytes

warps_per_block = math.ceil(tpb / warp_size)

# Limit 1: registers
reg_limit = max_regs_sm // max(1, regs * tpb) if regs > 0 else max_blocks_sm

# Limit 2: shared memory (round up to 256-byte boundary)
smem_alloc = math.ceil(smem / 256) * 256
smem_limit = max_blocks_sm if smem_alloc == 0 else smem_sm // smem_alloc

# Limit 3: thread count
thread_limit = max_threads_sm // tpb

# Limit 4: hardware block limit
blocks_per_sm = min(reg_limit, smem_limit, thread_limit, max_blocks_sm)
achieved_warps = blocks_per_sm * warps_per_block
max_warps_sm = max_threads_sm // warp_size
occupancy_pct = 100.0 * achieved_warps / max_warps_sm
print(f'{occupancy_pct:.1f}')
"
}

# ─── Compile-and-bench helper ──────────────────────────────────────────────
BENCH_DIR="$(dirname "$0")"
KERNEL_DIR="${BENCH_DIR}/../kernels"

compile_and_bench() {
    local kernel_name=$1
    local src_file=$2
    local BM=$3 BN=$4 BK=$5 TM=$6 TN=$7
    local threads_per_block=$(( (BM / TM) * (BN / TN) ))

    # Validate: threads must be ≤ 1024
    if [ "$threads_per_block" -gt 1024 ] || [ "$threads_per_block" -lt 32 ]; then
        return
    fi

    local bin="${SCRATCH_DIR}/bench_param_${BM}_${BN}_${BK}_${TM}_${TN}"

    # Compile with overridden defines + ptxas logging
    local ptxas_log="${SCRATCH_DIR}/ptxas_${BM}_${BN}_${BK}_${TM}_${TN}.log"

    if [ "$SM_MAJOR" -ge 7 ]; then
        ARCH_FLAG="-arch=sm_75"
    else
        ARCH_FLAG="-arch=sm_61"
    fi

    nvcc -O3 -std=c++14 \
        --ptxas-options=-v \
        "$ARCH_FLAG" \
        -DBM="$BM" -DBN="$BN" -DBK="$BK" -DTM="$TM" -DTN="$TN" \
        -DOVERRIDE_PARAMS \
        -I"$KERNEL_DIR" \
        -lcublas \
        -o "$bin" \
        "${BENCH_DIR}/bench_runner.cu" \
        "${KERNEL_DIR}/00_naive.cu" \
        "${KERNEL_DIR}/01_coalesced.cu" \
        "${KERNEL_DIR}/02_smem_tiling.cu" \
        "${KERNEL_DIR}/03_1d_blocktile.cu" \
        "${KERNEL_DIR}/04_2d_blocktile.cu" \
        "${KERNEL_DIR}/05_vectorized.cu" \
        "${KERNEL_DIR}/06_warptiling.cu" \
        2>"$ptxas_log" || {
            echo "  Compilation failed for BM=$BM BN=$BN BK=$BK TM=$TM TN=$TN"
            return
        }

    # Extract register count and smem from ptxas output
    local regs=$(grep -oP 'Used \K[0-9]+(?= registers)' "$ptxas_log" | tail -1)
    local smem=$(grep -oP 'Used .* bytes smem' "$ptxas_log" | grep -oP '[0-9]+(?= bytes smem)' | tail -1)
    regs=${regs:-0}; smem=${smem:-0}

    # Calculate occupancy
    local occ=$(calc_occupancy "$threads_per_block" "$regs" "$smem")

    # Which kernel id: 4=2d_blocktile, 6=warptile
    if [[ "$kernel_name" == "warptile" ]]; then
        local kid=6
    else
        local kid=4
    fi

    # Run benchmark
    local result=$("$bin" "$kid" "$M" "$N" "$K" "$WARMUP" "$ITERS" 1.0 0.0 2>/dev/null || echo "ERROR")
    if [[ "$result" == "ERROR" ]]; then return; fi

    local gf_med=$(echo "$result" | cut -d, -f6)
    local gf_std=$(echo "$result" | cut -d, -f7)

    echo "${kernel_name},${BM},${BN},${BK},${TM},${TN},${threads_per_block},${gf_med},${gf_std},${regs},${smem},${occ},${M},${N},${K}" \
        >> "$OUTFILE"
    echo "  ${kernel_name} BM=$BM BN=$BN BK=$BK TM=$TM TN=$TN → ${gf_med} GFLOPS, occ=${occ}% [regs=${regs}, smem=${smem}B]"

    rm -f "$bin" "$ptxas_log"
}

echo "=== §2.5 Parameter Sensitivity Sweep ==="
echo "GPU: $GPU_NAME, Fixed size: ${M}×${N}×${K}"
echo ""

# ─── 2D blocktile parameter grid ──────────────────────────────────────────
echo "--- 2D blocktile (kernel 4) ---"
for BM in 32 64 128; do
for BN in 32 64 128; do
for BK in 8 16 32; do
for TM in 4 8; do
for TN in 4 8; do
    compile_and_bench "2d_blocktile" "04_2d_blocktile.cu" $BM $BN $BK $TM $TN
done; done; done; done; done

echo ""
echo "--- Warptile (kernel 6) ---"
for BM in 64 128; do
for BN in 64 128; do
for BK in 8 16; do
for TM in 4 8; do
for TN in 4 8; do
    compile_and_bench "warptile" "06_warptiling.cu" $BM $BN $BK $TM $TN
done; done; done; done; done

echo ""
echo "=== Parameter sweep complete ==="
echo "Results: $OUTFILE"
HOME_RESULTS="/home2/${USER}/cuda_gemm/results"
mkdir -p "$HOME_RESULTS"
cp "$OUTFILE" "$HOME_RESULTS/"
echo "Copied to: $HOME_RESULTS/$(basename $OUTFILE)"
