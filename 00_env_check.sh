#!/usr/bin/env bash
# 00_env_check.sh — Environment discovery for CUDA GEMM study
# Run this FIRST on both a Pascal and Turing node (inside srun interactive session)
#
# Usage (inside srun/sinteractive):
#   module load u18/cuda/11.6
#   bash 00_env_check.sh 2>&1 | tee env_check_$(hostname -s).log
#
# This populates the "Test Environment" section of the final report.

set -euo pipefail

echo "========================================"
echo "CUDA GEMM — Environment Discovery"
echo "Date: $(date)"
echo "Host: $(hostname -f)"
echo "========================================"

echo ""
echo "=== 1. SLURM Allocation ==="
echo "SLURM_JOB_ID:   ${SLURM_JOB_ID:-not_in_slurm}"
echo "SLURM_NODELIST: ${SLURM_NODELIST:-none}"
echo "SLURM_GPUS:     ${SLURM_GPUS:-not_set}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"

echo ""
echo "=== 2. CUDA Module ==="
module load u18/cuda/11.6 2>&1 || {
    echo "WARNING: u18/cuda/11.6 failed, trying other versions..."
    for v in 10.2 10.0 9.2; do
        module load u18/cuda/$v 2>&1 && { echo "Loaded cuda/$v"; break; } || true
    done
}
module list 2>&1 | grep -i cuda || echo "No CUDA module detected"

echo ""
echo "=== 3. GPU Info (nvidia-smi) ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,uuid \
    --format=csv,noheader

echo ""
echo "=== 4. nvcc Version ==="
nvcc --version || echo "nvcc not found"

echo ""
echo "=== 5. Compilation Test ==="
cat > /tmp/test_gemm.cu << 'EOF'
#include <stdio.h>
__global__ void hello(float* out) { *out = 42.0f; }
int main() {
    float *d; cudaMalloc(&d, 4);
    hello<<<1,1>>>(d); cudaFree(d);
    printf("CUDA OK\n");
    return 0;
}
EOF

# Test sm_61
if nvcc -arch=sm_61 /tmp/test_gemm.cu -o /tmp/test_sm61 2>/dev/null; then
    echo "sm_61 compile: OK"
    /tmp/test_sm61 2>/dev/null || echo "sm_61 run: FAILED (wrong GPU?)"
else
    echo "sm_61 compile: FAILED"
fi

# Test sm_75
if nvcc -arch=sm_75 /tmp/test_gemm.cu -o /tmp/test_sm75 2>/dev/null; then
    echo "sm_75 compile: OK"
    /tmp/test_sm75 2>/dev/null || echo "sm_75 run: FAILED (wrong GPU?)"
else
    echo "sm_75 compile: FAILED"
fi

echo ""
echo "=== 6. Profiler Availability ==="

# Check ncu (Nsight Compute)
NCU_PATH=$(find $(dirname $(which nvcc))/.. -iname 'ncu' -type f 2>/dev/null | head -1)
if [ -n "$NCU_PATH" ]; then
    echo "ncu found: $NCU_PATH"
    echo "Testing ncu counter access..."
    "$NCU_PATH" --metrics sm__cycles_elapsed.avg /tmp/test_sm75 2>&1 | head -5
else
    echo "ncu: NOT FOUND in CUDA toolkit directory"
fi

# Check nsys (Nsight Systems)
if command -v nsys &>/dev/null; then
    echo ""
    echo "nsys found: $(which nsys)"
    echo "nsys version: $(nsys --version 2>&1 | head -1)"
    echo "Testing nsys CUDA trace (should work per cluster testing)..."
    nsys profile --trace=cuda --stats=true /tmp/test_sm75 2>&1 | tail -20
else
    echo "nsys: NOT FOUND"
fi

# Check nvprof (legacy, bundled with older CUDA)
if command -v nvprof &>/dev/null; then
    echo ""
    echo "nvprof found: $(which nvprof)"
    nvprof --version 2>&1 | head -1
else
    echo "nvprof: NOT FOUND (may need older cuda module)"
fi

echo ""
echo "=== 7. ptxas Test (--ptxas-options=-v) ==="
echo "This always works — no hardware access needed:"
nvcc -arch=sm_61 --ptxas-options=-v /tmp/test_gemm.cu -o /tmp/test_ptxas 2>&1 | \
    grep -E "ptxas info|Used" || echo "No ptxas output (trivial kernel uses 0 regs)"

echo ""
echo "=== 8. cuBLAS Availability ==="
cat > /tmp/test_cublas.cu << 'EOF'
#include <cublas_v2.h>
#include <stdio.h>
int main() {
    cublasHandle_t h;
    cublasCreate(&h);
    printf("cuBLAS OK\n");
    cublasDestroy(h);
    return 0;
}
EOF
if nvcc -arch=sm_61 /tmp/test_cublas.cu -lcublas -o /tmp/test_cublas 2>/dev/null; then
    /tmp/test_cublas && echo "cuBLAS link: OK" || echo "cuBLAS link: compile OK but run failed"
else
    echo "cuBLAS link: FAILED"
fi

echo ""
echo "=== 9. SLURM Account ==="
sacctmgr show assoc user=$USER format=Account,QOS,DefaultQOS 2>/dev/null || echo "sacctmgr not available"

echo ""
echo "=== 10. CPU Info ==="
lscpu | grep -E "Model name|Socket|Thread|Core|NUMA" || true

echo ""
echo "=== 11. Storage ==="
df -h /scratch 2>/dev/null || echo "/scratch not available on this node"
df -h /ssd_scratch 2>/dev/null || echo "/ssd_scratch not available"
df -h /home2/$USER 2>/dev/null || true

echo ""
echo "=== 12. Node range check ==="
NODE=$(hostname -s)
NODE_NUM=$(echo $NODE | grep -oP '\d+$')
if [ -n "$NODE_NUM" ]; then
    if [ "$NODE_NUM" -ge 1 ] && [ "$NODE_NUM" -le 40 ]; then
        echo "Node $NODE is in Pascal range (gnode01-40) → expected sm_61"
    elif [ "$NODE_NUM" -ge 43 ] && [ "$NODE_NUM" -le 92 ]; then
        echo "Node $NODE is in Turing range (gnode43-92) → expected sm_75"
    else
        echo "Node $NODE is outside known GPU ranges — verify manually"
    fi
fi

echo ""
echo "========================================"
echo "Environment check complete — $(date)"
echo "Save this log: tee env_check_$(hostname -s).log"
echo "========================================"
