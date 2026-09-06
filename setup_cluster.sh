#!/usr/bin/env bash
# setup_cluster.sh — One-shot setup script to push the project to Ada cluster
# Run from your LOCAL MACHINE:
#   bash setup_cluster.sh <your_username>@ada.iiit.ac.in
#
# Or run directly on the cluster login node after copying the PAA directory:
#   bash setup_cluster.sh --local

set -euo pipefail

REMOTE="${1:-}"
PROJECT_NAME="cuda_gemm"
REMOTE_DIR="/home2/\${USER}/${PROJECT_NAME}"

if [ "$REMOTE" = "--local" ]; then
    echo "=== Setting up on local/cluster machine ==="
    DEST_DIR="/home2/$(whoami)/${PROJECT_NAME}"
    mkdir -p "$DEST_DIR/logs" "$DEST_DIR/results" "$DEST_DIR/plots"
    # Assumes this script is run from inside the PAA directory
    cp -r kernels bench slurm analysis "$DEST_DIR/"
    cp 00_env_check.sh "$DEST_DIR/"
    chmod +x "$DEST_DIR/bench/"*.sh
    chmod +x "$DEST_DIR/"*.sh
    echo "Setup complete in $DEST_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. cd $DEST_DIR"
    echo "  2. sinfo -a   # check available nodes"
    echo "  3. Get Pascal node: srun --pty --partition=long -A cvit --gres=gpu:1 --mem-per-cpu=2G -c 10 --nodelist=gnode10 bash -l"
    echo "  4. On the node: module load u18/cuda/11.6 && bash /home2/\$USER/cuda_gemm/00_env_check.sh"
    echo "  5. Build: cd /home2/\$USER/cuda_gemm/bench && make all"
    echo "  6. Submit jobs: sbatch slurm/pascal_sweep.slurm && sbatch slurm/turing_sweep.slurm"
    echo "  7. Monitor: squeue -u \$USER"
    echo "  8. After jobs finish: python3 analysis/plot_results.py --results-dir results --output-dir plots"
else
    echo "Usage: bash setup_cluster.sh --local"
    echo "       (run this script on the Ada login node)"
fi
