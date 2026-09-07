# Benchmark Results Directory

## Canonical Datasets
- **`param_sweep_NVIDIA_GeForce_RTX_2080_Ti_20260907_153954.csv`**: **Canonical parameter sweep.** Contains the full 144-configuration empirical grid search over tile parameters ($BM, BN, BK, TM, TN$) on the Turing RTX 2080 Ti ($M=N=K=4096$, 50 iterations per point). Cited across the report and used to generate `plots/fig4_param_sensitivity.png`.
- **`results_gnode001_NVIDIA_GeForce_GTX_1080_Ti_20260906_082253.csv`**: Canonical kernel progression and dimension sweep benchmarks on Pascal (sm_61).
- **`results_gnode084_NVIDIA_GeForce_RTX_2080_Ti_20260906_083708.csv`**: Canonical kernel progression and dimension sweep benchmarks on Turing (sm_75).

## Audit Trail / Header-Only Files
- **`param_sweep_NVIDIA_GeForce_RTX_2080_Ti_20260906_084534.csv`**: Early aborted parameter sweep run on Turing (header-only, 0 data rows); retained solely for SLURM job execution audit trail.
- **`param_sweep_NVIDIA_GeForce_GTX_1080_Ti_20260906_082928.csv`**: Early aborted parameter sweep run on Pascal (header-only, 0 data rows); retained solely for SLURM job execution audit trail.
