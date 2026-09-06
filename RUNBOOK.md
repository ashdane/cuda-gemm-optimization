# CUDA GEMM Study — Complete Start-to-Finish Runbook

## Overview
Total time: ~10–12 hours (mostly waiting for SLURM jobs)
Your hands-on time: ~45 minutes across the day

---

## PHASE 0 — Upload the project to Ada (5 min, on your Windows machine)

```powershell
# From your Windows machine — the zip is already on your Desktop:
scp C:\Users\LENOVO\Desktop\cuda_gemm_study.zip ashish123@ada.iiit.ac.in:~/
```

---

## PHASE 1 — Unpack and set up on the Ada login node (5 min)

```bash
# SSH into Ada login node
ssh ashish123@ada.iiit.ac.in

# Unpack
cd ~
unzip cuda_gemm_study.zip -d cuda_gemm
cd cuda_gemm

# Create required directories
mkdir -p logs results plots

# Make all scripts executable
chmod +x bench/run_sweep.sh bench/param_sensitivity.sh 00_env_check.sh

# Load CUDA module (do this every time you log in)
module load u18/cuda/11.6
```

---

## PHASE 2 — Verify node availability (5 min, on login node)

```bash
# See which nodes are idle
sinfo -a | grep -E "idle|mix" | head -30

# Check your SLURM account (should show 'cvit')
sacctmgr show assoc user=$USER format=Account,QOS,DefaultQOS

# Find an idle Pascal node (gnode01-40) and an idle Turing node (gnode43-92)
# Example: if gnode15 and gnode55 are idle, note those down
sinfo -N -l | grep -E "gnode[0-9]+" | grep -v "down\|drain" | head -40
```

> If gnode10 is busy, pick any other idle node in gnode01–40 for Pascal.
> If gnode50 is busy, pick any other idle node in gnode43–92 for Turing.
> **Update the `--nodelist=` line in both SLURM scripts before submitting.**

---

## PHASE 3 — Edit SLURM scripts with the correct nodes (2 min)

```bash
# Update Pascal script with your chosen Pascal node (e.g. gnode15)
sed -i 's/--nodelist=gnode10/--nodelist=gnode15/' ~/cuda_gemm/slurm/pascal_sweep.slurm

# Update Turing script with your chosen Turing node (e.g. gnode55)
sed -i 's/--nodelist=gnode50/--nodelist=gnode55/' ~/cuda_gemm/slurm/turing_sweep.slurm

# Verify the changes look right
grep nodelist ~/cuda_gemm/slurm/*.slurm
```

---

## PHASE 4 — Quick environment check on EACH architecture (15 min)

Run this to confirm the GPU, CUDA version, and that nsys trace works.
Do it interactively (not sbatch) so you see output immediately.

### Pascal check:
```bash
# Get interactive shell on Pascal node (replace gnode15 with your node)
srun --pty --partition=long -A cvit --gres=gpu:1 \
     --mem-per-cpu=2G -c 4 --nodelist=gnode15 bash -l

# Inside the interactive shell:
module load u18/cuda/11.6
bash ~/cuda_gemm/00_env_check.sh 2>&1 | tee ~/cuda_gemm/logs/env_pascal.log

# Check the output — look for:
#   GPU name: GeForce GTX 1080 Ti
#   compute_cap: 6.1
#   sm_61 compile: OK
#   nsys found + trace test: OK
#   cuBLAS link: OK

# Exit when done
exit
```

### Turing check:
```bash
# Get interactive shell on Turing node (replace gnode55 with your node)
srun --pty --partition=long -A cvit --gres=gpu:1 \
     --mem-per-cpu=2G -c 4 --nodelist=gnode55 bash -l

module load u18/cuda/11.6
bash ~/cuda_gemm/00_env_check.sh 2>&1 | tee ~/cuda_gemm/logs/env_turing.log

# Check the output — look for:
#   GPU name: GeForce RTX 2080 Ti
#   compute_cap: 7.5
#   sm_75 compile: OK
#   exit

exit
```

---

## PHASE 5 — Submit both SLURM jobs (2 min)

```bash
# Go back to login node (after exiting the interactive sessions)
cd ~/cuda_gemm

# Submit Pascal sweep
sbatch slurm/pascal_sweep.slurm
# Note the job ID printed: "Submitted batch job XXXXXXX"

# Submit Turing sweep (runs simultaneously)
sbatch slurm/turing_sweep.slurm
# Note the job ID printed

# Monitor — both jobs should appear
squeue -u $USER

# Expected output:
#   JOBID  PARTITION  NAME                ST  TIME  NODES  NODELIST
#   12345  long       gemm_pascal_sweep   R   0:05  1      gnode15
#   12346  long       gemm_turing_sweep   R   0:08  1      gnode55
```

**Each job takes approximately 4–8 hours.** You do not need to stay connected.

---

## PHASE 6 — Wait (4–8 hours)

Come back periodically and check:

```bash
# Check job status
squeue -u $USER

# Watch live log output (Pascal job, replace JOBID with your number)
tail -f ~/cuda_gemm/logs/pascal_sweep_JOBID.log

# If a job finishes, check for errors:
tail -50 ~/cuda_gemm/logs/pascal_sweep_JOBID.log
tail -50 ~/cuda_gemm/logs/pascal_sweep_JOBID.err
```

Jobs are complete when they no longer appear in `squeue -u $USER`.

---

## PHASE 7 — Verify results were copied home (2 min)

```bash
# Check what results landed in ~/cuda_gemm/results/
ls -lh ~/cuda_gemm/results/

# You should see files like:
#   results_gnode15_GeForce_GTX_1080_Ti_20260905_143022.csv   ← Pascal timing
#   results_gnode55_GeForce_RTX_2080_Ti_20260905_143108.csv   ← Turing timing
#   param_sweep_GeForce_GTX_1080_Ti_20260905_XXXXXX.csv       ← Pascal param sweep
#   param_sweep_GeForce_RTX_2080_Ti_20260905_XXXXXX.csv       ← Turing param sweep
#   ptxas_summary_sm61.txt                                     ← Pascal register info
#   ptxas_summary_sm75.txt                                     ← Turing register info
#   build_pascal.log                                           ← build output
#   build_turing.log
#   nsys_k5_stdout.txt                                         ← nsys traces (Turing)
#   nsys_k6_stdout.txt
#   nsys_k7_stdout.txt
#   nsys_k8_stdout.txt

# Quick sanity check: does the Pascal CSV have data?
wc -l ~/cuda_gemm/results/results_*GTX*.csv
head -3 ~/cuda_gemm/results/results_*GTX*.csv
```

If results files are missing or empty, check the job log:
```bash
grep -i "error\|fail\|not found" ~/cuda_gemm/logs/pascal_sweep_JOBID.err
```

---

## PHASE 8 — Run analysis scripts (15 min, on login node)

### 8a. Compute occupancy from ptxas output
```bash
cd ~/cuda_gemm

python3 analysis/occupancy_calc.py \
    --ptxas-log results/ptxas_summary_sm61.txt \
    --arch sm_61 \
    --csv results/occupancy_sm61.csv

python3 analysis/occupancy_calc.py \
    --ptxas-log results/ptxas_summary_sm75.txt \
    --arch sm_75 \
    --csv results/occupancy_sm75.csv
```

### 8b. Run code-level hardware analysis (no GPU needed)
```bash
python3 analysis/hw_analysis.py \
    --mnk 4096,4096,4096 \
    --json results/hw_analysis_4096.json \
    --csv  results/coalescing_analysis.csv

# Also run for small size to understand overhead regime
python3 analysis/hw_analysis.py --mnk 256,256,256
```

### 8c. Parse nsys output (Turing only, cross-validates timing)
```bash
# For each nsys file produced by the Turing sweep:
for kid in 5 6 7 8; do
    if [ -f results/nsys_k${kid}_stdout.txt ]; then
        python3 analysis/nsys_parser.py \
            results/nsys_k${kid}_stdout.txt \
            --mnk 4096,4096,4096 \
            --csv results/nsys_kernel${kid}_timing.csv
    fi
done
```

### 8d. Install plotting dependencies (if not already installed)
```bash
pip install --user matplotlib seaborn pandas numpy
# or:
pip3 install --user matplotlib seaborn pandas numpy
```

### 8e. Generate all plots
```bash
python3 analysis/plot_results.py \
    --results-dir results \
    --output-dir  plots

# You should see:
#   Saved: plots/fig1_kernel_progression.png
#   Saved: plots/fig2_gflops_vs_size.png
#   Saved: plots/fig3_roofline.png
#   Saved: plots/fig4_param_sensitivity.png
#   Saved: plots/fig5_multiarch_comparison.png

ls -lh plots/
```

---

## PHASE 9 — Fill in the report (1–2 hours)

Open [`report/CUDA_GEMM_Study.md`](file:///c:/Users/LENOVO/Desktop/PAA/report/CUDA_GEMM_Study.md).

Every `[MEASURED]` placeholder maps to a specific CSV column.
Use this lookup table:

| Placeholder type | Where to get the number |
|---|---|
| `[MEASURED]` GFLOPS | `gflops_median` column in `results_*.csv`, filter by `kernel_name` and `M=N=K=4096` |
| `[MEASURED]` p10–p90 | `gflops_p10` and `gflops_p90` columns |
| `[MEASURED]` Eff. BW | `eff_bw_gbs` column |
| `[MEASURED]` ms | `ms_median` column |
| `[CALCULATED]` Occ % | `occupancy_pct` column in `occupancy_sm61.csv` / `occupancy_sm75.csv` |
| `[ptxas]` regs | From `ptxas_summary_sm61.txt` — grep for `Used N registers` |
| `[ptxas]` smem | From `ptxas_summary_sm61.txt` — grep for `bytes smem` |
| Param sweep data | `param_sweep_*.csv` — sort by `gflops_median` descending |
| GPU names/driver | `env_pascal.log`, `env_turing.log` |

**Quick extraction commands:**
```bash
# All kernels at 4096³ on Pascal, sorted by GFLOPS
awk -F',' 'NR==1 || ($3==4096 && $4==4096 && $5==4096)' \
    results/results_*GTX*.csv | column -t -s','

# Best parameter config on Pascal
sort -t',' -k8 -rn results/param_sweep_*GTX*.csv | head -5

# ptxas register counts
grep -A1 "Function properties" results/ptxas_summary_sm61.txt | \
    grep "Used" | head -20

# Cross-check nsys vs CUDA events (should agree within 1%)
echo "=== nsys vs CUDA events comparison ==="
for f in results/nsys_kernel*.csv; do
    echo "File: $f"
    cat "$f" | column -t -s','
done
```

---

## PHASE 10 — Copy everything to home for safekeeping (2 min)

```bash
# Scratch is purged after 7 days — results are already in ~/cuda_gemm/results
# but double-check nothing is missing from scratch before it expires

# Final directory check
du -sh ~/cuda_gemm/
ls ~/cuda_gemm/results/ | wc -l   # should be 15+ files
ls ~/cuda_gemm/plots/ | wc -l     # should be 5 files

# Check quota
quota -s | head -5
# If close to 25 GB limit, move plots/results to /share1:
# mv ~/cuda_gemm /share1/ashish123/cuda_gemm
```

---

## PHASE 11 — Download results to your Windows machine (5 min)

```powershell
# On your Windows machine:

# Download results CSV and plots
scp -r ashish123@ada.iiit.ac.in:~/cuda_gemm/results C:\Users\LENOVO\Desktop\PAA\
scp -r ashish123@ada.iiit.ac.in:~/cuda_gemm/plots   C:\Users\LENOVO\Desktop\PAA\
scp    ashish123@ada.iiit.ac.in:~/cuda_gemm/report/CUDA_GEMM_Study.md C:\Users\LENOVO\Desktop\PAA\report\
```

Then share the CSVs, ptxas logs, and nsys files back here and the agent will fill in all the `[MEASURED]` placeholders in the report automatically.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `sbatch` fails: "Invalid account" | Run `sacctmgr show assoc user=$USER` and use whatever account appears under `Account`. Edit the `-A cvit` line in the `.slurm` files. |
| Job stays in `PD` (pending) for >30 min | The chosen node may be busy. Run `sinfo -N \| grep gnode15` to check. Edit `--nodelist=` to a different idle node. |
| Binary not found in SLURM job | The job copies source from `/home2/$USER/cuda_gemm` to scratch and builds there. If the copy fails, check that `~/cuda_gemm/kernels/` and `~/cuda_gemm/bench/` exist. |
| `make` fails: "nvcc not found" | Add `module load u18/cuda/11.6` to your `~/.bashrc` or confirm it's in the SLURM script's module line. |
| Results CSV is empty | Check `~/cuda_gemm/logs/pascal_sweep_JOBID.err` for the actual error. Usually a build failure or wrong binary name. |
| `plot_results.py` fails: "No results_*.csv found" | Run it from `~/cuda_gemm/` and pass the full path: `--results-dir ~/cuda_gemm/results` |
| Turing node has sm_61 GPU (wrong architecture) | The SLURM script has a runtime check that will exit with "Wrong node!" — resubmit with a different `--nodelist`. |
| Quota exceeded | Move `~/cuda_gemm` to `/share1/$USER/cuda_gemm` and update paths in the scripts. |
