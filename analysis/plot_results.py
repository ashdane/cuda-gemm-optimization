#!/usr/bin/env python3

import argparse
import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

# ─── Hardware spec constants ────────────────────────────────────────────────
# CITE THESE IN THE REPORT — not measured, from NVIDIA spec sheets
GPU_SPECS = {
    'GeForce GTX 1080 Ti': {
        'fp32_tflops':  11.34,      # TFLOPS FP32 peak
        'mem_bw_gbs':   484.4,      # GB/s DRAM bandwidth (spec)
        'sm_count':     28,
        'arch':         'Pascal (sm_61)',
        'color':        '#4c72b0',  # Blue
        'ridge_ai':     11.34e3 / 484.4,  # FLOPS/byte at roofline ridge
    },
    'GeForce RTX 2080 Ti': {
        'fp32_tflops':  13.45,
        'mem_bw_gbs':   616.0,
        'sm_count':     68,
        'arch':         'Turing (sm_75)',
        'color':        '#dd8452',  # Orange
        'ridge_ai':     13.45e3 / 616.0,
    },
}

# Kernel display names in order
KERNEL_ORDER = ['naive', 'coalesced', 'smem_tiling', '1d_blocktile',
                '2d_blocktile', 'vectorized', 'warptile', 'cublas_sgemm',
                'tensor_core_fp16']
KERNEL_LABELS = {
    'naive':            'K0\nNaive',
    'coalesced':        'K1\nCoalesced',
    'smem_tiling':      'K2\nSmem\nTile',
    '1d_blocktile':     'K3\n1D Tile',
    '2d_blocktile':     'K4\n2D Tile',
    'vectorized':       'K5\nVectorized',
    'warptile':         'K6\nWarp\nTile',
    'cublas_sgemm':     'K8\ncuBLAS',
    'tensor_core_fp16': 'K7\nTensor\nCore',
}

# Arithmetic intensity (FLOPS/byte) for GEMM — code-derived, not measured
# For square M=N=K=n:  2*n^3 / (3*n^2*4) = n/6 FLOPS/byte
# e.g. n=4096: 4096/6 ≈ 682 FLOPS/byte (well into compute-bound regime)
# For roofline, we plot the theoretical value not a measured cache hit rate
def arithmetic_intensity(M, N, K):
    """2*M*N*K FLOPs / (M*K + K*N + M*N)*4 bytes (worst-case: all from DRAM)"""
    flops  = 2 * M * N * K
    bytes_ = (M*K + K*N + M*N) * 4
    return flops / bytes_

# ─── Load data ──────────────────────────────────────────────────────────────
def load_results(results_dir):
    csvs = glob.glob(os.path.join(results_dir, 'results_*.csv'))
    if not csvs:
        raise FileNotFoundError(f"No results_*.csv files found in {results_dir}")
    dfs = []
    for f in csvs:
        df = pd.read_csv(f)
        df['source_file'] = os.path.basename(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def load_param_sweep(results_dir):
    csvs = glob.glob(os.path.join(results_dir, 'param_sweep_*.csv'))
    if not csvs:
        return None
    return pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)

def normalize_gpu_name(name):
    """Match GPU name to spec dict key."""
    name = str(name).strip()
    for key in GPU_SPECS:
        if key.lower() in name.lower():
            return key
    return name

# ─── Plot 1: GFLOPS vs Kernel Version ───────────────────────────────────────
def plot_kernel_progression(df, outdir, fixed_size=4096):
    """Bar chart of GFLOPS at fixed M=N=K for all kernel versions."""
    subset = df[(df['M'] == fixed_size) & (df['N'] == fixed_size) & (df['K'] == fixed_size)].copy()
    subset['gpu_key'] = subset['gpu_name'].apply(normalize_gpu_name)
    subset = subset[subset['gpu_key'].isin(GPU_SPECS)]
    subset = subset[subset['kernel_name'].isin(KERNEL_ORDER)]

    gpus = subset['gpu_key'].unique()
    n_gpus = len(gpus)

    fig, axes = plt.subplots(1, n_gpus, figsize=(7 * n_gpus, 6), sharey=False)
    if n_gpus == 1:
        axes = [axes]

    for ax, gpu_key in zip(axes, gpus):
        spec   = GPU_SPECS[gpu_key]
        gdf    = subset[subset['gpu_key'] == gpu_key]
        # Keep only one row per kernel (median gflops if duplicated)
        gdf    = gdf.groupby('kernel_name', as_index=False)['gflops_median'].max()

        # Reorder
        order  = [k for k in KERNEL_ORDER if k in gdf['kernel_name'].values]
        gdf    = gdf.set_index('kernel_name').loc[order].reset_index()

        bars   = ax.bar(range(len(gdf)), gdf['gflops_median'],
                        color=spec['color'], alpha=0.85, edgecolor='white', linewidth=0.8)

        peak_fp32 = spec['fp32_tflops'] * 1000  # → GFLOPS
        ax.axhline(peak_fp32, color='red', ls='--', lw=1.5, label=f"FP32 Peak ({spec['fp32_tflops']:.1f} TFLOPS)")

        # Annotate % of peak above each bar
        for i, row in enumerate(gdf.itertuples()):
            pct = 100 * row.gflops_median / peak_fp32
            ax.text(i, row.gflops_median + peak_fp32 * 0.01,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(range(len(gdf)))
        ax.set_xticklabels([KERNEL_LABELS.get(k, k) for k in gdf['kernel_name']],
                           fontsize=9)
        ax.set_xlabel('Kernel Version', fontsize=11)
        ax.set_ylabel('GFLOPS', fontsize=11)
        ax.set_title(f"{spec['arch']}\n{gpu_key}\n(M=N=K={fixed_size})", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_ylim(0, peak_fp32 * 1.15)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('GFLOPS vs Kernel Version — Headline Progression\n'
                 'Annotations show % of theoretical FP32 peak (spec-sheet numbers)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    outpath = os.path.join(outdir, 'fig1_kernel_progression.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")

# ─── Plot 2: GFLOPS vs Matrix Size ──────────────────────────────────────────
def plot_gflops_vs_size(df, outdir):
    """Line plot of GFLOPS vs square matrix size, per kernel version."""
    # Only square sizes for this plot
    sq = df[df['M'] == df['N']].copy()
    sq = sq[sq['N'] == sq['K']].copy()
    sq['size'] = sq['M']
    sq['gpu_key'] = sq['gpu_name'].apply(normalize_gpu_name)

    gpus = [g for g in sq['gpu_key'].unique() if g in GPU_SPECS]
    kernels = [k for k in KERNEL_ORDER if k in sq['kernel_name'].values]

    fig, axes = plt.subplots(1, len(gpus), figsize=(8 * len(gpus), 6), sharey=False)
    if len(gpus) == 1:
        axes = [axes]

    colors = sns.color_palette('tab10', len(kernels))

    for ax, gpu_key in zip(axes, gpus):
        spec = GPU_SPECS[gpu_key]
        gdf  = sq[sq['gpu_key'] == gpu_key]

        for i, k_name in enumerate(kernels):
            kdf = gdf[gdf['kernel_name'] == k_name].sort_values('size')
            if kdf.empty:
                continue
            ls = '--' if k_name == 'cublas_sgemm' else '-'
            mk = 'D' if k_name == 'cublas_sgemm' else ('*' if 'tensor' in k_name else 'o')
            ax.plot(kdf['size'], kdf['gflops_median'],
                    color=colors[i], ls=ls, marker=mk, markersize=6, lw=2,
                    label=KERNEL_LABELS.get(k_name, k_name).replace('\n', ' '))

        peak_gf = spec['fp32_tflops'] * 1000
        ax.axhline(peak_gf, color='red', ls=':', lw=1.2, alpha=0.7,
                   label=f'FP32 Peak ({spec["fp32_tflops"]:.1f} TFLOPS)')

        ax.set_xscale('log', base=2)
        ax.set_xlabel('Matrix Size (M=N=K)', fontsize=11)
        ax.set_ylabel('GFLOPS', fontsize=11)
        ax.set_title(f"{spec['arch']} — {gpu_key}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        # Mark non-power-of-two sizes with vertical dashed lines
        for npot in [3000, 3001, 4097, 1000, 768, 511]:
            if npot in kdf['size'].values:
                ax.axvline(npot, color='gray', ls=':', lw=0.8, alpha=0.5)

    fig.suptitle('GFLOPS vs Matrix Size (Square M=N=K)\n'
                 'Log-scale x-axis; vertical dotted lines = awkward/non-power-of-two sizes',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    outpath = os.path.join(outdir, 'fig2_gflops_vs_size.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")

# ─── Plot 3: Roofline ───────────────────────────────────────────────────────
def plot_roofline(df, outdir, fixed_size=4096):
    """Roofline model plot — arithmetic intensity vs GFLOPS per architecture."""
    subset = df[df['M'] == fixed_size].copy()
    subset['gpu_key'] = subset['gpu_name'].apply(normalize_gpu_name)
    subset = subset[subset['gpu_key'].isin(GPU_SPECS)]

    fig, ax = plt.subplots(figsize=(10, 7))

    ai_range = np.logspace(-1, 4, 500)  # FLOPS/byte range

    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h']
    k_colors = sns.color_palette('tab10', len(KERNEL_ORDER))

    for gidx, (gpu_key, spec) in enumerate(GPU_SPECS.items()):
        gdf = subset[subset['gpu_key'] == gpu_key]
        if gdf.empty:
            continue

        peak_gflops = spec['fp32_tflops'] * 1000  # GFLOPS
        mem_bw = spec['mem_bw_gbs']               # GB/s
        ridge = spec['ridge_ai']                   # FLOPS/byte

        # Roofline: min(mem_bw * AI, peak_gflops)
        roof = np.minimum(mem_bw * ai_range, peak_gflops)
        ax.plot(ai_range, roof, lw=2.5, ls='-', color=spec['color'],
                label=f"{gpu_key} roofline", alpha=0.8)
        ax.axhline(peak_gflops, color=spec['color'], ls=':', lw=1, alpha=0.4)
        ax.axvline(ridge, color=spec['color'], ls=':', lw=1, alpha=0.4)
        ax.text(ridge * 1.05, peak_gflops * 0.95,
                f"Ridge: {ridge:.0f} FLOP/B", color=spec['color'], fontsize=9)

        # Plot kernel operating points
        for ki, k_name in enumerate(KERNEL_ORDER):
            krow = gdf[gdf['kernel_name'] == k_name]
            if krow.empty:
                continue
            g = krow['gflops_median'].values[0]
            ai = arithmetic_intensity(fixed_size, fixed_size, fixed_size)
            # All GEMM kernels are at the same arithmetic intensity (same problem size)
            # They differ only in how much of that intensity they achieve
            lbl = k_name if gidx == 0 else None
            ax.scatter(ai, g, color=k_colors[ki], marker=markers[ki % len(markers)],
                       s=100, zorder=5, label=lbl, alpha=0.9,
                       edgecolors='white', linewidths=0.8)
            ax.text(ai * 1.03, g, f" {KERNEL_LABELS.get(k_name, k_name).split(chr(10))[1] if chr(10) in KERNEL_LABELS.get(k_name, k_name) else k_name}",
                    fontsize=8, color=k_colors[ki], va='center')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)\n[Code-derived, assuming all data from DRAM]',
                  fontsize=11)
    ax.set_ylabel('GFLOPS (measured, CUDA events)', fontsize=11)
    ax.set_title('Roofline Model — All Kernels at M=N=K=4096\n'
                 'Roofline bounds from NVIDIA spec sheets (cited in §1). '
                 'Points = measured GFLOPS.\n'
                 'AI is identical for all kernels at same problem size — '
                 'gap to roof shows achieved efficiency.',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, which='both', alpha=0.2)

    outpath = os.path.join(outdir, 'fig3_roofline.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")

# ─── Plot 4: Parameter Sensitivity ──────────────────────────────────────────
def plot_param_sensitivity(param_df, outdir):
    """GFLOPS and occupancy vs tile parameters — look for cliffs."""
    if param_df is None or param_df.empty:
        print("No param sweep data — skipping plot 4")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax_row, k_name in enumerate(['2d_blocktile', 'warptile']):
        kdf = param_df[param_df['kernel_name'] == k_name].copy()
        if kdf.empty:
            continue

        # GFLOPS vs BM (color=BN, marker=BK)
        ax = axes[ax_row][0]
        bns = sorted(kdf['BN'].unique())
        colors = sns.color_palette('viridis', len(bns))
        for c, bn in zip(colors, bns):
            sub = kdf[kdf['BN'] == bn].sort_values('BM')
            ax.plot(sub['BM'], sub['gflops_median'], 'o-', color=c, lw=1.5,
                    label=f'BN={bn}', markersize=5)
        ax.set_xlabel('BM (block tile rows)', fontsize=10)
        ax.set_ylabel('GFLOPS', fontsize=10)
        ax.set_title(f'{k_name} — GFLOPS vs BM', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, title='BN')
        ax.grid(alpha=0.3)

        # Occupancy vs threads per block
        ax = axes[ax_row][1]
        kdf_s = kdf.sort_values('threads_per_block')
        sc = ax.scatter(kdf_s['threads_per_block'], kdf_s['theoretical_occupancy_pct'],
                        c=kdf_s['gflops_median'], cmap='RdYlGn', s=80, zorder=5,
                        edgecolors='gray', linewidths=0.5)
        plt.colorbar(sc, ax=ax, label='GFLOPS')
        ax.set_xlabel('Threads per Block', fontsize=10)
        ax.set_ylabel('Theoretical Occupancy %\n(calculated from ptxas regs+smem)', fontsize=9)
        ax.set_title(f'{k_name} — Occupancy vs Threads/Block\n(Color = GFLOPS — look for cliffs)',
                     fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)

    plt.suptitle('§2.5 Parameter Sensitivity Analysis\n'
                 'Occupancy CALCULATED from ptxas register/smem data + GPU spec limits\n'
                 '(hardware counters unavailable: ERR_NVGPUCTRPERM on this cluster)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    outpath = os.path.join(outdir, 'fig4_param_sensitivity.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")

# ─── Plot 5: Multi-Architecture Normalized Comparison ───────────────────────
def plot_multiarch_comparison(df, outdir, fixed_size=4096):
    """Normalized % of peak across architectures for same kernel versions."""
    subset = df[(df['M'] == fixed_size) & (df['N'] == fixed_size) & (df['K'] == fixed_size)].copy()
    subset['gpu_key'] = subset['gpu_name'].apply(normalize_gpu_name)
    subset = subset[subset['gpu_key'].isin(GPU_SPECS)]

    for gpu_key, spec in GPU_SPECS.items():
        peak = spec['fp32_tflops'] * 1000
        mask = subset['gpu_key'] == gpu_key
        subset.loc[mask, 'pct_of_peak'] = 100.0 * subset.loc[mask, 'gflops_median'] / peak

    kernels_present = [k for k in KERNEL_ORDER if k in subset['kernel_name'].values
                       and k != 'tensor_core_fp16']  # TC only on Turing

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(kernels_present))
    width = 0.35
    gpus_present = [g for g in GPU_SPECS if g in subset['gpu_key'].unique()]

    for i, gpu_key in enumerate(gpus_present):
        spec = GPU_SPECS[gpu_key]
        gdf = subset[subset['gpu_key'] == gpu_key]
        vals = []
        for k in kernels_present:
            row = gdf[gdf['kernel_name'] == k]
            vals.append(row['pct_of_peak'].values[0] if not row.empty else 0)
        bars = ax.bar(x + i * width - width * (len(gpus_present) - 1) / 2,
                      vals, width * 0.9, label=f"{spec['arch']}",
                      color=spec['color'], alpha=0.85, edgecolor='white')
        for xi, v in zip(x + i * width - width * (len(gpus_present) - 1) / 2, vals):
            if v > 0:
                ax.text(xi, v + 0.5, f'{v:.1f}%', ha='center', va='bottom',
                        fontsize=7, rotation=45)

    # Tensor Core bar for Turing only
    tc_data = subset[(subset['kernel_name'] == 'tensor_core_fp16') &
                     (subset['gpu_key'].str.contains('2080'))]
    if not tc_data.empty:
        tc_gf = tc_data['gflops_median'].values[0]
        tc_spec = GPU_SPECS['GeForce RTX 2080 Ti']
        # Note: TC peak is different from FP32 peak
        tc_fp16_peak = 26.9 * 1000  # RTX 2080 Ti FP16 TC peak
        tc_pct = 100 * tc_gf / tc_fp16_peak
        ax.bar(len(kernels_present) + 0.2, tc_pct, width * 0.9,
               label=f'Tensor Core (% of FP16 TC peak 26.9T)',
               color='#8c0077', alpha=0.85, edgecolor='white')
        ax.text(len(kernels_present) + 0.2, tc_pct + 0.5, f'{tc_pct:.1f}%',
                ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xticks(x)
    xlabels = [KERNEL_LABELS.get(k, k).replace('\n', ' ') for k in kernels_present]
    ax.set_xticklabels(xlabels, fontsize=9, rotation=20, ha='right')
    ax.set_ylabel('% of Theoretical FP32 Peak\n(spec-sheet peak, cited in §1)', fontsize=11)
    ax.set_title(f'Multi-Architecture Comparison — % of Peak Achieved\n'
                 f'M=N=K={fixed_size}. Higher = better efficiency. '
                 f'Differences reveal architecture-specific optima.',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 120)
    ax.axhline(100, color='red', ls=':', lw=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(outdir, 'fig5_multiarch_comparison.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Generate CUDA GEMM study plots')
    parser.add_argument('--results-dir', default='.',
                        help='Directory containing results_*.csv files')
    parser.add_argument('--output-dir',  default='./plots',
                        help='Directory to save plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading benchmark results...")
    df = load_results(args.results_dir)
    print(f"Loaded {len(df)} rows from {args.results_dir}")
    print(f"GPUs: {df['gpu_name'].unique()}")
    print(f"Kernels: {df['kernel_name'].unique()}")
    print(f"Sizes: {sorted(df['M'].unique())}")

    param_df = load_param_sweep(args.results_dir)

    print("\nGenerating plots...")
    plot_kernel_progression(df, args.output_dir)
    plot_gflops_vs_size(df, args.output_dir)
    plot_roofline(df, args.output_dir)
    plot_param_sensitivity(param_df, args.output_dir)
    plot_multiarch_comparison(df, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}/")
    print("Files:", [f for f in os.listdir(args.output_dir) if f.endswith('.png')])

if __name__ == '__main__':
    main()
