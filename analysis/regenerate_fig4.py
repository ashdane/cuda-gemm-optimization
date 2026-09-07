import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('plots', exist_ok=True)

param_csv = 'results/param_sweep_NVIDIA_GeForce_RTX_2080_Ti_20260907_153954.csv'
df = pd.read_csv(param_csv)

fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

# ─── Row 0: 2D Blocktile ───
# Left: GFLOPS vs BM for 2D blocktile
kdf_2d = df[df['kernel_name'] == '2d_blocktile'].copy()
ax_2d_l = axes[0][0]
bns_2d = sorted(kdf_2d['BN'].unique())
colors_2d = sns.color_palette('viridis', len(bns_2d))
for c, bn in zip(colors_2d, bns_2d):
    sub = kdf_2d[kdf_2d['BN'] == bn].groupby('BM')['gflops_median'].mean().reset_index()
    ax_2d_l.plot(sub['BM'], sub['gflops_median'], 'o-', color=c, lw=1.8, label=f'BN={int(bn)}', markersize=6)
ax_2d_l.set_xlabel('BM (Block Tile Rows)', fontsize=10, fontweight='bold')
ax_2d_l.set_ylabel('Measured GFLOPS', fontsize=10, fontweight='bold')
ax_2d_l.set_title('2D Blocktile — GFLOPS vs BM Dimension', fontsize=11, fontweight='bold')
ax_2d_l.legend(title='BN Dimension', fontsize=8.5)
ax_2d_l.grid(True, linestyle='--', alpha=0.5)

# Right: Occupancy vs GFLOPS for all 6 Progression Kernels on Turing
ax_occ = axes[0][1]
# Table 7 data on Turing:
prog_kernels = [
    ('K0 Naive', 1024, 100.0, 1545.3),
    ('K2 Smem', 1024, 100.0, 2115.5),
    ('K3 1D Block', 512, 50.0, 4356.7),
    ('K4 2D Block', 256, 50.0, 6557.1),
    ('K5 Vectorized', 256, 50.0, 6856.2),
    ('K6 Warptile', 256, 100.0, 3515.8),
    ('K7 DoubleBuf', 256, 50.0, 5927.7),
]
p_names, p_threads, p_occ, p_gf = zip(*prog_kernels)
sc_occ = ax_occ.scatter(p_threads, p_occ, c=p_gf, cmap='RdYlGn', s=160, zorder=5, edgecolors='black', linewidths=1.2)
cbar_occ = plt.colorbar(sc_occ, ax=ax_occ, label='Measured GFLOPS (RTX 2080 Ti)')
for name, th, oc, gf in prog_kernels:
    offset_x = 25 if th < 800 else -180
    offset_y = -3 if name in ['K4 2D Block', 'K7 DoubleBuf'] else 2
    ax_occ.annotate(f"{name}\n({gf:.0f} GF)", (th, oc), xytext=(th + offset_x, oc + offset_y),
                    fontsize=8, fontweight='bold', arrowprops=dict(arrowstyle='->', lw=0.8, color='gray'))

ax_occ.set_xlabel('Threads per Block', fontsize=10, fontweight='bold')
ax_occ.set_ylabel('Theoretical Occupancy % (Turing sm_75)', fontsize=10, fontweight='bold')
ax_occ.set_title('Kernel Progression: Theoretical Occupancy vs Threads/Block\n(Table 7 Kernels; Color = Measured GFLOPS)', fontsize=11, fontweight='bold')
ax_occ.set_ylim(20, 115)
ax_occ.set_xlim(100, 1150)
ax_occ.grid(True, linestyle='--', alpha=0.5)

# ─── Row 1: Warptile ───
# Left: GFLOPS vs BM for warptile (valid configurations)
kdf_warp = df[df['kernel_name'] == 'warptile'].copy()
ax_w_l = axes[1][0]

# Valid points have threads_per_block <= 256 (where max_abs_err was verified < 0.01)
# Flag invalid points (threads_per_block >= 512, which had index clamp errors)
valid_warp = kdf_warp[kdf_warp['threads_per_block'] <= 256]
invalid_warp = kdf_warp[kdf_warp['threads_per_block'] > 256]

for bn in sorted(valid_warp['BN'].unique()):
    sub_v = valid_warp[valid_warp['BN'] == bn].sort_values('BM')
    ax_w_l.plot(sub_v['BM'], sub_v['gflops_median'], 's-', label=f'Valid (BN={int(bn)})', lw=1.8, markersize=6)

ax_w_l.set_xlabel('BM (Block Tile Rows)', fontsize=10, fontweight='bold')
ax_w_l.set_ylabel('Measured GFLOPS', fontsize=10, fontweight='bold')
ax_w_l.set_title('Warptile — GFLOPS vs BM Dimension (Verified Valid Configurations)', fontsize=11, fontweight='bold')
ax_w_l.legend(fontsize=8.5)
ax_w_l.grid(True, linestyle='--', alpha=0.5)

# Right: Warptile sweep scatter showing valid points vs flagged invalid overflow points
ax_w_r = axes[1][1]

# Valid scatter
sc_v = ax_w_r.scatter(valid_warp['threads_per_block'], valid_warp['gflops_median'],
                      c='#27AE60', s=90, marker='o', edgecolors='black', label='Valid (max_abs_err < 0.01)')
# Invalid scatter (flagged with red X)
sc_inv = ax_w_r.scatter(invalid_warp['threads_per_block'], invalid_warp['gflops_median'],
                        c='#C0392B', s=110, marker='x', linewidths=2.5, label='Invalid / Index Clamp Overflow (max_abs_err > 5.0)')

ax_w_r.axhline(13450, color='red', linestyle=':', lw=1.5, label='Turing FP32 Hardware Peak (13,450 GFLOPS)')
ax_w_r.set_xlabel('Threads per Block', fontsize=10, fontweight='bold')
ax_w_r.set_ylabel('Measured GFLOPS (param sweep CSV)', fontsize=10, fontweight='bold')
ax_w_r.set_title('Warptile Parameter Sweep: Valid Configurations vs Invalid Overflows\n(Red X = points with max_abs_err > 5.0 excluded from valid analysis)', fontsize=10.5, fontweight='bold')
ax_w_r.legend(fontsize=8, loc='upper left')
ax_w_r.grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Figure 3: Empirical Parameter Sensitivity & Occupancy Analysis (RTX 2080 Ti, sm_75)\n'
             'Tile sweeps from canonical grid; occupancy from Table 7 microarchitectural limits',
             fontsize=12, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('plots/fig4_param_sensitivity.png', dpi=300)
plt.savefig('report/fig4_param_sensitivity.png', dpi=300)
plt.close()

print("Regenerated fig4_param_sensitivity.png successfully with corrected K6/K7 and marked invalid overflow points.")
