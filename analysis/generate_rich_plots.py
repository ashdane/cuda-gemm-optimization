import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('plots', exist_ok=True)

# ─── 1. Memory Traffic and Sector Count Reduction (Log Scale Bar Chart) ───
kernels_mem = ['K0 Naive', 'K1 Coalesced', 'K2 Smem Tiling', 'K4 2D Blocktile', 'cuBLAS SGEMM']
sectors = [167903232, 167903232, 8519680, 3128674, 131072]
traffic_gb = [s * 32 / (1024**3) for s in sectors]  # 32 bytes per sector in GB

fig, ax1 = plt.subplots(figsize=(8, 4.8))
colors = ['#C0392B', '#E67E22', '#F39C12', '#2980B9', '#27AE60']
bars = ax1.bar(kernels_mem, traffic_gb, color=colors, width=0.55, edgecolor='black', linewidth=0.8)

ax1.set_yscale('log')
ax1.set_ylabel('Global Memory Traffic Transferred (GB, Log Scale)', fontsize=11, fontweight='bold')
ax1.set_title('Global Memory Traffic Reduction per Kernel (1024x1024x1024)', fontsize=12, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.7, which='both')

# Add values above bars
for bar, gb, sec in zip(bars, traffic_gb, sectors):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.35, f"{gb:.2f} GB\n({sec/1e6:.1f}M sectors)",
             ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax1.set_ylim(0.001, 15)
plt.xticks(fontsize=9.5)
plt.tight_layout()
plt.savefig('plots/memory_traffic_reduction.png', dpi=300)
plt.close()

# ─── 2. Tile Sensitivity Heatmap: BM vs BN for TN=4 vs TN=8 (at BK=16) ───
bm_labels = ['32', '64', '128']
bn_labels = ['32', '64', '128']

# Measured GFLOPS from param_sweep_NVIDIA_GeForce_RTX_2080_Ti_20260907_153954.csv for BK=16, TM=8
# For TN=4:
grid_tn4 = np.array([
    [3014.4, 4237.7, 5856.6],  # BM=32, BN=32,64,128
    [3607.0, 7286.8, 7560.0],  # BM=64, BN=32,64,128
    [4120.5, 6890.2, 8084.9]   # BM=128, BN=32,64,128
])

# For TN=8:
grid_tn8 = np.array([
    [2787.4, 3444.7, 4428.0],  # BM=32, BN=32,64,128
    [2950.1, 3620.4, 4719.1],  # BM=64, BN=32,64,128
    [3293.9, 4105.6, 5210.3]   # BM=128, BN=32,64,128
])

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

im1 = ax_a.imshow(grid_tn4, cmap='YlGnBu', vmin=2500, vmax=8200)
ax_a.set_title('TN = 4 (Conflict-Free)', fontsize=11, fontweight='bold')
ax_a.set_xticks(range(3))
ax_a.set_xticklabels([f'BN={x}' for x in bn_labels])
ax_a.set_yticks(range(3))
ax_a.set_yticklabels([f'BM={y}' for y in bm_labels])
ax_a.set_xlabel('BN Dimension', fontsize=10)
ax_a.set_ylabel('BM Dimension', fontsize=10)

for i in range(3):
    for j in range(3):
        ax_a.text(j, i, f"{grid_tn4[i, j]:.0f}\nGFLOPS", ha='center', va='center',
                  color='white' if grid_tn4[i, j] > 5500 else 'black', fontsize=9, fontweight='bold')

im2 = ax_b.imshow(grid_tn8, cmap='YlGnBu', vmin=2500, vmax=8200)
ax_b.set_title('TN = 8 (2-Way Bank Conflict Cliff)', fontsize=11, fontweight='bold')
ax_b.set_xticks(range(3))
ax_b.set_xticklabels([f'BN={x}' for x in bn_labels])
ax_b.set_xlabel('BN Dimension', fontsize=10)

for i in range(3):
    for j in range(3):
        ax_b.text(j, i, f"{grid_tn8[i, j]:.0f}\nGFLOPS", ha='center', va='center',
                  color='white' if grid_tn8[i, j] > 5500 else 'black', fontsize=9, fontweight='bold')

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.18, 0.025, 0.65])
fig.colorbar(im2, cax=cbar_ax, label='Measured GFLOPS (RTX 2080 Ti)')

plt.suptitle('Parameter Sweep Heatmap: Impact of TN on Shared Memory Bank Conflicts (BK=16, TM=8)',
             fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 0.86, 0.94])
plt.savefig('plots/tile_sweep_heatmap.png', dpi=300)
plt.close()

print("Generated memory_traffic_reduction.png and tile_sweep_heatmap.png successfully.")
