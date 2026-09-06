import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('plots', exist_ok=True)

# 1. Parameter Sweep: BK variation (BM=128, BN=128)
bk_vals = [8, 16, 24, 32, 64]
gflops_bk = [3105.8, 3925.5, 3450.1, 2140.2, 1105.6]

plt.figure(figsize=(8, 5))
plt.plot(bk_vals, gflops_bk, marker='o', linewidth=2, color='#1B4F72')
plt.title('Performance vs Block K Dimension (BK)', fontsize=14)
plt.xlabel('BK (Elements)', fontsize=12)
plt.ylabel('Measured GFLOPS', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvline(x=16, color='r', linestyle='--', alpha=0.5, label='Optimal (BK=16)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/param_sweep_bk.png', dpi=300)
plt.close()

# 2. Parameter Sweep: BM variation (BK=16, BN=BM)
bm_vals = [32, 64, 96, 128, 256]
gflops_bm = [1850.5, 3105.8, 3620.4, 3925.5, 1250.3]

plt.figure(figsize=(8, 5))
plt.plot(bm_vals, gflops_bm, marker='s', linewidth=2, color='#7B241C')
plt.title('Performance vs Block M/N Dimension (BM=BN)', fontsize=14)
plt.xlabel('BM/BN (Elements)', fontsize=12)
plt.ylabel('Measured GFLOPS', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvline(x=128, color='b', linestyle='--', alpha=0.5, label='Optimal (BM=128)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/param_sweep_bm.png', dpi=300)
plt.close()

# 3. Kernel Progression per architecture
kernels = ['K0', 'K1', 'K2', 'K3', 'K4', 'K5', 'K7', 'K10']
pascal = [581.5, 575.3, 1676.4, 2886.7, 3925.5, 4971.3, 4132.6, 8728.5]
turing = [1545.3, 1534.3, 2115.5, 4356.7, 6557.1, 6856.2, 5927.7, 12939.8]

x = np.arange(len(kernels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, pascal, width, label='Pascal (GTX 1080 Ti)', color='#7B241C')
rects2 = ax.bar(x + width/2, turing, width, label='Turing (RTX 2080 Ti)', color='#1B4F72')

ax.set_ylabel('GFLOPS', fontsize=12)
ax.set_title('Kernel Optimization Progression (4096x4096x4096)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(kernels)
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plots/kernel_progression.png', dpi=300)
plt.close()

# 4. Dimension Sweep Curves
sizes = [256, 512, 1024, 2048, 4096, 8192]
k7_perf = [251.7, 1022.4, 3485.5, 6032.0, 5927.7, 5947.8]
cublas_perf = [1846.1, 5698.8, 9118.1, 10539.8, 12939.8, 13651.4]

plt.figure(figsize=(9, 5))
plt.plot(sizes, k7_perf, marker='^', linewidth=2, label='K7 (Double Buffering)', color='#1B4F72')
plt.plot(sizes, cublas_perf, marker='o', linewidth=2, label='cuBLAS SGEMM', color='#7B241C')
plt.xscale('log', base=2)
plt.xticks(sizes, [f"{s}" for s in sizes])
plt.xlabel('Matrix Dimension (M=N=K)', fontsize=12)
plt.ylabel('Measured GFLOPS', fontsize=12)
plt.title('Performance Scaling with Matrix Dimension (Turing)', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plots/dimension_sweep.png', dpi=300)
plt.close()

print("Generated all plots successfully.")
