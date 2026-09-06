import matplotlib.pyplot as plt
import os
import os

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

configs = ['BM=128, BK=16', 'BM=128, BK=32', 'BM=64, BK=16']
gflops = [3925.5, 2140.2, 3105.8]
occupancy = [25.0, 12.5, 25.0]

fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:blue'
ax1.set_xlabel('Tile Configuration (BM, BK)', fontsize=12)
ax1.set_ylabel('Measured GFLOPS', color=color, fontsize=12)
bars = ax1.bar(configs, gflops, color=color, width=0.4, label='GFLOPS')
ax1.tick_params(axis='y', labelcolor=color)

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 100, f"{yval:.1f}", ha='center', va='bottom', color=color, fontweight='bold')

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Calculated Occupancy (%)', color=color, fontsize=12)
ax2.set_ylim(0, 35)
line = ax2.plot(configs, occupancy, color=color, marker='o', linestyle='dashed', linewidth=2, markersize=8, label='Occupancy')
ax2.tick_params(axis='y', labelcolor=color)

for i, v in enumerate(occupancy):
    ax2.text(i + 0.1, v, f"{v}%", color=color, fontweight='bold', va='center')

plt.title('Parameter Sweep: The Shared Memory Cliff on Pascal', fontsize=14, pad=15)
fig.tight_layout()

plt.savefig('plots/param_sweep.png', dpi=300)
print("Saved plots/param_sweep.png")
