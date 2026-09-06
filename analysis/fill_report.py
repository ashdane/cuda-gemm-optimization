import pandas as pd
import numpy as np
import re

pascal_csv = 'results/results_gnode001_NVIDIA_GeForce_GTX_1080_Ti_20260906_082253.csv'
turing_csv = 'results/results_gnode084_NVIDIA_GeForce_RTX_2080_Ti_20260906_083708.csv'

df_p = pd.read_csv(pascal_csv)
df_t = pd.read_csv(turing_csv)

def get_row(df, k, m, n=None, k_dim=None):
    if n is None: n = m
    if k_dim is None: k_dim = m
    res = df[(df['kernel_name'] == k) & (df['M'] == m) & (df['N'] == n) & (df['K'] == k_dim)]
    if len(res) == 0:
        return None
    return res.iloc[0]

kernel_names = [
    'naive', 'coalesced', 'smem_tiling', '1d_blocktile', '2d_blocktile',
    'vectorized', 'double_buffering', 'cublas_sgemm_fp32'
]

print("### 3.1 Pascal — GTX 1080 Ti (sm_61)")
print("| Kernel | GFLOPS *(median)* | p10–p90 | % of FP32 Peak (11,340 GFLOPS) | Eff. BW (GB/s) | vs. K0 speedup |")
print("|---|---|---|---|---|---|")
k0_p = get_row(df_p, 'naive', 4096)['gflops_median']
for k in kernel_names:
    r = get_row(df_p, k, 4096)
    if r is not None:
        print(f"| {k} | {r['gflops_median']:.1f} | {r['gflops_p10']:.1f}–{r['gflops_p90']:.1f} | {r['gflops_median']/11340*100:.1f}% | {r['eff_bw_gbs']:.1f} | {r['gflops_median']/k0_p:.1f}x |")

print("\n### 3.2 Turing — RTX 2080 Ti (sm_75)")
print("| Kernel | GFLOPS | p10–p90 | % of FP32 Peak (13,450 GFLOPS) | vs. K0 | vs. Pascal same kernel |")
print("|---|---|---|---|---|---|")
k0_t = get_row(df_t, 'naive', 4096)['gflops_median']
for k in kernel_names + ['tensor_core_wmma', 'cublas_fp16_tc']:
    rt = get_row(df_t, k, 4096)
    rp = get_row(df_p, k, 4096)
    if rt is not None:
        peak = 26900 if 'tc' in k or 'wmma' in k else 13450
        peak_str = f"{rt['gflops_median']/peak*100:.1f}%" + (" of TC peak" if peak == 26900 else "")
        vsp = f"{rt['gflops_median']/rp['gflops_median']:.1f}x" if rp is not None else "Turing only"
        print(f"| {k} | {rt['gflops_median']:.1f} | {rt['gflops_p10']:.1f}–{rt['gflops_p90']:.1f} | {peak_str} | {rt['gflops_median']/k0_t:.1f}x | {vsp} |")

print("\n### 4.1 Measured Performance Ratio")
for k in ['naive', 'smem_tiling', '2d_blocktile', 'vectorized', 'double_buffering', 'cublas_sgemm_fp32']:
    rt = get_row(df_t, k, 4096)
    rp = get_row(df_p, k, 4096)
    ratio = rt['gflops_median'] / rp['gflops_median']
    pp = rp['gflops_median'] / 11340 * 100
    pt = rt['gflops_median'] / 13450 * 100
    print(f"| {k} | {rp['gflops_median']:.1f} | {rt['gflops_median']:.1f} | {ratio:.2f} | {pp:.1f}% | {pt:.1f}% | {pt-pp:+.1f} pp |")

print("\n### 5.1 Square Power-of-Two Sizes (Turing)")
print("| Size | K5 (GFLOPS) | K6 (GFLOPS) | K7 (GFLOPS) | cuBLAS (GFLOPS) | K7/cuBLAS |")
print("|---|---|---|---|---|---|")
for sz in [256, 512, 1024, 2048, 4096, 8192]:
    k5 = get_row(df_t, 'vectorized', sz)
    k6 = get_row(df_t, 'warptile', sz)
    k7 = get_row(df_t, 'double_buffering', sz)
    cub = get_row(df_t, 'cublas_sgemm_fp32', sz)
    
    k5_val = f"{k5['gflops_median']:.1f}" if k5 is not None else "N/A"
    k6_val = f"{k6['gflops_median']:.1f}" if k6 is not None else "N/A"
    k7_val = f"{k7['gflops_median']:.1f}" if k7 is not None else "N/A"
    cub_val = f"{cub['gflops_median']:.1f}" if cub is not None else "N/A"
    ratio = f"{k7['gflops_median']/cub['gflops_median']*100:.1f}%" if (k7 is not None and cub is not None) else "N/A"
    
    print(f"| {sz}^3 | {k5_val} | {k6_val} | {k7_val} | {cub_val} | {ratio} |")

print("\n### 5.2/5.3 Non-Square / Awkward")
for dims in [(8192,512,512), (512,8192,512), (512,512,8192), (2048,512,4096), (1024,4096,2048), (256,256,16384),
             (3000,1500,1500), (3001,3001,3001), (4097,4097,4097), (1000,1000,1000), (768,768,768), (511,513,511)]:
    k7 = get_row(df_t, 'double_buffering', dims[0], dims[1], dims[2])
    cub = get_row(df_t, 'cublas_sgemm_fp32', dims[0], dims[1], dims[2])
    k7_val = f"{k7['gflops_median']:.1f}" if k7 is not None else "N/A"
    cub_val = f"{cub['gflops_median']:.1f}" if cub is not None else "N/A"
    ratio = f"{k7['gflops_median']/cub['gflops_median']*100:.1f}%" if (k7 is not None and cub is not None) else "N/A"
    print(f"| {dims[0]}x{dims[1]}x{dims[2]} | {k7_val} | {cub_val} | {ratio} |")
