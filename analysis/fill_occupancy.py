import pandas as pd
df_p = pd.read_csv('results/occupancy_sm61.csv')
df_t = pd.read_csv('results/occupancy_sm75.csv')
for k in ['naive', 'smem_tiling', '2d_blocktile', 'vectorized', 'double_buffering']:
    if k == 'naive': kn = '_Z11sgemm_naive'
    if k == 'smem_tiling': kn = '_Z10sgemm_smem'
    if k == '2d_blocktile': kn = '_Z18sgemm_2d'
    if k == 'vectorized': kn = '_Z16sgemm_vec'
    if k == 'double_buffering': kn = '_Z23sgemm_double'
    rp = df_p[df_p['kernel_name'].str.startswith(kn)]
    rt = df_t[df_t['kernel_name'].str.startswith(kn)]
    if len(rp)>0 and len(rt)>0:
        rp = rp.iloc[0]; rt = rt.iloc[0]
        lp = str(rp['limiting_factor']).strip()
        lt = str(rt['limiting_factor']).strip()
        print(f"| {k} | {rp['registers_per_thread']} | {rp['smem_bytes']} | {rp['threads_per_block']} | {rp['blocks_per_sm']} | {rp['occupancy_pct']} | {rt['blocks_per_sm']} | {rt['occupancy_pct']} | Pascal: {lp}, Turing: {lt} |")
