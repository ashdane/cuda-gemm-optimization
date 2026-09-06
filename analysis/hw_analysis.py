#!/usr/bin/env python3
"""
hw_analysis.py — Code-derived hardware analysis for all GEMM kernel versions.

Since ncu hardware counters are BLOCKED on the Ada cluster (ERR_NVGPUCTRPERM),
this script performs static analysis of each kernel's memory access patterns
to derive:

  1. Coalescing efficiency: fraction of memory transactions that are coalesced
     (consecutive warp threads accessing consecutive addresses)
  2. Shared memory bank conflicts: whether the smem access stride causes conflicts
  3. Global memory traffic: bytes read/written at the problem level
  4. Arithmetic intensity: FLOP/byte at the kernel level (code-derived)
  5. Theoretical occupancy limiting factor: from ptxas data (see occupancy_calc.py)

All outputs are labeled [CODE-DERIVED] or [CALCULATED] in the report.
No GPU hardware is required to run this script.

Hardware assumptions (from NVIDIA documentation, cited):
  Warp size: 32 threads
  L1/Shared memory bank width: 4 bytes (32 banks × 4 bytes = 128 bytes/cycle)
  Global memory transaction granularity: 32 bytes (L1 cache line on Pascal/Turing)
  For 128-bit (float4) loads: 4 floats = 16 bytes → 1 transaction per warp quarter
"""

import math
import argparse
import json
import csv
import sys

WARP_SIZE    = 32
FLOAT_BYTES  = 4
SMEM_BANKS   = 32       # Both Pascal (sm_61) and Turing (sm_75)
BANK_WIDTH   = 4        # bytes per bank (32-bit banks)
CACHE_LINE   = 128      # bytes, L1/L2 cache line (Pascal/Turing)
SECTOR_SIZE  = 32       # bytes, minimum global memory transaction (sm_61+)

def coalescing_efficiency(stride_floats: int, access_width_floats: int = 1) -> dict:
    """
    Compute coalescing efficiency for a warp accessing floats with given stride.

    stride_floats: stride between consecutive thread indices (in floats)
    access_width_floats: number of floats per thread (1=scalar, 4=float4)

    Returns: transactions per warp, ideal transactions, efficiency %
    """
    bytes_per_access = access_width_floats * FLOAT_BYTES
    warp_span_bytes  = (WARP_SIZE - 1) * stride_floats * FLOAT_BYTES + bytes_per_access
    # Number of 32-byte sectors touched
    # Start address assumed aligned to sector boundary
    sectors_touched   = math.ceil(warp_span_bytes / SECTOR_SIZE)
    # Ideal: 1 sector per 32/access_width_floats threads  (fully coalesced)
    ideal_sectors     = math.ceil(WARP_SIZE * bytes_per_access / SECTOR_SIZE)
    efficiency        = 100.0 * ideal_sectors / max(sectors_touched, ideal_sectors)
    return {
        'stride_floats':       stride_floats,
        'access_width_floats': access_width_floats,
        'sectors_per_warp':    sectors_touched,
        'ideal_sectors':       ideal_sectors,
        'coalescing_eff_pct':  min(efficiency, 100.0),
    }

def smem_bank_conflicts(row_width_floats: int, access_col_stride: int) -> dict:
    """
    Estimate shared memory bank conflicts for a row-major smem access pattern.

    row_width_floats: number of floats in one smem row (e.g. BK or BN)
    access_col_stride: stride in floats between successive thread accesses
                       (1 = consecutive, row_width = column access)

    Bank for element at float index i: bank = (i % SMEM_BANKS)
    Conflict when multiple threads in a warp access the same bank
    (but different addresses — broadcast to same address is NOT a conflict).

    Returns: conflict multiplier (1=none, 2=2-way, 32=catastrophic)
    """
    # Simulate: which bank does thread t access?
    banks = [(t * access_col_stride) % SMEM_BANKS for t in range(WARP_SIZE)]
    from collections import Counter
    bank_counts = Counter(banks)
    max_conflict = max(bank_counts.values())
    n_conflicted_banks = sum(1 for v in bank_counts.values() if v > 1)
    return {
        'row_width_floats':    row_width_floats,
        'access_col_stride':   access_col_stride,
        'max_conflict_way':    max_conflict,
        'n_conflicted_banks':  n_conflicted_banks,
        'has_conflicts':       max_conflict > 1,
        'description': 'broadcast' if len(bank_counts) == 1 else
                        ('conflict-free' if max_conflict == 1 else f'{max_conflict}-way conflict'),
    }

def arithmetic_intensity(M: int, N: int, K: int) -> float:
    """
    Code-derived arithmetic intensity for GEMM assuming ALL data from DRAM.
    (Lower bound on real AI since caching increases it; upper bound on BW pressure.)
    """
    flops = 2.0 * M * N * K
    # Bytes: read A(M×K) + B(K×N) + read/write C(M×N)
    bytes_ = FLOAT_BYTES * (M*K + K*N + 2*M*N)
    return flops / bytes_

# ─── Per-kernel analysis ──────────────────────────────────────────────────
KERNEL_ANALYSES = {
    # ── K0: Naive ──────────────────────────────────────────────────────────
    'naive': {
        'description': '1 thread per output C[row,col]. A accessed row-wise (stride=K), B column-wise (stride=1 for col, but different K each iter).',
        'global_A_access': {
            'pattern': 'row broadcast',
            'stride_floats': 1,  # consecutive threads read same row → broadcast
            'note': 'All 32 threads in warp have same row → same address → 1 transaction (broadcast). Efficient!',
        },
        'global_B_access': {
            'pattern': 'column stride',
            'stride_floats': 0,  # col = blockIdx.x*32 + threadIdx.x → stride 1 in N dim
            'note': 'In the naive kernel, consecutive threads access consecutive COLS of B for each k → coalesced. BUT k iterates for each thread → N different k values across time.',
            # Actually in the naive kernel: thread reads B[k, col] where col = bx*32+tx
            # consecutive tx → consecutive col → COALESCED ✓
        },
        'global_C_access': {'stride_floats': 1, 'note': 'Consecutive cols → coalesced write'},
        'smem_usage': None,
        'coalescing_global_B': coalescing_efficiency(1, 1),
        'smem_bank_conflicts': None,
        'key_problem': 'No data reuse: A[row,k] and B[k,col] are loaded for EVERY output element. For K=4096: 4096 global loads per thread per output. O(K) global mem bandwidth per output element.',
    },

    # ── K1: Coalesced ──────────────────────────────────────────────────────
    'coalesced': {
        'description': 'Fixes thread indexing: col=threadIdx.x (fast dim) → B and C accesses are coalesced.',
        'note': 'In naive, if col=threadIdx.y*32+threadIdx.x, B access may be uncoalesced. Coalesced fixes warp-level address ordering. Same arithmetic as naive.',
        'coalescing_global_B': coalescing_efficiency(1, 1),  # stride-1 → 100%
        'smem_bank_conflicts': None,
        'key_improvement': 'Near-100% global memory transaction efficiency. Still O(K) global loads per output — bandwidth bound but now at theoretical peak bandwidth.',
    },

    # ── K2: Shared Memory Tiling ───────────────────────────────────────────
    'smem_tiling': {
        'description': 'TILE_SIZE×TILE_SIZE tiles of A and B loaded into shared memory. Each element reused TILE_SIZE times.',
        'tile_size': 32,
        'smem_bytes': 32*32*4 * 2,  # 2 tiles of float32
        # Load A into smem: thread (ty,tx) loads A[row, t*TILE+tx]
        # → consecutive tx → consecutive K column → coalesced ✓
        'load_A_coalescing': coalescing_efficiency(1, 1),
        # Load B into smem: thread (ty,tx) loads B[t*TILE+ty, col]
        # → consecutive tx → consecutive N column → coalesced ✓
        'load_B_coalescing': coalescing_efficiency(1, 1),
        # Compute from smem: tileA[ty][k] → same ty for all warp threads
        # → broadcast, no conflict.
        # tileB[k][tx] → consecutive tx → consecutive smem addr → banks 0..31 → no conflict ✓
        'smem_A_compute': smem_bank_conflicts(32, 32),  # tileA[ty][k]: stride=TILE along k → different row per thread? No. All threads access same ty → broadcast
        'smem_B_compute': smem_bank_conflicts(32, 1),   # tileB[k][tx]: stride=1 → banks 0..31 → conflict-free
        'global_load_reduction': 'TILE_SIZE× reduction in global loads (each element loaded once, reused TILE_SIZE times)',
        'arithmetic_intensity_multiplier': 32,  # vs naive: ~32× better AI
    },

    # ── K3: 1D Block Tiling ────────────────────────────────────────────────
    '1d_blocktile': {
        'description': 'Each thread computes TM=8 output rows. BM=64, BN=64, BK=8.',
        'BM': 64, 'BN': 64, 'BK': 8, 'TM': 8,
        'threads_per_block': 512,
        'smem_bytes': (64*8 + 8*64)*4,
        'load_coalescing': coalescing_efficiency(1, 1),
        # Compute: As[(threadRow*TM+resIdx)*BK + dotIdx]
        # threadRow = threadIdx.x / BN; all threads with same threadRow → different resIdx
        # → different rows of As → different banks (no conflict for row-major layout)
        'smem_A_compute': smem_bank_conflicts(8, 8),   # stride = BK = 8
        'smem_B_compute': smem_bank_conflicts(64, 1),   # stride = 1 along BN
        'key_improvement': 'TM=8 FMAs per register load from smem. Better ILP. Reduces sync barrier overhead per unit output.',
    },

    # ── K4: 2D Block Tiling ────────────────────────────────────────────────
    '2d_blocktile': {
        'description': 'Each thread computes TM×TN=8×8=64 outputs. BM=BN=128, BK=8.',
        'BM': 128, 'BN': 128, 'BK': 8, 'TM': 8, 'TN': 8,
        'threads_per_block': 256,
        'smem_bytes': (128*8 + 8*128)*4,
        'load_coalescing': coalescing_efficiency(1, 1),
        # regA[i] = As[(threadRow*TM+i)*BK + dotIdx]: stride = BK = 8
        'smem_A_compute': smem_bank_conflicts(8, 8),
        # regB[j] = Bs[dotIdx*BN + threadCol*TN + j]: stride = 1
        'smem_B_compute': smem_bank_conflicts(128, 1),
        'fma_per_load_ratio': 8*8,  # 64 FMAs from TM+TN=16 register loads per BK step
        'key_improvement': '64 FMAs from 16 register loads per BK step. Approaches roofline compute ceiling.',
    },

    # ── K5: Vectorized ─────────────────────────────────────────────────────
    'vectorized': {
        'description': 'float4 global loads. BM=BN=128, BK=16. 4 floats per load instruction.',
        'BM': 128, 'BN': 128, 'BK': 16,
        'threads_per_block': 256,
        'smem_bytes': (128*16 + 16*128)*4,
        # float4 load: 128-bit transaction. At 16-byte aligned address:
        # 1 float4 = 1 cache sector. 4× fewer load instructions than scalar.
        'global_load_float4': coalescing_efficiency(1, 4),  # 4 floats per thread, stride 1
        'instruction_count_reduction': '4× fewer global load instructions vs scalar',
        'smem_A_compute': smem_bank_conflicts(16, 16),   # BK=16, dotIdx stride
        'smem_B_compute': smem_bank_conflicts(128, 1),
        'key_improvement': 'Reduces L/S unit pressure. Better utilization of 128-byte cache lines. Critical on Turing (wider memory bus).',
        'bank_conflict_risk': 'BK=16 for A smem: stride=16. Bank = (col % 32). Threads accessing As[(threadRow*TM+i)*16 + dotIdx] — consecutive threads differ in threadRow → different smem rows → NO conflict. Safe.',
    },

    # ── K6: Warp Tiling ────────────────────────────────────────────────────
    'warptile': {
        'description': 'Warp-level decomposition: each warp owns WM×WN sub-region. BM=BN=128, BK=16, WM=64, WN=32.',
        'BM': 128, 'BN': 128, 'BK': 16, 'WM': 64, 'WN': 32,
        'TM': 8, 'TN': 4,
        'threads_per_block': 256,
        'warps_per_block': 8,
        'smem_bytes': (128*16 + 16*128)*4,
        'key_improvement': 'Warp-level locality reduces inter-warp smem contention. All 32 threads in warp access a contiguous WM×BK region of As. Better L1 hit rate for smem re-reads.',
        'turing_hypothesis': 'Turing has independent concurrent FP32+INT32 pipelines. Warp tiling generates denser interleaved INT (index) + FP (FMA) instruction streams vs earlier kernels. Predicted 2-5% Turing-specific uplift over Pascal at this kernel version.',
        'smem_A_access': 'warpRow*WM + threadRowInWarp*TM + i: contiguous rows within warp → good spatial locality within warp',
        'smem_B_access': 'warpCol*WN + threadColInWarp*TN + j: contiguous cols → stride-1 → conflict-free',
    },

    # ── K7: Double Buffering ───────────────────────────────────────────────
    'double_buffering': {
        'description': 'Software-pipelined double buffer. Prefetches next tile into registers during compute phase of current tile.',
        'smem_bytes': 2 * (128*16 + 16*128)*4,  # 2× for double buffer
        'key_mechanism': 'Load tile i+1 into registers while FMA units compute from smem tile i. Hides global memory latency behind arithmetic.',
        'expected_gain': '5-15% at large sizes where global memory latency is binding. Less gain where L2 already absorbs most traffic.',
        'smem_pressure': '32 KB for double buffer vs 16 KB single buffer. More shared memory per block → potentially fewer blocks per SM → occupancy tradeoff.',
        'pascal_vs_turing': 'On Pascal (sm_61): uses register-staging prefetch (load to regs during compute). On Turing (sm_75): could use cp.async for true async DMA (sm_80 required for full pipeline barriers; sm_75 gets partial benefit).',
    },
}

# ─── Roofline calculation ─────────────────────────────────────────────────
GPU_SPECS = {
    'GTX_1080_Ti': {
        'fp32_peak_tflops': 11.34,   # Source: nvidia.com spec sheet
        'mem_bw_gbs':       484.4,   # Source: nvidia.com spec sheet
        'smem_bw_tbs':      96.0,    # TB/s, estimated from SM count × bandwidth
        'arch': 'Pascal sm_61',
    },
    'RTX_2080_Ti': {
        'fp32_peak_tflops': 13.45,   # Source: nvidia.com spec sheet
        'mem_bw_gbs':       616.0,   # Source: nvidia.com spec sheet
        'fp16_tc_tflops':   26.9,    # FP16 Tensor Core peak
        'int8_tc_tops':     53.8,    # INT8 Tensor Core peak
        'smem_bw_tbs':      121.0,   # Turing L1/shared BW (from GPU White Paper)
        'arch': 'Turing sm_75',
    },
}

def roofline_analysis(M, N, K):
    """Compute roofline bounds for a given problem size."""
    AI = arithmetic_intensity(M, N, K)
    results = {}
    for gpu_name, spec in GPU_SPECS.items():
        peak_gflops = spec['fp32_peak_tflops'] * 1000
        bw_gbs = spec['mem_bw_gbs']
        # Roofline: min(BW * AI, compute_peak)
        mem_roof_gflops = bw_gbs * AI
        roofline_gflops = min(mem_roof_gflops, peak_gflops)
        bottleneck = 'memory' if mem_roof_gflops < peak_gflops else 'compute'
        results[gpu_name] = {
            'arithmetic_intensity': AI,
            'roofline_gflops':      roofline_gflops,
            'mem_roof_gflops':      mem_roof_gflops,
            'compute_peak_gflops':  peak_gflops,
            'bottleneck_regime':    bottleneck,
            'ridge_point_AI':       peak_gflops / bw_gbs,
        }
    return results

# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Code-level hardware analysis for CUDA GEMM kernels')
    parser.add_argument('--mnk', default='4096,4096,4096',
                        help='M,N,K for roofline/AI analysis (default: 4096,4096,4096)')
    parser.add_argument('--csv', default=None, help='Output CSV file for coalescing/conflict data')
    parser.add_argument('--json', default=None, help='Output full analysis as JSON')
    args = parser.parse_args()

    M, N, K = [int(x) for x in args.mnk.split(',')]

    print("=" * 72)
    print(f"Code-Level Hardware Analysis — CUDA GEMM Kernels")
    print(f"Problem size: M={M}, N={N}, K={K}")
    print(f"ALL values are [CODE-DERIVED] or [CALCULATED] — no GPU required")
    print("=" * 72)

    # ── Global memory coalescing summary ──────────────────────────────────
    print("\n── §2.4 Global Memory Coalescing (Code-Derived) ──────────────────")
    print(f"{'Kernel':<22} {'A-load':>14} {'B-load':>14} {'B-store':>14}  Notes")
    print("-" * 80)

    coalescing_rows = []
    for kname, kdata in KERNEL_ANALYSES.items():
        A_eff = kdata.get('load_A_coalescing', {}).get('coalescing_eff_pct', 'N/A')
        B_eff = kdata.get('load_B_coalescing',
                kdata.get('coalescing_global_B', {})
               ).get('coalescing_eff_pct', 'N/A')
        note = kdata.get('key_improvement', kdata.get('key_problem', ''))[:40]
        A_s = f"{A_eff:.0f}%" if isinstance(A_eff, float) else str(A_eff)
        B_s = f"{B_eff:.0f}%" if isinstance(B_eff, float) else str(B_eff)
        print(f"{kname:<22} {A_s:>14} {B_s:>14} {'100%':>14}  {note}")
        coalescing_rows.append({'kernel': kname, 'A_coalescing_pct': A_s, 'B_coalescing_pct': B_s})

    # ── Shared memory bank conflicts ───────────────────────────────────────
    print("\n── §2.4 Shared Memory Bank Conflicts (Code-Derived) ──────────────")
    print(f"{'Kernel':<22} {'A compute':>16} {'B compute':>16}  Conflict description")
    print("-" * 80)

    for kname, kdata in KERNEL_ANALYSES.items():
        A_conf = kdata.get('smem_A_compute', {})
        B_conf = kdata.get('smem_B_compute', {})
        if A_conf is None or B_conf is None:
            A_d = 'N/A (no smem)'
            B_d = 'N/A (no smem)'
        else:
            A_d = A_conf.get('description', 'unknown')
            B_d = B_conf.get('description', 'unknown')
        print(f"{kname:<22} {A_d:>16} {B_d:>16}")

    # ── Arithmetic intensity ───────────────────────────────────────────────
    print(f"\n── Arithmetic Intensity (Code-Derived, assuming all data from DRAM) ──")
    AI = arithmetic_intensity(M, N, K)
    print(f"  M={M}, N={N}, K={K}: AI = {AI:.1f} FLOP/byte")
    print(f"  All kernels operate at the same AI for the same problem size.")
    print(f"  What differs is whether they ACHIEVE this AI (cache reuse) or fall short.")

    # ── Roofline analysis ─────────────────────────────────────────────────
    print(f"\n── Roofline Model (CALCULATED from spec-sheet numbers) ──────────────")
    roofline = roofline_analysis(M, N, K)
    for gpu_name, r in roofline.items():
        spec = GPU_SPECS[gpu_name]
        print(f"\n  {gpu_name} ({spec['arch']}):")
        print(f"    FP32 peak:      {spec['fp32_peak_tflops']:.2f} TFLOPS = {spec['fp32_peak_tflops']*1000:.0f} GFLOPS")
        print(f"    Memory BW:      {spec['mem_bw_gbs']:.1f} GB/s")
        print(f"    Ridge point:    {r['ridge_point_AI']:.1f} FLOP/byte")
        print(f"    AI (M={M}):      {r['arithmetic_intensity']:.1f} FLOP/byte")
        print(f"    Regime:         {r['bottleneck_regime'].upper()}-BOUND at this size")
        print(f"    Roofline limit: {r['roofline_gflops']:.0f} GFLOPS (theoretical)")

    # ── Turing INT32+FP32 hypothesis ──────────────────────────────────────
    print(f"\n── Turing FP32+INT32 Concurrent Execution Hypothesis ────────────────")
    print("""  Turing (sm_75) introduced independent concurrent FP32 and INT32 pipelines.
  Pascal (sm_61) must time-share the same ALU for both.

  Impact on GEMM kernels:
    Warptile (K6) and double-buffering (K7) kernels mix:
      - Integer ops: index arithmetic (addr = row*stride + col), loop counters
      - FP32 ops:    FMA accumulation (the dominant FLOP work)
    On Pascal: if the FP32 pipeline is saturated, INT ops wait.
    On Turing: INT address calculations can execute in parallel with FP32 FMAs.

  Prediction (testable):
    Turing/Pascal ratio should be slightly higher for K6/K7 (warptile/double_buf)
    than for K2/K3 (smem_tiling/1d_blocktile) where integer overhead is lower.
    Expected magnitude: 2-5% uplift in normalized efficiency (% of peak).
    This is a small but architecturally meaningful signal.
    [TO BE VERIFIED against measured GFLOPS from both nodes]""")

    # ── Output JSON ───────────────────────────────────────────────────────
    if args.json:
        output = {
            'problem_size': {'M': M, 'N': N, 'K': K},
            'arithmetic_intensity': AI,
            'roofline': roofline,
            'kernel_analyses': {k: {kk: str(vv) for kk, vv in v.items()}
                                 for k, v in KERNEL_ANALYSES.items()},
        }
        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nFull analysis saved to: {args.json}")

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['kernel', 'A_coalescing_pct', 'B_coalescing_pct'])
            w.writeheader()
            w.writerows(coalescing_rows)
        print(f"Coalescing CSV saved to: {args.csv}")

    print("\n[!] All values in this report are CODE-DERIVED or CALCULATED.")
    print("    Hardware counter access is blocked (ERR_NVGPUCTRPERM) on Ada cluster.")
    print("    Timing/GFLOPS come from CUDA events in bench_runner.cu.")

if __name__ == '__main__':
    main()
