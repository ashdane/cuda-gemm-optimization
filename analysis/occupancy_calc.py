#!/usr/bin/env python3

import re
import argparse
import csv
import sys
import math

# ─── Architecture hardware limits ───────────────────────────────────────────
ARCH_LIMITS = {
    'sm_61': {
        'max_threads_per_sm':   2048,
        'max_blocks_per_sm':    32,
        'max_regs_per_sm':      65536,
        'max_smem_per_sm':      49152,   # 48 KB
        'max_regs_per_thread':  255,
        'warp_size':            32,
        'gpu_name':             'GTX 1080 Ti (Pascal)',
    },
    'sm_75': {
        'max_threads_per_sm':   1024,
        'max_blocks_per_sm':    16,
        'max_regs_per_sm':      65536,
        'max_smem_per_sm':      65536,   # 64 KB
        'max_regs_per_thread':  255,
        'warp_size':            32,
        'gpu_name':             'RTX 2080 Ti (Turing)',
    },
}

# ─── ptxas output parser ─────────────────────────────────────────────────────
PTXAS_RE = re.compile(
    r'ptxas info\s*:\s*Function properties for\s+(\S+)\n'  # kernel name (greedy)
    r'(?:.*\n)*?'
    r'.*?ptxas info\s*:\s*'
    r'Used (\d+) registers.*?(\d+) bytes smem',
    re.MULTILINE
)

# Alternative simpler patterns (ptxas output varies by CUDA version)
KERNEL_RE   = re.compile(r'ptxas info\s*:\s*Function properties for\s+(.+)')
RESOURCE_RE = re.compile(r'ptxas info\s*:\s*Used (\d+) registers,.*?(?:(\d+) bytes cumulative stack size,)?.*?(\d+) bytes smem,\s*(\d+) bytes cmem')
THREADS_RE  = re.compile(r'launch_bounds\s+(\d+)')  # if present

def parse_ptxas_log(log_text):
    """Parse ptxas log and return list of (kernel_name, registers, smem_bytes, threads)."""
    entries = []
    lines = log_text.split('\n')

    current_kernel = None
    for i, line in enumerate(lines):
        km = KERNEL_RE.search(line)
        if km:
            current_kernel = km.group(1).strip()
            continue

        rm = RESOURCE_RE.search(line)
        if rm and current_kernel:
            regs  = int(rm.group(1))
            smem  = int(rm.group(3))
            # threads per block: not in ptxas directly, we get it from kernel name convention
            # or we set it explicitly via compilation
            entries.append({
                'kernel_name': current_kernel,
                'registers':   regs,
                'smem_bytes':  smem,
            })
            current_kernel = None  # reset

    return entries

# ─── Occupancy calculator ────────────────────────────────────────────────────
def calc_occupancy(threads_per_block, registers_per_thread, smem_bytes, arch):
    """
    Calculate theoretical occupancy using CUDA occupancy calculator formula.
    Returns dict with breakdown and occupancy percentage.
    """
    limits = ARCH_LIMITS[arch]
    warp_size          = limits['warp_size']
    max_threads_sm     = limits['max_threads_per_sm']
    max_blocks_sm      = limits['max_blocks_per_sm']
    max_regs_sm        = limits['max_regs_per_sm']
    max_smem_sm        = limits['max_smem_per_sm']

    if threads_per_block <= 0:
        return {'occupancy_pct': 0.0, 'limiting_factor': 'invalid_threads'}

    warps_per_block = math.ceil(threads_per_block / warp_size)
    max_warps_sm    = max_threads_sm // warp_size

    # Register limit: CUDA allocates registers in 256-register granularity per warp
    if registers_per_thread > 0:
        regs_per_warp   = math.ceil(registers_per_thread * warp_size / 256) * 256
        regs_per_block  = regs_per_warp * warps_per_block
        reg_limit       = max_regs_sm // regs_per_block if regs_per_block > 0 else max_blocks_sm
    else:
        reg_limit = max_blocks_sm

    # Shared memory limit: smem allocated in 256-byte granularity
    smem_per_block = math.ceil(max(smem_bytes, 1) / 256) * 256
    smem_limit     = max_smem_sm // smem_per_block

    # Thread count limit
    thread_limit   = max_threads_sm // threads_per_block

    # Hardware block limit
    blocks_per_sm  = min(reg_limit, smem_limit, thread_limit, max_blocks_sm)
    blocks_per_sm  = max(blocks_per_sm, 0)

    achieved_warps = blocks_per_sm * warps_per_block
    occupancy_pct  = 100.0 * achieved_warps / max_warps_sm

    # Identify limiting factor
    lims = {
        'registers':     reg_limit,
        'shared_memory': smem_limit,
        'thread_count':  thread_limit,
        'block_limit':   max_blocks_sm,
    }
    limiting = min(lims, key=lims.get)

    return {
        'warps_per_block':           warps_per_block,
        'max_warps_sm':              max_warps_sm,
        'blocks_per_sm_reg':         reg_limit,
        'blocks_per_sm_smem':        smem_limit,
        'blocks_per_sm_threads':     thread_limit,
        'blocks_per_sm_hw':          max_blocks_sm,
        'blocks_per_sm':             blocks_per_sm,
        'achieved_warps_per_sm':     achieved_warps,
        'occupancy_pct':             occupancy_pct,
        'limiting_factor':           limiting,
    }

# Kernel-to-threads-per-block mapping (from our source code)
KERNEL_THREADS = {
    'sgemm_naive':           1024,   # 32×32
    'sgemm_coalesced':       1024,   # 32×32
    'sgemm_smem':            1024,   # 32×32
    'sgemm_1d_blocktile':    512,    # (BM/TM)*BN = 8*64
    'sgemm_2d_blocktile':    256,    # (BM/TM)*(BN/TN) = 16*16
    'sgemm_vectorized':      256,
    'sgemm_warptile':        256,
    'sgemm_tensor_core':     128,
}

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='CUDA occupancy calculator from ptxas output')
    parser.add_argument('--ptxas-log', required=True,
                        help='Path to ptxas log file (captured from nvcc --ptxas-options=-v)')
    parser.add_argument('--arch', required=True, choices=list(ARCH_LIMITS.keys()),
                        help='Target architecture (sm_61 or sm_75)')
    parser.add_argument('--csv', default=None,
                        help='Optional output CSV file path')
    args = parser.parse_args()

    with open(args.ptxas_log, 'r') as f:
        log_text = f.read()

    entries = parse_ptxas_log(log_text)
    if not entries:
        print("ERROR: No ptxas entries found. Check log format.", file=sys.stderr)
        print("Expected pattern:", file=sys.stderr)
        print("  ptxas info : Function properties for <kernel_name>", file=sys.stderr)
        print("  ptxas info : Used N registers, M bytes smem, ...", file=sys.stderr)
        sys.exit(1)

    limits = ARCH_LIMITS[args.arch]
    print(f"\n=== Occupancy Analysis: {args.arch} ({limits['gpu_name']}) ===")
    print(f"{'Source':40s} {'Reg':>5} {'Smem':>7} {'Thd/Blk':>8} {'Blk/SM':>7} {'Occ%':>6} {'Limit':>16}")
    print("-" * 105)

    results = []
    for entry in entries:
        k_name = entry['kernel_name']
        regs   = entry['registers']
        smem   = entry['smem_bytes']

        # Lookup threads per block from our known mapping
        # Fall back to 256 if not recognized
        threads = 256
        for canonical, t in KERNEL_THREADS.items():
            if canonical in k_name or k_name in canonical:
                threads = t
                break

        occ = calc_occupancy(threads, regs, smem, args.arch)

        row = {
            'arch':                  args.arch,
            'kernel_name':           k_name,
            'threads_per_block':     threads,
            'registers_per_thread':  regs,
            'smem_bytes':            smem,
            'blocks_per_sm':         occ['blocks_per_sm'],
            'occupancy_pct':         occ['occupancy_pct'],
            'limiting_factor':       occ['limiting_factor'],
            'warps_per_sm':          occ['achieved_warps_per_sm'],
            'max_warps_sm':          occ['max_warps_sm'],
        }
        results.append(row)

        print(f"{k_name[:40]:40s} {regs:>5} {smem:>7} {threads:>8} "
              f"{occ['blocks_per_sm']:>7} {occ['occupancy_pct']:>5.1f}% "
              f"{occ['limiting_factor']:>16}")

    print("-" * 105)
    print(f"\n[!] Occupancy values are CALCULATED from ptxas data + GPU spec limits.")
    print(f"    Hardware counter measurement blocked (ERR_NVGPUCTRPERM on Ada cluster).")
    print(f"    Formula: CUDA Programming Guide, Appendix G (Occupancy Calculator).")

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV written to: {args.csv}")

if __name__ == '__main__':
    main()
