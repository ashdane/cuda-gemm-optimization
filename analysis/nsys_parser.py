#!/usr/bin/env python3
"""
nsys_parser.py — Parse `nsys profile --trace=cuda --stats=true` stdout output
to extract kernel GPU durations. This is our secondary timing source
(primary = CUDA events in bench_runner; secondary = nsys for cross-validation).

nsys --stats=true prints a table like:

 Time(%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)  Name
 -------  ---------------  ---------  --------  --------  --------  --------  -----------  ----
   89.5%      123456789        100     1234567   1230000   1210000   1280000       12345    sgemm_warptile(...)

Usage:
    nsys profile --trace=cuda --stats=true ./bench_runner_sm75 6 4096 4096 4096 5 20 > nsys_out.txt 2>&1
    python3 nsys_parser.py nsys_out.txt

Output: CSV rows with kernel name, count, avg_ms, med_ms, min_ms, max_ms, stddev_ms
"""

import sys
import re
import argparse
import csv

# Match the GPU kernel timing table from nsys --stats=true output
# nsys outputs a "CUDA Kernel Statistics" or "GPU Kernel Summary" section
HEADER_PATTERNS = [
    r"CUDA Kernel Statistics",
    r"GPU Kernel Summary",
    r"Time\(%\)\s+Total Time",
]
ROW_RE = re.compile(
    r"^\s*(\d+\.?\d*)\%\s+"    # Time(%)
    r"(\d+)\s+"                 # Total Time (ns)
    r"(\d+)\s+"                 # Instances
    r"(\d+\.?\d*)\s+"          # Avg (ns)
    r"(\d+\.?\d*)\s+"          # Med (ns)
    r"(\d+\.?\d*)\s+"          # Min (ns)
    r"(\d+\.?\d*)\s+"          # Max (ns)
    r"(\d+\.?\d*)\s+"          # StdDev (ns)
    r"(.+)$"                    # Name
)

def ns_to_ms(ns_str):
    return float(ns_str) / 1e6

def parse_nsys_stats(text):
    """Parse nsys --stats=true output, return list of kernel stat dicts."""
    results = []
    in_kernel_section = False
    past_header_line = False

    for line in text.splitlines():
        # Detect start of GPU kernel stats section
        if any(re.search(p, line, re.IGNORECASE) for p in HEADER_PATTERNS):
            in_kernel_section = True
            past_header_line = False
            continue

        if in_kernel_section:
            # Skip header/separator lines (contain dashes)
            if re.match(r"^\s*[-]+\s*$", line) or re.match(r"^\s*Time", line):
                past_header_line = True
                continue

            if not past_header_line:
                continue

            # Empty line ends the section
            if line.strip() == "":
                if results:  # Only stop if we've collected something
                    in_kernel_section = False
                continue

            m = ROW_RE.match(line)
            if m:
                pct, total_ns, instances, avg_ns, med_ns, min_ns, max_ns, std_ns, name = m.groups()
                # Strip template parameters from kernel name for readability
                clean_name = re.sub(r'\([^)]*\)', '', name.strip())
                clean_name = clean_name.split('<')[0].strip()
                results.append({
                    'time_pct':    float(pct),
                    'total_ns':    int(total_ns),
                    'instances':   int(instances),
                    'avg_ms':      ns_to_ms(avg_ns),
                    'med_ms':      ns_to_ms(med_ns),
                    'min_ms':      ns_to_ms(min_ns),
                    'max_ms':      ns_to_ms(max_ns),
                    'stddev_ms':   ns_to_ms(std_ns),
                    'kernel_name': clean_name,
                    'full_name':   name.strip(),
                })

    return results

def compute_gflops_from_nsys(row, M, N, K):
    """Compute GFLOPS from nsys median duration."""
    if row['med_ms'] == 0:
        return 0.0
    return 2.0 * M * N * K / (row['med_ms'] * 1e6)

def main():
    parser = argparse.ArgumentParser(
        description='Parse nsys --stats=true output for kernel GPU durations')
    parser.add_argument('nsys_file', help='nsys stdout capture file')
    parser.add_argument('--mnk', default=None,
                        help='M,N,K for GFLOPS calculation (e.g. 4096,4096,4096)')
    parser.add_argument('--csv', default=None, help='Output CSV path')
    args = parser.parse_args()

    with open(args.nsys_file, 'r') as f:
        text = f.read()

    results = parse_nsys_stats(text)

    if not results:
        print("WARNING: No GPU kernel statistics found in nsys output.")
        print("  Ensure the file was produced with: nsys profile --trace=cuda --stats=true")
        print("  and that at least one CUDA kernel ran successfully.")
        sys.exit(1)

    M, N, K = (4096, 4096, 4096)
    if args.mnk:
        M, N, K = [int(x) for x in args.mnk.split(',')]

    print(f"\n=== nsys GPU Kernel Timing Summary ===")
    print(f"    (Cross-validation source — primary timing from CUDA events in bench_runner)")
    print(f"    M={M} N={N} K={K}")
    print()
    print(f"{'Kernel':<40} {'Instances':>10} {'Med(ms)':>10} {'GFLOPS':>10} {'StdDev(ms)':>12}")
    print("-" * 88)

    output_rows = []
    for row in results:
        gf = compute_gflops_from_nsys(row, M, N, K)
        print(f"{row['kernel_name']:<40} {row['instances']:>10} {row['med_ms']:>10.4f} "
              f"{gf:>10.1f} {row['stddev_ms']:>12.4f}")
        output_rows.append({**row, 'GFLOPS_from_nsys': gf, 'M': M, 'N': N, 'K': K})

    print()
    print("NOTE: GFLOPS computed as 2*M*N*K / med_ms. This matches bench_runner's")
    print("      CUDA-event timing within ~1% for kernels without CPU overlap.")

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=output_rows[0].keys())
            w.writeheader()
            w.writerows(output_rows)
        print(f"\nCSV written: {args.csv}")

if __name__ == '__main__':
    main()
