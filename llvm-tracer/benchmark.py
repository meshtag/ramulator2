#!/usr/bin/env python3
"""
benchmark.py — Run all test workloads across LLVM optimization levels,
               collect Ramulator2 cycle counts, and generate a bar chart.

Usage:
    python3 benchmark.py

Requires:
    - llvm-tracer built (make)
    - Ramulator2 built
    - matplotlib (pip3 install matplotlib)
"""

import subprocess
import os
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_TRACE = os.path.join(SCRIPT_DIR, "run_trace.sh")
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "ramulator_config.yaml")

RAMULATOR2 = os.environ.get(
    "RAMULATOR2",
    os.path.join(SCRIPT_DIR, "..", "build", "ramulator2"),
)

WORKLOADS = [
    ("matmul",      os.path.join(SCRIPT_DIR, "test", "matmul.c")),
    ("stencil",     os.path.join(SCRIPT_DIR, "test", "stencil.c")),
    ("linked_list", os.path.join(SCRIPT_DIR, "test", "linked_list.c")),
]

OPT_LEVELS = ["none", "O1", "O2", "O3"]


def run_workload(source_path, opt_level):
    """Run a single workload through trace generation + Ramulator2.

    Returns dict with keys:
        cycles, dram_reqs, total_accesses, hit_rate, misses
    """
    trace_path = os.path.join(SCRIPT_DIR, "_bench_trace.txt")

    env = os.environ.copy()
    env["OPT_LEVEL"] = opt_level
    env["RAMULATOR2"] = ""
    env["TRACE_OUTPUT"] = trace_path
    env["RAMULATOR_TRACE_OUT"] = trace_path

    trace_result = subprocess.run(
        [RUN_TRACE, source_path],
        capture_output=True, text=True, env=env, cwd=SCRIPT_DIR,
    )
    trace_output = trace_result.stdout + "\n" + trace_result.stderr

    r = {"cycles": None, "dram_reqs": None, "total_accesses": None,
         "hit_rate": None, "misses": None}

    for line in trace_output.splitlines():
        m = re.search(r"Total accesses\s*:\s*(\d+)", line)
        if m:
            r["total_accesses"] = int(m.group(1))
        m = re.search(r"DRAM requests\s*:\s*(\d+)", line)
        if m:
            r["dram_reqs"] = int(m.group(1))
        m = re.search(r"Misses\s*:\s*(\d+)", line)
        if m:
            r["misses"] = int(m.group(1))
        m = re.search(r"Hits\s*:\s*\d+\s*\(([\d.]+)%\)", line)
        if m:
            r["hit_rate"] = float(m.group(1))

    # Run Ramulator2 on the generated trace
    tmp_config = os.path.join(SCRIPT_DIR, "_bench_config.yaml")
    if os.path.isfile(trace_path) and os.path.getsize(trace_path) > 0:
        with open(DEFAULT_CONFIG) as f:
            cfg = f.read()
        cfg = re.sub(r"(path:\s*).*", rf"\g<1>{trace_path}", cfg)
        with open(tmp_config, "w") as f:
            f.write(cfg)

        ram_result = subprocess.run(
            [RAMULATOR2, "--config_file", tmp_config],
            capture_output=True, text=True, cwd=SCRIPT_DIR,
        )
        ram_output = ram_result.stdout + "\n" + ram_result.stderr
        for line in ram_output.splitlines():
            m = re.search(r"memory_system_cycles:\s*(\d+)", line)
            if m:
                r["cycles"] = int(m.group(1))

    for f in [trace_path, tmp_config]:
        if os.path.exists(f):
            os.unlink(f)

    if r["cycles"] is None:
        print(f"\n  WARNING: Failed to get cycles for {source_path} -{opt_level}")

    return r


def cleanup_intermediates():
    test_dir = os.path.join(SCRIPT_DIR, "test")
    for f in os.listdir(test_dir):
        if f.endswith((".ll", ".exe")):
            os.unlink(os.path.join(test_dir, f))


def fmt(val, width=12):
    if val is not None:
        return f"{val:>{width},}"
    return " " * (width - 5) + "ERROR"


def main():
    if not os.path.isfile(RAMULATOR2):
        print(f"ERROR: ramulator2 not found at {RAMULATOR2}")
        print("Set RAMULATOR2 env var or build ramulator2 first.")
        sys.exit(1)

    print(f"Ramulator2: {RAMULATOR2}")
    print(f"Workloads:  {[w[0] for w in WORKLOADS]}")
    print(f"Opt levels: {OPT_LEVELS}")
    print()

    results = {}

    for wl_name, wl_path in WORKLOADS:
        results[wl_name] = {}
        for opt in OPT_LEVELS:
            label = f"{wl_name} / -{opt}" if opt != "none" else f"{wl_name} / -O0 (none)"
            print(f"  Running {label} ...", end=" ", flush=True)
            r = run_workload(wl_path, opt)
            results[wl_name][opt] = r
            hit_str = f"{r['hit_rate']:.2f}" if r["hit_rate"] is not None else "N/A"
            print(f"accesses={r['total_accesses']}, misses={r['misses']}, "
                  f"cycles={r['cycles']}, hit={hit_str}%")

    cleanup_intermediates()

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print(f"{'Workload':<15} {'Opt':<6} {'Total Accesses':>15} {'Misses':>12} "
          f"{'Cycles':>12} {'Hit%':>10}")
    print("-" * 88)
    for wl_name in [w[0] for w in WORKLOADS]:
        for opt in OPT_LEVELS:
            r = results[wl_name][opt]
            opt_label = f"-{opt}" if opt != "none" else "-O0"
            h = f"{r['hit_rate']:.2f}%" if r["hit_rate"] is not None else "ERROR"
            print(f"{wl_name:<15} {opt_label:<6} {fmt(r['total_accesses'], 15)} "
                  f"{fmt(r['misses'])} {fmt(r['cycles'])} {h:>10}")
        print()

    # ── Plot: 3 separate figures ────────────────────────────────────
    workload_names = [w[0] for w in WORKLOADS]
    n_wl = len(workload_names)
    n_opts = len(OPT_LEVELS)
    x = np.arange(n_wl)
    bar_width = 0.18

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    opt_labels = ["-O0 (none)", "-O1", "-O2", "-O3"]

    panels = [
        ("Total Memory Accesses (pre-cache)", "total_accesses",
         "Total Accesses (log scale)", "total_accesses.png", True),
        ("DRAM Requests (cache misses + writebacks)", "dram_reqs",
         "DRAM Requests", "dram_requests.png", False),
        ("Ramulator2 DRAM Cycles", "cycles",
         "Cycles", "dram_cycles.png", False),
    ]

    for title, key, ylabel, filename, use_log in panels:
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, (opt, color, label) in enumerate(zip(OPT_LEVELS, colors, opt_labels)):
            vals = [results[wl][opt].get(key) or 0 for wl in workload_names]
            offset = (i - (n_opts - 1) / 2) * bar_width
            bars = ax.bar(x + offset, vals, bar_width,
                          label=label, color=color, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0:
                    if val >= 1_000_000:
                        txt = f"{val/1e6:.1f}M"
                    elif val >= 10_000:
                        txt = f"{val/1e3:.0f}K"
                    else:
                        txt = f"{val:,}"
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            txt, ha="center", va="bottom", fontsize=8, rotation=45)

        if use_log:
            ax.set_yscale("log")

        ax.set_xlabel("Workload", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(workload_names, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(SCRIPT_DIR, filename)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {out_path}")


if __name__ == "__main__":
    main()
