#!/usr/bin/env python3
"""Anaheim's Fig. 9 trend check: each instruction's PIM time at B = 4..64, relative to its
smallest supported B, against the bars of Fig. 9(a) (A100 80GB, near-bank).

The bars were read off the paper's figure, so they carry about +-0.02. Fig. 9 also gives
speedups over Cheddar's GPU kernels. Those need the A100 GPU times and are checked elsewhere.

  fig9_sweep.py                    # (a): A100 config, C 16, 11 limbs, 32-chunk rows
  fig9_sweep.py --fig c --config ../anaheim_a6000.yaml --chunks 32 --limbs 18 \
      --row-chunks 64 --cg-max 16
"""
import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

from gen_trace import chunk_size, generate, lookup

HERE = Path(__file__).resolve().parent
BUFFERS = (4, 8, 16, 32, 64)
# Relative time per B read off Fig. 9 (None: not supported at that B). (a) is the A100 80GB
# near-bank. (c) is the RTX 4090 (GDDR6X) near-bank, a shape check for our GDDR6 port.
FIG9 = {"a": {
    "Add":     (1.00, 0.64, 0.64, 0.64, 0.64),
    "Mult":    (1.00, 0.64, 0.64, 0.64, 0.64),
    "CMAC":    (1.00, 0.65, 0.65, 0.65, 0.65),
    "Tensor":  (None, 1.00, 0.55, 0.30, 0.21),
    "PAccum4": (None, 1.00, 0.60, 0.42, 0.42),
    "CAccum4": (1.00, 0.70, 0.70, 0.70, 0.70),
}, "c": {
    "Add":     (1.00, 0.60, 0.40, 0.40, 0.40),
    "Mult":    (1.00, 0.60, 0.40, 0.40, 0.40),
    "CMAC":    (1.00, 0.60, 0.40, 0.40, 0.40),
    "Tensor":  (None, 1.00, 0.51, 0.28, 0.17),
    "PAccum4": (None, 1.00, 0.54, 0.34, 0.25),
    "CAccum4": (1.00, 0.65, 0.47, 0.38, 0.38),
}}


def simulate(config, lines, workdir):
    trace = workdir / "t.trace"
    trace.write_text("\n".join(lines) + "\n")
    cfg = workdir / "run.yaml"
    cfg.write_text(Path(config).read_text().replace("path: ./anaheim.trace", f"path: {trace}"))
    out = subprocess.run([str(HERE.parent / "build" / "ramulator2"), "-f", str(cfg)],
                         capture_output=True, text=True, check=True).stdout
    return int(re.search(r"memory_system_cycles: (\d+)", out).group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE.parent / "anaheim_a100.yaml"))
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--limbs", type=int, default=11)
    ap.add_argument("--row-chunks", type=int, default=32)
    ap.add_argument("--cg-max", type=int, default=8)
    ap.add_argument("--fig", choices=("a", "c"), default="a")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        workdir = Path(d)
        print(f"{'instr':8s} {'B':>3s} {'G':>2s} {'cycles':>8s} {'ours':>5s} {'Fig9':>5s}")
        for name, paper in FIG9[a.fig].items():
            cycles = {}
            for b in BUFFERS:
                g = chunk_size(lookup(name), b, a.row_chunks, a.cg_max)
                if g:
                    cycles[b] = (g, simulate(a.config, list(generate(
                        name, b, a.chunks, a.limbs, a.row_chunks, a.cg_max)), workdir))
            base = cycles[min(cycles)][1]
            for b, want in zip(BUFFERS, paper):
                if b not in cycles:
                    print(f"{name:8s} {b:3d}  - {'unsupported':>8s}  {'-' if want is None else want}")
                    continue
                g, c = cycles[b]
                print(f"{name:8s} {b:3d} {g:2d} {c:8d} {c / base:5.2f} "
                      f"{'-' if want is None else f'{want:.2f}':>5s}")


if __name__ == "__main__":
    main()
