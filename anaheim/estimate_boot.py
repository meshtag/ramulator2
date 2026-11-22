#!/usr/bin/env python3
"""Anaheim's time for one bootstrap, from the nsys profile of the same bootstrap in Cheddar.

This follows the paper's method (Sec. VII-A). Every element-wise GPU kernel is replaced by
the Anaheim instructions that do its work, and their PIM time comes from the simulator.
NTT, BConv, automorphism and copies keep their measured GPU time. GPU and PIM never
overlap, and each GPU/PIM switch costs SWITCH_US.

Three Cheddar kernels fuse element-wise work with GPU-only work:
  NTTPhase2ForModDown: the NTT stays at a plain NTTPhase2's measured time per limb, and the
      epilogue becomes ModDownEp.
  BSFusedKernel: R KeyMults become PAccum<D>, and the automorphism stays at Permute's
      measured time per polynomial limb.
  GSFusedKernel: gs giant steps become PAccum<bs>. Anaheim has no instruction that reuses
      baby steps across giant steps, as this kernel does.
The fused kernels' bs, gs, D and R are read exactly from the pointer arrays Cheddar copies
to the GPU before each launch.

Not modeled: Anaheim's GPU kernels changed for PIM (L2 write-backs, which the paper puts at
1.03-1.11x on (I)NTT and automorphism) and its op reordering.

  estimate_boot.py <profile>.sqlite [--config ../anaheim_a6000.yaml] [--preset ...]
"""
import argparse
import math
import re
import sqlite3
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from fig9_sweep import simulate
from gen_trace import generate

HERE = Path(__file__).resolve().parent
N = 1 << 16
SWITCH_US = 2.0
GPU_ONLY = {"NTTPhase1", "NTTPhase2", "INTTPhase1", "INTTPhase2", "ModSwitchMatrixMult",
            "Permute", "PermuteAccum", "PermuteAccumAdd", "ModUpToMax1", "ModUpToMax2"}


def pim_ops(base, nums, lists, limbs, extra):
    """The Anaheim instructions doing one Cheddar kernel's element-wise work: [(name, limbs)]."""
    if base == "CPAccum":
        npoly, const = nums[0], nums[1]
        return [(f"{'CAccum' if const else 'PAccum'}{lists}x{npoly}", limbs)]
    if base == "CPAccumAdd":
        return [(f"CAccum{lists + 1}x{nums[0]}", limbs)]
    if base == "Sum":
        return [(f"CAccum{lists}x{nums[0]}", limbs)]
    if base == "Tensor":
        return [("Tensor", limbs)]
    if base == "TensorSquare":
        return [("TensorSq", limbs)]
    if base in ("AddConst", "SubConst", "SubOppositeConst", "Neg"):
        return [("CMult", limbs)] * nums[0]
    if base == "Sub":
        return [("Add", limbs)] * nums[0]
    if base == "MultImaginaryUnit":
        return [("Mult", limbs)] * nums[0]
    if base == "GSFusedKernel":
        bs, gs = extra
        return [(f"PAccum{bs}", limbs)] * gs
    if base == "BSFusedKernel":
        d, r = extra
        return [(f"PAccum{d}", limbs)] * r
    if base == "NTTPhase2ForModDown":
        return [("ModDownEp", limbs)]
    raise KeyError(f"no Anaheim mapping for Cheddar kernel {base}")


def load(path):
    """Kernels and copies in time order: (start, end, kind, base, nums, lists, grid, htod)."""
    c = sqlite3.connect(path)
    names = dict(c.execute("select id, value from StringIds"))
    acts = []
    for s, e, nid, gx, gy, gz, bx in c.execute(
            "select start, end, demangledName, gridX, gridY, gridZ, blockX "
            "from CUPTI_ACTIVITY_KIND_KERNEL"):
        m = re.search(r"kernel::([A-Za-z_0-9]+)(?:<(.*?)>)?\(", names[nid])
        targs = m.group(2) or ""
        acts.append((s, e, "kernel", m.group(1),
                     [int(x) for x in re.findall(r"\((?:int|bool)\)(\d+)", targs)],
                     len(re.findall(r"PtrList<", targs)), (gx, gy, gz, bx), None))
    for s, e, nbytes, kind in c.execute(
            "select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
        acts.append((s, e, "memcpy", "memcpy", [], 0, None, (kind, nbytes)))
    # nsys leaves out the table of an activity kind that never happened.
    tables = {t for (t,) in c.execute("select name from sqlite_master where type = 'table'")}
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        for s, e in c.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMSET"):
            acts.append((s, e, "memset", "memset", [], 0, None, None))
    acts.sort()
    return acts


def rates(acts):
    """Measured GPU rates for the parts of fused kernels that stay on the GPU: NTTPhase2 in
    us per limb and Permute in us per polynomial limb."""
    ntt2 = [((e - s) / 1e3) / g[1] for s, e, k, b, n, l, g, h in acts if b == "NTTPhase2"]
    perm = [((e - s) / 1e3) / (n[0] * g[0] * g[3] / N) for s, e, k, b, n, l, g, h in acts
            if b == "Permute"]
    return statistics.median(ntt2), statistics.median(perm)


def launches(acts):
    """Every kernel and copy in time order as (base, us, side, kept, ops, limbs, extra): its
    Cheddar time, "gpu", "pim" or None for a copy, the GPU time it keeps as {label: us}, the
    Anaheim instructions doing its element-wise work, and its limbs and (bs, gs) or (D, R)."""
    ntt2_us_per_limb, perm_us_per_poly_limb = rates(acts)
    htod = []
    for s, e, kind, base, nums, lists, grid, copy in acts:
        us = (e - s) / 1e3
        if kind != "kernel":
            if copy and copy[0] == 1:
                htod.append(copy[1])
            yield base, us, None, {"copies": us}, [], None, None
            continue
        if base in GPU_ONLY:
            yield base, us, "gpu", {base: us}, [], None, None
        else:
            kept, extra = {}, None
            if base == "NTTPhase2ForModDown":
                limbs = grid[1]
                kept["NTT (ModDown)"] = min(us, ntt2_us_per_limb * limbs)
            else:
                limbs = grid[0] * grid[3] / N
                if base == "GSFusedKernel":
                    bx, ax, mx, dbx, dax = htod[-5:]
                    extra = (bx // 8, dax // 8)
                elif base == "BSFusedKernel":
                    modup, ka, kb, db, da, kx, gal = htod[-7:]
                    extra = (modup // 8, gal // 4)
                    kept["automorphism (KeyMult)"] = (extra[1] * 2 * limbs
                                                      * perm_us_per_poly_limb)
            yield (base, us, "pim", kept, pim_ops(base, nums, lists, limbs, extra), limbs,
                   extra)
        htod = []


def estimate(a, acts, work, replace=None):
    """The estimate's parts: Cheddar's total, the GPU time kept and the PIM time by kind (us),
    the switch count and Cheddar's time per kernel. replace(base, extra, limbs per group,
    Anaheim's time), where it returns a time, stands in for a launch's Anaheim instructions."""
    config = Path(a.config).read_text()
    if a.preset:
        config = re.sub(r"(timing:\s*\n\s*preset:) \S+", rf"\1 {a.preset}", config)
    cfg = work / "base.yaml"
    cfg.write_text(config)
    cache = {}

    def pim_us(name, limbs):
        per_group = math.ceil(limbs / a.groups)
        key = (name, per_group)
        if key not in cache:
            lines = list(generate(name, a.buffer, a.chunks, per_group, a.row_chunks, a.cg_max))
            cache[key] = simulate(cfg, lines, work) * a.tck_ns / 1e3
        return cache[key]

    total = 0.0
    kept, pim, cheddar = defaultdict(float), defaultdict(float), defaultdict(float)
    switches, last = 0, "gpu"
    for base, us, side, gpu, ops, limbs, extra in launches(acts):
        total += us
        for label, t in gpu.items():
            kept[label] += t
        if side is None:
            continue
        cheddar[base] += us
        times = [(name.rstrip("0123456789x") or name, pim_us(name, lb)) for name, lb in ops]
        mine = None
        if replace and ops:
            mine = replace(base, extra, math.ceil(limbs / a.groups), sum(t for _, t in times))
        for kind, t in times if mine is None else [(base, mine)]:
            pim[kind] += t
        if side != last:
            switches += 1
            last = side
    return total, kept, pim, switches, cheddar


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile")
    ap.add_argument("--config", default=str(HERE.parent / "anaheim_a6000.yaml"))
    ap.add_argument("--preset", help="override the DRAM timing preset")
    ap.add_argument("--groups", type=int, default=3, help="die groups (A6000 port: 3)")
    ap.add_argument("--chunks", type=int, default=32, help="C, chunks per bank per limb")
    ap.add_argument("--row-chunks", type=int, default=64)
    ap.add_argument("--cg-max", type=int, default=16)
    ap.add_argument("--buffer", type=int, default=32, help="B")
    ap.add_argument("--tck-ns", type=float, default=0.5)
    return ap


def main():
    a = parser().parse_args()
    acts = load(a.profile)
    ntt2_us_per_limb, perm_us_per_poly_limb = rates(acts)
    with tempfile.TemporaryDirectory() as d:
        total_gpu, kept, pim, switches, cheddar = estimate(a, acts, Path(d))
    gpu_kept = sum(kept.values())
    pim_total = sum(pim.values())
    switch_total = switches * SWITCH_US
    est = gpu_kept + pim_total + switch_total
    print(f"Cheddar (GPU only): {total_gpu / 1e3:.2f} ms of kernels and copies")
    print(f"Anaheim estimate:   {est / 1e3:.2f} ms = GPU {gpu_kept / 1e3:.2f} + PIM "
          f"{pim_total / 1e3:.2f} + {switches} switches x {SWITCH_US:g} us "
          f"({switch_total / 1e3:.2f}); speedup {total_gpu / est:.2f}x")
    print(f"  measured rates: NTTPhase2 {ntt2_us_per_limb:.3f} us/limb, Permute "
          f"{perm_us_per_poly_limb:.3f} us per poly-limb")
    print("  GPU kept (ms): " + ", ".join(f"{k} {v / 1e3:.2f}" for k, v in
                                          sorted(kept.items(), key=lambda x: -x[1])))
    print("  PIM (ms):      " + ", ".join(f"{k} {v / 1e3:.2f}" for k, v in
                                          sorted(pim.items(), key=lambda x: -x[1])))
    ew = {k: v for k, v in cheddar.items() if k not in GPU_ONLY}
    print("  Cheddar time of the replaced kernels (ms): " + ", ".join(
        f"{k} {v / 1e3:.2f}" for k, v in sorted(ew.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
