#!/usr/bin/env python3
"""Command traces for Anaheim's PIM instructions (Kim et al., HPCA'25, Table II).

Every instruction follows the pattern of the paper's Alg. 1 (PAccum<4>). The bank's chunks
are processed G at a time, and each group of G visits the instruction's PolyGroups in turn:
  - load: read a chunk into the buffer, past the MMAC lanes
  - mac: read a chunk and apply one MMAC with buffered operands
  - mac2: the same, plus a second MMAC on that chunk (one extra PIM cycle)
  - cmp: one buffer-only MMAC
  - write: write a buffer entry back
G = floor(B / slots), rounded down to a power of two that divides the PolyGroups' column-group
width, as Alg. 1's "assume G divides 4". The trace names only chunks. The simulator opens
and closes rows, so a group that stays in one row pays one ACT/PRE per PolyGroup visit.

Layout (paper Sec. VI-B, Fig. 7): a PolyGroup with P polynomials splits each row into column
groups (CGs) of w = min(cg_max, row_chunks / P') chunks, P' the next power of two >= P. The
paper's 32-chunk HBM rows give CGs of at most 8 chunks. Fig. 9(c) saturating at B = 16
suggests 16 for 64-chunk GDDR rows. Limb l of
poly i takes C chunks per bank, wrapped over C / w adjacent rows in CG i. Distinct
PolyGroups use distinct rows.

  gen_trace.py --instr PAccum4 --buffer 16 --chunks 16 --limbs 11 --row-chunks 32 -o t.trace
"""
import argparse
import math

# The paper does not spell out any schedule but PAccum<4>'s. The others follow its pattern,
# and their slot counts are checked against Fig. 9 by fig9_sweep.py. Tensor's 6 slots
# (a, b, x, y, z and a temporary) fit Fig. 9 better than 5.
INSTRUCTIONS = {
    # x = a (op) b: load a, stream b through the MMAC into a's slot, write x.
    "Add":    {"slots": 1, "pgs": [["a", "b"], ["x"]],
               "phases": [[("load", "a"), ("mac", "b")], [("write", "x")]]},
    "Mult":   {"slots": 1, "pgs": [["a", "b"], ["x"]],
               "phases": [[("load", "a"), ("mac", "b")], [("write", "x")]]},
    # x = C * a + b
    "CMAC":   {"slots": 1, "pgs": [["a", "b"], ["x"]],
               "phases": [[("load", "b"), ("mac", "a")], [("write", "x")]]},
    "CMult":  {"slots": 1, "pgs": [["a"], ["x"]],
               "phases": [[("mac", "a")], [("write", "x")]]},
    # x = C * (a - b)
    "ModDownEp": {"slots": 1, "pgs": [["a", "b"], ["x"]],
                  "phases": [[("load", "a"), ("mac", "b"), ("cmp", None)], [("write", "x")]]},
    # x = a*p, y = b*p
    "PMult":  {"slots": 3, "pgs": [["p"], ["a", "b"], ["x", "y"]],
               "phases": [[("load", "p")], [("mac", "a"), ("mac", "b")],
                          [("write", "x"), ("write", "y")]]},
    # x = a*c, y = a*d + b*c, z = b*d
    "Tensor": {"slots": 6, "pgs": [["a", "b"], ["c", "d"], ["x", "y", "z"]],
               "phases": [[("load", "a"), ("load", "b")], [("mac2", "c"), ("mac2", "d")],
                          [("write", "x"), ("write", "y"), ("write", "z")]]},
    # x = a*a, y = 2ab, z = b*b
    "TensorSq": {"slots": 5, "pgs": [["a", "b"], ["x", "y", "z"]],
                 "phases": [[("load", "a"), ("load", "b"), ("cmp", None), ("cmp", None),
                             ("cmp", None)], [("write", "x"), ("write", "y"), ("write", "z")]]},
}


def paccum(k, npoly=2):
    """Alg. 1: p_0..p_{K-1} loaded from PolyGroup0, the K ciphertexts' polys streamed from
    PolyGroup1, the npoly outputs written to PolyGroup2. G = floor(B / (K + npoly)).
    npoly = 2 is Table II's PAccum<K>."""
    p = [f"p{i}" for i in range(k)]
    outs = [f"x{j}" for j in range(npoly)]
    ins = [f"a{i}_{j}" for i in range(k) for j in range(npoly)]
    return {"slots": k + npoly, "pgs": [p, ins, outs],
            "phases": [[("load", n) for n in p], [("mac", n) for n in ins],
                       [("write", n) for n in outs]]}


def caccum(k, npoly=2):
    """x_j = C0 + sum_i C_i a_i_j: every input is used once, so they all stream from one
    PolyGroup into npoly accumulators, which are then written. npoly = 2 is Table II's
    CAccum<K>. With unit constants this is also a K-way sum."""
    outs = [f"x{j}" for j in range(npoly)]
    ins = [f"a{i}_{j}" for i in range(k) for j in range(npoly)]
    return {"slots": npoly, "pgs": [ins, outs],
            "phases": [[("mac", n) for n in ins], [("write", n) for n in outs]]}


def lookup(name):
    """Table II names, plus PAccum<K>x<n> and CAccum<K>x<n> for n polys per operand."""
    for prefix, build in (("PAccum", paccum), ("CAccum", caccum)):
        if name.startswith(prefix):
            k, _, n = name[len(prefix):].partition("x")
            return build(int(k), int(n) if n else 2)
    return INSTRUCTIONS[name]


def cg_width(npolys, row_chunks, cg_max=8):
    return min(cg_max, row_chunks // (1 << math.ceil(math.log2(npolys))))


def chunk_size(instr, buffer, row_chunks, cg_max=8):
    """G, or 0 if the buffer cannot hold one chunk position."""
    g = buffer // instr["slots"]
    if g == 0:
        return 0
    cap = min(cg_width(len(pg), row_chunks, cg_max) for pg in instr["pgs"])
    g = min(g, cap)
    return 1 << int(math.log2(g))


def generate(name, buffer, chunks, limbs, row_chunks, cg_max=8, channel=0):
    """Yields trace lines for one instruction on `limbs` limbs of C = `chunks` chunks per bank."""
    instr = lookup(name)
    g = chunk_size(instr, buffer, row_chunks, cg_max)
    if g == 0:
        raise ValueError(f"{name} needs B >= {instr['slots']}")
    where = {}
    for pg_id, pg in enumerate(instr["pgs"]):
        w = cg_width(len(pg), row_chunks, cg_max)
        rows_per_limb = -(-chunks // w)
        for i, poly in enumerate(pg):
            where[poly] = (pg_id, i, w, rows_per_limb)

    def addr(poly, limb, c):
        pg_id, i, w, rows_per_limb = where[poly]
        row = (pg_id << 12) + limb * rows_per_limb + c // w
        return (row << 24) | ((i * w + c % w) << 8) | channel

    for limb in range(limbs):
        for start in range(0, chunks, g):
            pos = range(start, min(start + g, chunks))
            for phase in instr["phases"]:
                for op, poly in phase:
                    for c in pos:
                        if op == "cmp":
                            yield f"PIM_CMP {channel:#x}"
                        elif op == "write":
                            yield f"PIM_WR_AB {addr(poly, limb, c):#x}"
                        else:
                            yield f"PIM_RD_AB {addr(poly, limb, c):#x}"
                            if op == "mac2":
                                yield f"PIM_CMP {channel:#x}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instr", required=True)
    ap.add_argument("--buffer", type=int, required=True, help="B, buffer entries per unit")
    ap.add_argument("--chunks", type=int, default=16, help="C, chunks per bank per limb")
    ap.add_argument("--limbs", type=int, default=11, help="limbs per die group")
    ap.add_argument("--row-chunks", type=int, default=32, help="chunks per bank row")
    ap.add_argument("--cg-max", type=int, default=8, help="widest column group, in chunks")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w") as f:
        for line in generate(a.instr, a.buffer, a.chunks, a.limbs, a.row_chunks, a.cg_max):
            f.write(line + "\n")


if __name__ == "__main__":
    main()
