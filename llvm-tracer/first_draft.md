# HBM PIM: First Draft

## Goal

Map PIM-compatible kernels onto an HBM Processing-in-Memory (PIM) architecture,
generate cycle-accurate PIM traces via LLVM instrumentation, and simulate them
in Ramulator2 with a custom HBM3_PIM DRAM model. Included examples: attention
(single-head scaled dot-product) and AXPY. A stepping stone toward Triton-level
PIM integration.

---

## Architecture Overview

```
  attention_pim.c /        pim_runtime.c           Ramulator2
  axpy_pim.c               ┌────────────────────┐    ┌─────────────────┐
  ┌──────────────┐    ┌──>│ Tensor registry    │───>│ PimTrace        │
  │ PIM-annotated│    │   │ Phase-aware trace  │    │ PimDRAMSystem   │
  │ C source     │────┘   │ translation        │    │ HBM3_PIM DRAM   │
  └──────────────┘        └────────┬───────────┘    │ PIM Controller  │
         │                         │                └────────┬────────┘
    clang + opt              pim_trace.txt              cycle count
    + MemTracePass           (BR/BW/R/W ops)             + stats
```

Three components connected by a file-based interface:

1. **Kernel** (`attention_pim.c`, `axpy_pim.c`, etc.) -- C code with explicit PIM annotations
2. **PIM Runtime** (`pim_runtime.c`) -- intercepts LLVM-instrumented loads/stores,
   translates them to PIM operations
3. **Ramulator2 PIM** -- simulates the PIM trace on a cycle-accurate HBM3 model

---

## Pipeline (6 steps)

```
input.c ─→ LLVM IR ─→ optimized IR ─→ instrumented IR ─→ execute ─→ Ramulator2
         clang -S     opt -O2         MemTracePass       pim_runtime   HBM3_PIM
         -emit-llvm                                      (pim_trace)
```

Run with: `./test_all_pim.sh` (builds MemTracePass + Ramulator2, runs all tests)
or `./run_pim_trace.sh test/attention_pim.c` (single file, no build step).

| Step | Tool | Output |
|------|------|--------|
| 1 | `clang -S -emit-llvm` | `.ll` (unoptimized IR) |
| 2 | `opt -O2` | `.opt.ll` (optimized IR, stack spills eliminated) |
| 3 | `opt --load-pass-plugin=MemTracePass.dylib --passes=mem-trace` | `.instrumented.ll` (loads/stores hooked) |
| 4 | `clang` + link `pim_runtime.c` | executable |
| 5 | Execute | `pim_trace.txt` (PIM operations) |
| 6 | `ramulator2 --config_file config.yaml` | cycle count, request statistics |

---

## HBM3_PIM DRAM Model

### Why a New DRAM Model

Standard HBM3 in Ramulator2 has no concept of:
- **Subarrays** as a hierarchy level (needed for subarray-level parallelism)
- **Processing Elements** in bank groups (MAC units)
- **PIM-specific commands** (bank-read triggers MAC, not a bus transfer)

### Hierarchy

```
Channel ─→ PseudoChannel ─→ BankGroup ─→ Bank ─→ Subarray ─→ Row ─→ Column
  (1)          (2)             (4)        (4)      (64)       (512)   (64)
```

The **subarray** level is the key addition. Standard HBM3 stops at bank.

### Organization (HBM3_8Gb preset)

| Parameter | Value |
|-----------|-------|
| Channels | 1 |
| Pseudo-channels per channel | 2 |
| Bank groups per pseudo-channel | 4 |
| Banks per bank group | 4 |
| Subarrays per bank | 64 |
| Rows per subarray | 512 |
| Columns per row | 64 |
| DQ width | 128 bits |
| PEs per bank group | 2 |
| PE operand width | 16 bits (FP16) |

### Commands

| Command | Scope | Purpose |
|---------|-------|---------|
| `ACT` | row | Activate a row in a subarray |
| `PRE` | subarray | Precharge (close) a subarray |
| `PREASA` | bank | Precharge all subarrays in a bank |
| `PREA` | channel | Precharge all |
| `RD` / `WR` | column | Standard read/write (requires Selected state) |
| `SASEL` | row | Select subarray (Opened -> Selected transition) |
| `SARD` / `SAWR` | column | Subarray-local read/write (only needs Opened) |
| `BKRD` | column | Bank-read: data streams through PE, triggers MAC |
| `BKWR` | column | Bank-write: write result from PE to bank |
| `PIMOp` | row | Explicit PIM compute (closes subarray after) |
| `REFab` / `REFsb` | channel/bank | Refresh commands |

### Subarray State Machine

```
         ACT(row)              SASEL
Closed ──────────→ Opened ──────────→ Selected
  ↑                  │                    │
  │     PRE          │ SARD/SAWR          │ RD/WR/BKRD/BKWR
  └──────────────────┘ PIMOp              │
  └───────────────────────────────────────┘
         PRE / RDA / WRA / PIMOp
```

**Why three states?** Standard DRAM has two bank states (Closed/Opened).
HBM PIM adds Selected because:

- Commands that use the bank I/O path (`RD`, `WR`, `BKRD`, `BKWR`) need the
  subarray to be *selected* (requires `ACT` then `SASEL`)
- Subarray-local commands (`SARD`, `SAWR`, `PIMOp`) only need the row to be
  *opened* (just `ACT`, no `SASEL`)

This distinction models the real Samsung AiM HBM-PIM architecture where the PE
sits between the bank sense amplifier and the global I/O, so data must be
"selected" to reach it.

### Prerequisite Logic

| Command | Subarray State | Prerequisite |
|---------|---------------|-------------|
| BKRD, BKWR, RD, WR | Closed | ACT |
| BKRD, BKWR, RD, WR | Opened (correct row) | SASEL |
| BKRD, BKWR, RD, WR | Opened (wrong row) | PRE |
| BKRD, BKWR, RD, WR | Selected (correct row) | Ready |
| BKRD, BKWR, RD, WR | Selected (wrong row) | PRE |
| SARD, SAWR, SASEL, PIMOp | Closed | ACT |
| SARD, SAWR, SASEL, PIMOp | Opened (correct row) | Ready |
| SARD, SAWR, SASEL, PIMOp | Opened (wrong row) | PRE |

### Timing

Based on HBM3 JEDEC timings at 2 Gbps (tCK = 1 ns). Key parameters:

| Timing | Value (cycles) | Meaning |
|--------|---------------|---------|
| nCL | 7 | CAS latency |
| nRCDRD | 7 | ACT to read |
| nRAS | 17 | ACT to PRE |
| nRC | 19 | ACT to ACT (same subarray) |
| nRP | 7 | PRE to ACT |
| nBL | 4 | Burst length |

Subarray-level timing constraints enforce per-subarray nRC/nRP and
inter-subarray sibling isolation delays.

---

## PIM Runtime Library

### Design Rationale

The LLVM `MemTracePass` instruments all loads and stores in the optimized IR
by inserting calls to `__mem_trace_load(addr, size)` and
`__mem_trace_store(addr, size)`. The standard `trace_runtime.c` emits CPU-centric
`LD`/`ST` traces. The PIM runtime replaces it with PIM-aware translation.

Key design choice: **the same LLVM pass is reused**. We only swap the runtime
library that implements the hooks. This means the instrumentation infrastructure
is shared, and the PIM mapping logic is entirely in userspace C code -- making it
easy to iterate on mapping strategies without touching LLVM.

### API

```c
void pim_init(const char *trace_file);

int  pim_register_tensor(void *ptr, const int *dims, int ndims,
                         int elem_size, pim_role_t role);

void pim_set_phase(pim_phase_t phase);

void pim_finalize(void);
```

**Tensor roles** determine how memory accesses are translated:

| Role | Meaning |
|------|---------|
| `PIM_ROLE_STREAMED` | Bank-resident; load streams through PE (BR) |
| `PIM_ROLE_OPERAND` | Written to PE register via bus (W) |
| `PIM_ROLE_ACCUMULATOR` | Accumulated in PE; load reads (R), store writes (BW) |

**Phases** change the interpretation mode:

| Phase | Load from tensor | Store to tensor |
|-------|-----------------|-----------------|
| `PIM_PHASE_COMPUTE` + STREAMED | BR (bank-read: MAC) | ignored |
| `PIM_PHASE_COMPUTE` + OPERAND | W (write to PE) | ignored |
| `PIM_PHASE_COMPUTE` + ACCUMULATOR | R (read partial sum) | BW (bank-write) |
| `PIM_PHASE_HOST` + any | R (read to host) | W (write from host) |
| `PIM_PHASE_IDLE` | ignored | ignored |

### Bank Mapping

Each registered tensor is assigned to a flat bank (round-robin across channel,
pseudochannel, bank group, and bank):

```
Tensor 0 → ch=0 pch=0 bg=0 bank=0
Tensor 1 → ch=0 pch=0 bg=0 bank=1
...
Tensor 16 → ch=0 pch=1 bg=0 bank=0   (wraps to next pseudochannel)
```

Within a bank, elements are laid out linearly in rows spanning subarrays:

```
values_per_column = DQ_bits / (elem_size * 8)    -- e.g., 128/32 = 4 floats
values_per_row    = num_columns * values_per_column  -- e.g., 64 * 4 = 256

linear_row = base_row + i / values_per_row
subarray   = linear_row / rows_per_subarray
row        = linear_row % rows_per_subarray
col        = (i % values_per_row) / values_per_column
```

Row allocation is **per-bank**: each bank maintains its own row counter. If a
tensor exceeds a bank's capacity (`subarrays × rows_per_subarray`), registration
fails with an error (no silent row wrap).

### Extensibility

The `pim_op_t` enum reserves slots for future PE capabilities:

```c
PIM_OP_MAC   // current: multiply-accumulate
PIM_OP_EXP   // future: exponential in DRAM
PIM_OP_ADD   // future: element-wise add
PIM_OP_DIV   // future: division
```

When EXP becomes available in hardware, the softmax phase can transition from
HOST to a new `PIM_PHASE_EXP` that emits PIM compute operations instead of
host read/write transfers.

---

## Attention Kernel

### Algorithm

Single-head scaled dot-product attention with dimensions:
- `SEQ_LEN = 64`, `D_HEAD = 64`
- Tile size: `BLOCK_M = BLOCK_N = BLOCK_K = 16`

```
Phase 1 (GEMM):  S = Q * K^T          [64x64] = [64x64] × [64x64]^T
Phase 2 (HOST):  P = softmax(S/√d)    [64x64]   (numerically stable)
Phase 3 (GEMM):  O = P * V            [64x64] = [64x64] × [64x64]
```

### PIM Mapping

**Phase 1 -- Q * K^T:**

Q is registered as OPERAND. K is registered as STREAMED. S is ACCUMULATOR.

In the inner loop `acc += Q[i][k] * K[j][k]`:
- Load Q → W (write operand to PE register)
- Load K → BR (data streams through PE, triggers MAC)
- After tile: load S → R, store S → BW (read/write accumulated partial sum)

The local accumulator variable `acc` is in a CPU register, not in any
tensor's memory range, so loads/stores to it are **not traced**. This mirrors
how the PE's internal accumulator works: multiple MACs accumulate without
generating memory traffic.

**Phase 2 -- Softmax (host-side):**

The PE only supports MAC. Softmax requires exp, max, sum, and division.
These are computed on the host:
- Load S → R (read scores from bank to host)
- Store S → W (write softmax result back to bank)

This accurately models the data movement cost of transferring data
between PIM banks and the host for non-MAC operations.

**Phase 3 -- P * V:**

S (now containing softmax output) is still registered as ACCUMULATOR.
V is registered as STREAMED.

- Load S → R (read softmax result; ACCUMULATOR load in COMPUTE = read from bank)
- Load V → BR (STREAMED data through PE, triggers MAC)
- Store O → BW (write final output)

### Trace Statistics (64x64 attention)

| Operation | Count | Source |
|-----------|-------|--------|
| BR (bank-read/MAC) | 327,680 | STREAMED loads in both GEMMs |
| BW (bank-write) | 20,480 | ACCUMULATOR stores in both GEMMs |
| R (read) | 44,032 | ACCUMULATOR reads + softmax reads |
| W (write) | 19,456 | OPERAND loads + softmax writes |
| **Total PIM ops** | **411,648** | |
| **HBM3_PIM cycles** | **~981K** | Simulated by Ramulator2 |

---

## AXPY Kernel

AXPY: `y[i] = a * x[i] + y[i]`. Simple vector operation demonstrating the same
role mapping:

- `x[]` → STREAMED (data streams through PE on load, BR)
- `y[]` → ACCUMULATOR (load R, store BW)
- Scalar `a` → in PE register (not traced; assumed pre-loaded)

Run with `./test_all_pim.sh axpy_pim`. Trace: ~768 ops, ~4K cycles.

---

## Files

### Ramulator2 Modifications (`src/`)

| File | Change |
|------|--------|
| `base/request.h` | Added `op` string field, PIM type enums, new constructor |
| `base/request.cpp` | New constructor: `Request(AddrVec_t, string)` |
| `dram/spec.h` | PIM fields in `Organization` (pe_per_bankgroup, etc.) |
| `dram/lambdas/action.h` | `Subarray::ACT`, `PRE`, `SASEL`; `Bank::PREASA` |
| `dram/lambdas/preq.h` | `Subarray::RequireRowOpen`, `RequireRowSelected` |
| `dram/lambdas/rowhit.h` | `Subarray::RDWR`, `PIM` |
| `dram/lambdas/rowopen.h` | `Subarray::RDWR`, `PIM` |
| `memory_system/memory_system.h` | Added `virtual bool finished()` |

### Ramulator2 New Files (`src/`)

| File | Purpose |
|------|---------|
| `dram/impl/HBM3_PIM.cpp` | HBM3 with subarray level and PIM commands |
| `dram_controller/impl/pim_DRAM_controller.cpp` | PIM-aware controller with priority buffer |
| `memory_system/impl/pim_DRAM_system.cpp` | Routes PIM ops, counts BR/BW/R/W |
| `frontend/impl/memory_trace/pim_trace.cpp` | Parses `<OP> ch,pch,bg,bank,sa,row,col` traces |

### llvm-tracer Files

| File | Purpose |
|------|---------|
| `runtime/pim_runtime.h` | PIM API (init, register, phase, finalize) |
| `runtime/pim_runtime.c` | Tensor registry, bank mapping, phase-aware translation |
| `test/attention_pim.c` | Tiled single-head attention kernel |
| `test/axpy_pim.c` | AXPY kernel (y = a*x + y) |
| `config/hbmpim_config.yaml` | HBM3_PIM simulation configuration |
| `test_all_pim.sh` | Build MemTracePass + Ramulator2, run all tests |
| `run_pim_trace.sh` | Single-file pipeline (no build step) |

---

## Trace Format

Each line: `<OP> <ch>,<pch>,<bg>,<bank>,<sa>,<row>,<col>`

```
W  0,0,0,0,0,0,0       # Write operand to PE register
BR 0,0,0,1,0,0,0       # Bank-read: data through PE, triggers MAC
R  0,0,0,3,0,0,0       # Read accumulated result from bank
BW 0,0,0,3,0,0,15      # Bank-write: store result to bank row
```

All seven address components are explicit physical coordinates -- no address
translation is needed at the simulator side.

---

## Usage

```bash
# From the llvm-tracer directory:
cd ramulator2/llvm-tracer

# Build everything and run all tests (recommended)
./test_all_pim.sh

# Run a single test
./test_all_pim.sh axpy_pim
./test_all_pim.sh attention_pim

# Single-file pipeline (assumes MemTracePass and Ramulator2 are already built)
./run_pim_trace.sh test/attention_pim.c
./run_pim_trace.sh test/attention_pim.c config/hbmpim_config.yaml

# Environment overrides
OPT_LEVEL=O3 PIM_TRACE_OUT=my_trace.txt ./run_pim_trace.sh test/attention_pim.c
```

### Prerequisites

- LLVM/Clang with `opt` (the MemTracePass shared library must be built)
- Ramulator2 built with PIM support (the build picks up new files automatically)

### Writing a New PIM Kernel

```c
#include "pim_runtime.h"

int main(void) {
    pim_init("trace.txt");

    float A[N][K], B[K][M], C[N][M];
    int da[2] = {N, K}, db[2] = {K, M}, dc[2] = {N, M};

    pim_register_tensor(A, da, 2, sizeof(float), PIM_ROLE_OPERAND);
    pim_register_tensor(B, db, 2, sizeof(float), PIM_ROLE_STREAMED);
    pim_register_tensor(C, dc, 2, sizeof(float), PIM_ROLE_ACCUMULATOR);

    pim_set_phase(PIM_PHASE_COMPUTE);
    // ... tiled GEMM with local accumulator ...
    pim_set_phase(PIM_PHASE_IDLE);

    pim_finalize();
    return 0;
}
```

---

## Known Limitations

1. **Channel distribution**: Tensors distribute across ch/pch/bg/bank via
   round-robin. With the default config (1 channel), all stay in channel 0.
   Set `PIM_NUM_CHANNELS` and `PIM_NUM_PCH` to exploit multi-channel.

2. **No drain cycle**: The simulator stops when the last trace is accepted by
   the memory controller. A few final cycles of pending requests are not
   counted in the cycle total. The error is bounded by the controller queue
   depth (~32 requests * ~20 cycles = ~640 cycles out of ~981K).

3. **Operand addressing**: In COMPUTE phase, OPERAND loads emit `W` ops
   addressed to the OPERAND tensor's bank. The real HBM-PIM transfers data
   via the bus to the PE in the STREAMED bank. Ramulator2 models bus
   occupancy regardless of target bank.

4. **Fixed element size**: The prototype uses `float` (4 bytes). Real HBM-PIM
   PEs operate on FP16 (2 bytes). The operation counts and access patterns
   are identical; only the values-per-column ratio changes (4 vs 8).

5. **No cache model**: The trace captures all loads/stores at the DRAM level
   (no L1/L2 filtering). This is appropriate for PIM where data resides in
   DRAM, but overestimates host-side (softmax) traffic if a cache were present.
