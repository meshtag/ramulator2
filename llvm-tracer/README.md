# llvm-tracer: LLVM IR → Ramulator2 Memory Trace Pipeline

Instruments any LLVM IR program to capture load/store memory accesses,
filters them through a configurable cache model, and feeds only DRAM-bound
requests to Ramulator2 for cycle-accurate simulation.

## Overview

```
  .c / .ll          opt -O2        MemTracePass      execute with         ramulator2
  (any code)  ──►  (eliminate  ──► (instrument   ──► cache-filtered  ──► (cycle-accurate
                    stack spills)   loads/stores)     runtime             DRAM sim)
```

The pipeline has five stages:

1. **Compile** - If starting from C/C++, compile to LLVM IR with `-O0 -disable-O0-optnone`.
2. **Optimize** - Run LLVM's standard `-O2` pipeline (configurable to `-O1`, `-O3`, or
   `none`). This promotes stack variables to registers, eliminates redundant loads, and
   applies loop optimizations - removing artificial memory traffic from unoptimized IR
   while preserving real data accesses.
3. **Instrument** - The `MemTracePass` injects a callback before every remaining `load`
   and `store` instruction.
4. **Execute** - The instrumented program runs natively. The runtime library simulates a
   set-associative LRU cache and only writes **cache misses** (+ dirty writebacks) to the
   trace file in Ramulator2's `LoadStoreTrace` format.
5. **Simulate** - Ramulator2 replays the DRAM-only trace through a cycle-accurate model.

## Prerequisites

- LLVM 18+ (`opt`, `clang` in PATH or at `/usr/local/bin/`)
- Ramulator2 built (see parent directory)
- macOS or Linux
- Python 3 + `matplotlib` + `numpy` (only for `benchmark.py`)

## Build

```bash
cd llvm-tracer
make
```

Produces in `build/`:
- `MemTracePass.dylib` - LLVM pass plugin
- `libtrace_runtime.dylib` - runtime library with cache model

## Usage

### Quick start

```bash
export RAMULATOR2=/path/to/ramulator2/build/ramulator2
./run_trace.sh test/matmul.c
```

### Step by step

```bash
# 1. Compile C to LLVM IR
clang -S -emit-llvm -O0 -Xclang -disable-O0-optnone -o input.ll input.c

# 2. Optimize (LLVM standard pipeline)
opt -O2 -S input.ll -o input.opt.ll

# 3. Instrument
opt --load-pass-plugin=./build/MemTracePass.dylib \
    --passes="mem-trace" -S input.opt.ll -o instrumented.ll

# 4. Execute with cache-filtered runtime
clang -O0 instrumented.ll ./build/libtrace_runtime.dylib -o program
RAMULATOR_TRACE_OUT=my_trace.txt ./program

# 5. Run Ramulator2
#    Edit ramulator_config.yaml so that Frontend.path points to your trace,
#    then run:
ramulator2 --config_file ramulator_config.yaml
```

### Starting from LLVM IR directly

```bash
./run_trace.sh my_program.ll
```

## Environment variables

### Pipeline control (`run_trace.sh`)

| Variable | Description | Default |
|---|---|---|
| `RAMULATOR2` | Path to ramulator2 binary | *(skip simulation)* |
| `TRACE_OUTPUT` | Override trace file output path (also sets `RAMULATOR_TRACE_OUT`) | `ramulator_trace.txt` |
| `OPT_LEVEL` | `"O1"`, `"O2"`, `"O3"`, or `"none"` | `O2` |
| `OPT` | Path to `opt` binary | `/usr/local/bin/opt` |
| `CLANG` | Path to `clang` binary | `clang` |

### Cache model (read by the runtime library)

| Variable | Description | Default |
|---|---|---|
| `RAMULATOR_TRACE_OUT` | Output trace file path (used by the runtime) | `ramulator_trace.txt` |
| `CACHE_ENABLE` | `"1"` to filter through cache, `"0"` for raw trace | `1` |
| `CACHE_SIZE_KB` | Total cache size in KB | `1024` (1 MB) |
| `CACHE_ASSOC` | Set associativity | `16` |
| `CACHE_LINE_BYTES` | Cache line size in bytes | `64` |

> **Note:** When using `TRACE_OUTPUT` to write the trace to a custom path, the
> Ramulator2 config (`ramulator_config.yaml`) still needs its `Frontend.path`
> field updated to match. The `benchmark.py` script handles this automatically;
> for manual `run_trace.sh` usage the default path `ramulator_trace.txt` is
> consistent between the script and the config.

### Example configurations

```bash
# Model a 2 MB 16-way LLC
CACHE_SIZE_KB=2048 CACHE_ASSOC=16 ./run_trace.sh test/matmul.c

# Model a 32 KB 8-way L1 (to see what escapes L1)
CACHE_SIZE_KB=32 CACHE_ASSOC=8 ./run_trace.sh test/matmul.c

# Disable cache to see all memory accesses
CACHE_ENABLE=0 ./run_trace.sh test/matmul.c

# Use -O3 for aggressive optimization (loop vectorization, etc.)
OPT_LEVEL=O3 ./run_trace.sh test/matmul.c

# Skip IR optimization to see unoptimized (worst-case) behavior
OPT_LEVEL=none CACHE_ENABLE=0 ./run_trace.sh test/matmul.c
```

## How the cache model works

The runtime implements a **set-associative LRU cache**:

- Each memory access is checked against the cache by tag and set index.
- **Hit**: no trace output (this access would be served by the CPU cache).
- **Miss**: the line is installed in the cache, and a `LD` entry is written to the trace
  (representing the DRAM fill request).
- **Dirty eviction**: when a dirty line is evicted by a miss, an additional `ST` entry
  is written (representing the DRAM writeback).

At program exit, the runtime prints cache statistics to stderr.

## Test workloads

Three sample programs are provided in `test/`:

| File | Description | Memory pattern |
|---|---|---|
| `test/matmul.c` | 64×64 matrix multiplication | Dense, regular stride, high spatial locality |
| `test/stencil.c` | 2D 5-point stencil (256×256 grid, 10 timesteps) | Regular stride, streaming access, larger working set |
| `test/linked_list.c` | Shuffled linked-list traversal (4096 nodes) | Irregular, pointer-chasing, low spatial locality |

## Benchmarking

`benchmark.py` automates running all test workloads across LLVM optimization levels
(`-O0`, `-O1`, `-O2`, `-O3`), collecting Ramulator2 cycle counts, and generating plots.

```bash
python3 benchmark.py
```

This produces three plots:

- **`total_accesses.png`** - Total memory accesses before cache filtering (log-scale).
  Shows the dramatic effect of LLVM optimization on eliminating redundant memory operations.
- **`dram_requests.png`** - DRAM requests after cache filtering (misses + writebacks).
- **`dram_cycles.png`** - Ramulator2 cycle-accurate DRAM simulation cycles.

The script expects `RAMULATOR2` to be set in the environment or defaults to
`../build/ramulator2`.

### Sample results

```
Workload        Opt     Total Accesses       Misses       Cycles       Hit%
----------------------------------------------------------------------------------------
matmul          -O0          3,494,533          769        4,170     99.98%
matmul          -O1            540,672          768        6,888     99.86%
matmul          -O2            534,528          768        6,888     99.86%
matmul          -O3            534,528          768        6,888     99.86%

stencil         -O0         17,970,363        8,161      101,026     99.95%
stencil         -O1          3,291,336        8,160      101,026     99.75%
stencil         -O2          1,108,584        8,160      101,026     99.26%
stencil         -O3          1,108,584        8,160      101,026     99.26%

linked_list     -O0            143,348        1,026       12,437     99.28%
linked_list     -O1             20,480        1,024       12,412     95.00%
linked_list     -O2             20,480        1,024       12,412     95.00%
linked_list     -O3             20,480        1,024       12,412     95.00%
```

**Key observation:** LLVM optimization dramatically reduces total memory accesses
(e.g. 16× for stencil at `-O2` vs `-O0`), but DRAM requests and cycles remain
nearly identical. This is because the extra accesses in unoptimized code are stack
spills and redundant loads to already-cached locations - the cache absorbs them
entirely. The actual data working set (and thus compulsory cold misses) is the same
regardless of optimization level.

## Impact of optimizations

Using the 64×64 matrix multiply test (`test/matmul.c`):

| Configuration | Trace entries | What they represent |
|---|---|---|
| Raw (no opt, no cache) | 3,494,533 | Every IR-level load/store including stack spills |
| Optimized IR (-O2), no cache | 534,528 | Real data accesses only (6.5× reduction) |
| Optimized IR (-O2) + 1MB cache | 768 | DRAM requests only (4,550× reduction) |

The 768 DRAM requests represent the cold misses to fill the three 16 KB arrays
(A, B, C) into cache - 48 KB / 64 B per line = 768 lines. This matches the
theoretical minimum.

## Trace format

Output is Ramulator2's `LoadStoreTrace` format:

```
LD 0x1047cc000
LD 0x1047cc040
ST 0x1047d0000
...
```

## Ramulator2 output

After simulation, Ramulator2 reports:

- `memory_system_cycles` - total DRAM cycles
- `total_num_read_requests` / `total_num_write_requests`
- `row_hits`, `row_misses`, `row_conflicts`
- `avg_read_latency`
