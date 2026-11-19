# llvm-tracer

The LLVM passes and C runtimes that turn an instrumented TritonIM kernel into a DRAM command
trace.

- `src/MemTracePass.cpp` and `src/ComputeTracePass.cpp` instrument the kernel's LLVM IR. `make`
  builds them into `build/`, as `benchmarks/cluster/build_passes.sh` does on the cluster.
- `runtime/` holds the runtimes that `benchmarks/pipeline.py` compiles and links with each
  kernel: `pim_runtime.c` for HBM-PIM and `simdram_runtime.c` for SIMDRAM, which share
  `im_trace_common.h`, with `im_runtime.c` and `im_addr_dedup.c`.

The TritonPIM superproject's `docs/CONTEXT_MAP.md` (section 3.2) describes the flow from a
kernel to a cycle count.
