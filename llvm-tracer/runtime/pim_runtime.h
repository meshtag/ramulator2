#ifndef PIM_RUNTIME_H
#define PIM_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Tensor roles: data-flow semantics for PIM operations */
typedef enum {
  PIM_ROLE_STREAMED,   /* Bank-resident; load streams through PE (BR) */
  PIM_ROLE_OPERAND,    /* Written to PE register via bus (W) */
  PIM_ROLE_ACCUMULATOR /* Accumulated in PE; load reads (R), store writes (BW)
                        */
} pim_role_t;

/* Computation phases */
typedef enum {
  PIM_PHASE_IDLE,    /* No PIM tracing (initialization, etc.) */
  PIM_PHASE_COMPUTE, /* PIM compute: loads/stores map to PE operations */
  PIM_PHASE_HOST,    /* Host-side compute (softmax etc.): read/write via bus */
} pim_phase_t;

/* PE capabilities - extensible for future PE designs */
typedef enum {
  PIM_OP_MAC = 0, /* Multiply-accumulate (current HBM-PIM) */
  PIM_OP_EXP,     /* Exponential (future) */
  PIM_OP_ADD,     /* Element-wise add (future) */
  PIM_OP_DIV,     /* Division (future) */
  PIM_OP_COUNT
} pim_op_t;

/* Reuse-as-layout: the physical layout/residency class decided by the
 * compiler's im-operand-residency-layout pass (the explicit, heuristic
 * analog of OptiPIM's DataLayout MILP) and pushed in via
 * pim_set_tensor_layout(). UNSET (=0) is the zero-initialized default for
 * every registered tensor and means "no compiler layout decision" — the
 * runtime maps elements and accounts for reuse exactly as before. The
 * non-UNSET kinds mirror the pass's reuse_class values. */
typedef enum {
  PIM_LAYOUT_KIND_UNSET = 0,   /* default: legacy behavior, no change */
  PIM_LAYOUT_KIND_RESIDENT,    /* stationary operand: read-once, bank-resident */
  PIM_LAYOUT_KIND_BANK_SPREAD, /* parallel output dim spread across the banks */
  PIM_LAYOUT_KIND_ROW_DUP,     /* broadcast operand replicated per bank-group */
  PIM_LAYOUT_KIND_LEADER       /* represented on a single leader bank */
} pim_layout_kind_t;

/* Initialize the PIM runtime, open the trace file */
void pim_init(const char *trace_file);

/*
 * Register a tensor with the PIM runtime.
 *   ptr       - base address of the tensor in memory
 *   dims      - array of dimension sizes (e.g., {64, 64})
 *   ndims     - number of dimensions
 *   elem_size - size of each element in bytes (e.g., sizeof(float))
 *   role      - PIM_ROLE_STREAMED, PIM_ROLE_OPERAND, or PIM_ROLE_ACCUMULATOR
 *
 * Returns a tensor id (>= 0) on success, -1 on failure.
 */
int pim_register_tensor(void *ptr, const int *dims, int ndims, int elem_size,
                        pim_role_t role);

/* Set the current computation phase */
void pim_set_phase(pim_phase_t phase);

/* Finalize: close trace file and print statistics */
void pim_finalize(void);

/* Called by the LLVM instrumentation pass (same ABI as trace_runtime) */
void __mem_trace_init(void);
void __mem_trace_fini(void);
void __mem_trace_load(void *addr, uint64_t size);
void __mem_trace_store(void *addr, uint64_t size);

/* Axis-wise persistent-scope load entry points emitted by MemTracePass. The
 * suffix names the axes the load is INVARIANT in (i.e. axes whose change does
 * NOT reset the dedup state). All four share the trace-emission shape of
 * __mem_trace_load; only the underlying dedup scope differs:
 *
 *   __pim_load_persistent     — invariant in {x,y,z}: reset only on phase
 *                               change. Pointer doesn't depend on any pid.
 *   __pim_load_persistent_yz  — invariant in {y,z}: reset on pid_x change.
 *                               Pointer depends on pid_x only.
 *   __pim_load_persistent_xz  — invariant in {x,z}: reset on pid_y change.
 *                               Pointer depends on pid_y only.
 *   __pim_load_persistent_xy  — invariant in {x,y}: reset on pid_z change.
 *                               Pointer depends on pid_z only.
 *
 * Loads whose pointer depends on multiple program-id axes are routed to
 * __mem_trace_load (per-program-id reset) by the compiler-side classifier. */
void __pim_load_persistent(void *addr, uint64_t size);
void __pim_load_persistent_yz(void *addr, uint64_t size);
void __pim_load_persistent_xz(void *addr, uint64_t size);
void __pim_load_persistent_xy(void *addr, uint64_t size);

/* ================================================================
 *  Per-tensor broadcast-scalar hint
 *
 *  Tells the runtime that accesses to a registered tensor are
 *  "broadcast scalar" reads — every bank reads the SAME physical
 *  address per access, the value is broadcast on the bus once, and
 *  subsequent re-reads of the same address (across program-id
 *  boundaries, K-loop iterations, etc.) hit the PE's internal
 *  register cache rather than re-broadcasting.
 *
 *  When set:
 *    - Every access to the tensor routes through the xyz-invariant
 *      persistent dedup state (g_dedup_persistent), regardless of
 *      which entry point the LLVM classifier chose.
 *    - Each unique physical (bank, row, col) tuple emits at most
 *      ONCE for the entire COMPUTE phase. Repeated accesses to the
 *      same tuple are suppressed (dedup hit).
 *
 *  Correctness invariant: this models a hardware PE-register cache
 *  that retains broadcast operand values for the COMPUTE phase. It
 *  does NOT collapse distinct physical tuples — different program-id
 *  values that produce different addresses (e.g., conv2d weight
 *  varying with pid_co) still emit each distinct tuple once. The flag
 *  is safe to set on any tensor whose kernel-side access pattern is a
 *  scalar broadcast (Triton scalar tl.load() with no per-bank
 *  variation in the address).
 *
 *  Examples of correct use:
 *    - matvec x:        scalar tl.load(x + k)
 *    - matmul row-tiled A: scalar tl.load(A + pid_row*K + k)
 *    - conv2d Weight:   scalar tl.load(w_ptr)
 *
 *  Examples where it would be UNSOUND (don't set):
 *    - matmul B (vector tl.load — different banks read different cols)
 *    - conv2d Input (vector tl.load with mask)
 *    - any tensor where bank-id leaks into the address chain. */
void pim_set_tensor_broadcast_scalar(int tensor_id, int on);

/* ================================================================
 *  Per-tensor reuse-as-layout descriptor (compiler-decided)
 *
 *  Pushed in by benchmarks/pass_ablation.py from the per-operand
 *  decisions stamped by the im-operand-residency-layout MLIR pass
 *  (carried in ccinfo.metadata, keyed by the kernel pointer-arg index,
 *  which maps 1:1 to the registered tensor id since pointer args are
 *  registered first and in order). This is the explicit, compiler-side
 *  realization of operand reuse — the alternative to recovering it
 *  post-hoc from trace dedup, and the basis for an apples-to-apples
 *  layout comparison against OptiPIM.
 *
 *    layout_kind        - pim_layout_kind_t (UNSET => no change)
 *    reduction_col_axis - tensor axis to place on the column-low bits
 *                         (the contraction dim), or -1 if none
 *    bank_spread_mask   - bitmask of tensor axes spread across the banks
 *    resident_capacity  - resident working-set budget in tuples; 0 means
 *                         unbounded (current behavior). A later increment
 *                         charges reuse beyond this budget as a re-fetch.
 *
 *  ABI-only at this stage: recording these fields changes no emitted
 *  trace until a later increment teaches map_element and the dedup
 *  accounting to honor them. Absent this call (e.g. an older runtime
 *  build) every tensor stays UNSET, so existing traces are unaffected. */
void pim_set_tensor_layout(int tensor_id, int layout_kind,
                           int reduction_col_axis, uint32_t bank_spread_mask,
                           int resident_capacity);

/* Compiler-honored reduction-to-column layout for one tensor: transpose its
 * contraction axis onto the column-low address bits (stride=N, extent=K). Both
 * must be pow2 > 1 to take effect. ABI-additive; absent (older build) the
 * tensor stays unremapped. Correctness-invariant (pure trace-address remap). */
void pim_set_tensor_redcol(int tensor_id, int redcol_stride, int redcol_extent);

#ifdef __cplusplus
}
#endif

#endif /* PIM_RUNTIME_H */
