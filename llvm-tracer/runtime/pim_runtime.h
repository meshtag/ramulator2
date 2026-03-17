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

#ifdef __cplusplus
}
#endif

#endif /* PIM_RUNTIME_H */
