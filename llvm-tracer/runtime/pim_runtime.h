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

/* Computation phases. pim_set_phase refuses any other value. */
typedef enum {
  PIM_PHASE_IDLE,    /* No PIM tracing (initialization, etc.) */
  PIM_PHASE_COMPUTE, /* PIM compute: loads/stores map to PE operations */
} pim_phase_t;

/* Initialize the PIM runtime, open the trace file. Exits unless PIM_TRACE_FORMAT=dcc. */
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

/* Called by the LLVM instrumentation pass */
void __mem_trace_load(void *addr, uint64_t size, uint64_t lanes);
void __mem_trace_store(void *addr, uint64_t size, uint64_t lanes);

#ifdef __cplusplus
}
#endif

#endif /* PIM_RUNTIME_H */
