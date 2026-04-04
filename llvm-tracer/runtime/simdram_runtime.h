#ifndef SIMDRAM_RUNTIME_H
#define SIMDRAM_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Tensor roles — same semantics as PIM runtime, different trace mapping */
typedef enum {
  SIMDRAM_ROLE_STREAMED,   /* Input: written to subarray bit-rows (W) */
  SIMDRAM_ROLE_OPERAND,    /* Weight: pre-stored in DRAM, no trace emission */
  SIMDRAM_ROLE_ACCUMULATOR /* Output: read (R) for partial sums; compute via
                              MAJ gates (BR) */
} simdram_role_t;

/* Computation phases */
typedef enum {
  SIMDRAM_PHASE_IDLE,    /* No tracing */
  SIMDRAM_PHASE_COMPUTE, /* SIMDRAM compute: loads/stores → W/R, arith → BR */
  SIMDRAM_PHASE_HOST     /* Host-side: loads → R, stores → W */
} simdram_phase_t;

/* Arithmetic opcodes — must match ComputeTracePass.cpp SimdramOp enum */
typedef enum {
  SIMDRAM_OP_ADD = 0,
  SIMDRAM_OP_SUB,
  SIMDRAM_OP_MUL,
  SIMDRAM_OP_DIV,
  SIMDRAM_OP_REM,
  SIMDRAM_OP_AND,
  SIMDRAM_OP_OR,
  SIMDRAM_OP_XOR,
  SIMDRAM_OP_SHL,
  SIMDRAM_OP_LSHR,
  SIMDRAM_OP_ASHR,
  SIMDRAM_OP_FADD,
  SIMDRAM_OP_FSUB,
  SIMDRAM_OP_FMUL,
  SIMDRAM_OP_FDIV,
  SIMDRAM_OP_FREM,
  SIMDRAM_OP_OTHER,
  SIMDRAM_OP_COUNT
} simdram_op_t;

/* Initialize the SIMDRAM runtime, open the trace file */
void simdram_init(const char *trace_file);

/*
 * Register a tensor with the SIMDRAM runtime.
 *   ptr       - base address of the tensor in memory
 *   dims      - array of dimension sizes (e.g., {64, 64})
 *   ndims     - number of dimensions
 *   elem_size - size of each element in bytes
 *   role      - SIMDRAM_ROLE_STREAMED, SIMDRAM_ROLE_OPERAND,
 *               or SIMDRAM_ROLE_ACCUMULATOR
 *
 * Returns a tensor id (>= 0) on success, -1 on failure.
 */
int simdram_register_tensor(void *ptr, const int *dims, int ndims,
                            int elem_size, simdram_role_t role);

/* Set the current computation phase */
void simdram_set_phase(simdram_phase_t phase);

/* Finalize: close trace file and print statistics */
void simdram_finalize(void);

/* Called by MemTracePass instrumentation (same ABI as pim_runtime) */
void __mem_trace_init(void);
void __mem_trace_fini(void);
void __mem_trace_load(void *addr, uint64_t size);
void __mem_trace_store(void *addr, uint64_t size);

/* Called by ComputeTracePass instrumentation */
void __compute_trace(int32_t opcode, int32_t bit_width);

#ifdef __cplusplus
}
#endif

#endif /* SIMDRAM_RUNTIME_H */
