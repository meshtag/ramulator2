/*
 * im_runtime.h — IM execution-model runtime
 *
 * Provides the bank-ID and program-ID primitives that Triton's IM backend
 * emits as extern calls (__pim_get_bank_id / __pim_get_program_id).
 *
 * Two consumers:
 *   1. CPU functional verification  — verify_im_cpu.py sets these before
 *      each kernel invocation to emulate multi-bank SIMD execution.
 *   2. PIM trace generation — the trace harness iterates over banks,
 *      calling the kernel once per bank to collect per-bank memory traces.
 *
 * The "set" functions are called by the host driver; the "get" functions
 * are called from inside the compiled kernel.
 */

#ifndef IM_RUNTIME_H
#define IM_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Bank ID (which PIM bank is executing) --- */
int32_t __pim_get_bank_id(void);
void __pim_set_bank_id(int32_t id);

/* --- Program ID (which tile / block of the input this invocation covers) ---
 * Axis 0 (X) is the primary dimension; Y and Z are optional for 2D/3D grids.
 */
int32_t __pim_get_program_id(void);
void __pim_set_program_id(int32_t id);

int32_t __pim_get_program_id_y(void);
void __pim_set_program_id_y(int32_t id);

int32_t __pim_get_program_id_z(void);
void __pim_set_program_id_z(int32_t id);

#ifdef __cplusplus
}
#endif

#endif /* IM_RUNTIME_H */
