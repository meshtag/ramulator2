/*
 * im_runtime.h — IM execution-model runtime
 *
 * Provides the bank-ID and program-ID primitives that Triton's IM backend
 * emits as extern calls (__pim_get_bank_id / __pim_get_program_id).
 *
 * benchmarks/pipeline.py compiles this file with every kernel. Its correctness
 * check and its trace run both call the kernel once per (program, bank) to emulate
 * multi-bank SIMD execution, setting the program ids once per instance and the bank id
 * before each call.
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
/* Bumped by each __pim_set_program_id* call. A trace runtime watches this rather
 * than comparing program ids, so the per-instance reset does not depend on the
 * order the host walks the grid. */
extern uint64_t __pim_program_epoch;

/* Signalled by a persistent kernel at each tile boundary; resets when the host picks
 * a bank, so it is a tile index that matches across bank replays. */
void __pim_tile_boundary(void);

/* Tile index within the current bank replay; 0 for a gridded kernel. The trace
 * runtime packs this with __pim_program_epoch into its collapse key. */
uint64_t __pim_tile_index(void);


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
