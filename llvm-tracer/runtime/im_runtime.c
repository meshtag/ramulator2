/*
 * im_runtime.c — IM execution-model runtime implementation
 *
 * Stores the current bank-ID and program-ID in file-scope globals.
 * Each kernel invocation reads these via the __pim_get_*() calls that
 * the Triton IM backend emits as extern function calls in the LLVM IR.
 *
 * Sequential (CPU) execution model:
 *   for pid in 0 .. num_programs-1:
 *       for bank in 0 .. num_banks-1:         // parallel on real HBM-PIM
 *           __pim_set_program_id(pid);
 *           __pim_set_bank_id(bank);
 *           kernel(args...);
 *
 * No thread-local storage is needed because the CPU verification and
 * trace-generation harnesses call the kernel sequentially.
 */

#include "im_runtime.h"

/* File-scope state — one bank executes at a time on CPU. */
static int32_t current_bank_id = 0;
static int32_t current_program_id = 0;

/* ---- Called from inside the compiled kernel ---- */

int32_t __pim_get_bank_id(void) { return current_bank_id; }

int32_t __pim_get_program_id(void) { return current_program_id; }

/* ---- Called by the host driver before each kernel invocation ---- */

void __pim_set_bank_id(int32_t id) { current_bank_id = id; }

void __pim_set_program_id(int32_t id) { current_program_id = id; }
