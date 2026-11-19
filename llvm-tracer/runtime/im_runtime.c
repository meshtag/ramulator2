/*
 * im_runtime.c — IM execution-model runtime implementation
 *
 * Stores the current bank-ID and program-ID in file-scope globals.
 * Each kernel invocation reads these via the __pim_get_*() calls that
 * the Triton IM backend emits as extern function calls in the LLVM IR.
 *
 * The replay loop (pipeline.trace):
 *   for pid_z: __pim_set_program_id_z(pid_z)
 *     for pid_y: __pim_set_program_id_y(pid_y)
 *       for pid: __pim_set_program_id(pid)
 *         for bank: __pim_set_bank_id(bank), kernel(args...)
 *
 * The program ids are set once per instance, outside the bank loop, so the epoch stays the
 * same across the banks of one instance. pim_runtime's lockstep collapse relies on that,
 * since it folds only the replicas that share an epoch.
 *
 * No thread-local storage is needed because pipeline.py calls the
 * kernel sequentially.
 */

#include "im_runtime.h"

/* File-scope state — one bank executes at a time on CPU. */
static int32_t current_bank_id = 0;
static int32_t current_program_id = 0;
static int32_t current_program_id_y = 0;
static int32_t current_program_id_z = 0;

/* Program-instance epoch. Bumped by every program-id setter, i.e. exactly when the
 * host starts a new instance, and NOT by the bank setter, since the banks of one
 * instance are one all-bank command. pim_runtime keys its lockstep collapse on it, and
 * simdram_runtime writes out and resets its streamed slabs when it changes. */
uint64_t __pim_program_epoch = 0;

/* Tile index within ONE bank replay, for persistent kernels.
 *
 * A persistent kernel puts the tile loop inside the kernel, so the bank loop is
 * OUTSIDE: bank 0 walks every tile before bank 1 starts. The lockstep collapse has to
 * fold the 32 replays of one tile and NOT fold two tiles, so the dispatch id must be
 * the same for bank 0 tile 5 and bank 1 tile 5. A free-running counter would not be.
 * So this resets when the host selects a bank and advances on each tile boundary the
 * kernel signals, which makes it a tile INDEX rather than a running total. */
static uint64_t current_tile_idx = 0;



/* ---- Called from inside the compiled kernel ---- */

int32_t __pim_get_bank_id(void) { return current_bank_id; }

int32_t __pim_get_program_id(void) { return current_program_id; }

int32_t __pim_get_program_id_y(void) { return current_program_id_y; }

int32_t __pim_get_program_id_z(void) { return current_program_id_z; }

/* ---- Called by the host driver before each kernel invocation ---- */

void __pim_set_bank_id(int32_t id) {
  current_bank_id = id;
  current_tile_idx = 0; /* a new replay walks the same tiles again */
}


void __pim_set_program_id(int32_t id) {
  current_program_id = id;
  __pim_program_epoch++;
}

void __pim_set_program_id_y(int32_t id) {
  current_program_id_y = id;
  __pim_program_epoch++;
}

void __pim_set_program_id_z(int32_t id) {
  current_program_id_z = id;
  __pim_program_epoch++;
}

/* Called by the kernel at the end of one tile of a persistent loop. */
void __pim_tile_boundary(void) { current_tile_idx++; }

/* Tile index within the current bank replay, which pim_runtime packs with the program
 * epoch into its dispatch id. */
uint64_t __pim_tile_index(void) { return current_tile_idx; }
