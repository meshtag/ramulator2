#ifndef IM_ADDR_DEDUP_H
#define IM_ADDR_DEDUP_H

#include <stdint.h>

/*
 * im_addr_dedup — open-addressing hashmap that tracks which physical
 * (key1, key2, key3) tuples have been emitted within the current "epoch".
 *
 * Used by PIM/SIMDRAM trace emitters to collapse spatially-redundant
 * accesses (e.g. multiple PEs in a bank-group reading the same row buffer)
 * into one trace line per unique physical address — matching OptiPIM's
 * codegen_no_duplicate behavior, but emitted at runtime by a TritonIM-
 * native primitive that does not depend on OptiPIM internals.
 *
 * Keying conventions:
 *   PIM:     key1 = global_bank, key2 = sa*num_rows + row, key3 = col
 *   SIMDRAM: key1 = global_bank, key2 = sa*num_rows + row + bit, key3 = col
 *
 * `addr_dedup_reset` is O(1) — it bumps an internal epoch counter so that
 * stale entries are treated as empty on the next probe.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct addr_dedup_state addr_dedup_state_t;

/* Returns NULL on alloc failure. `max_entries` is a soft hint; the actual
 * capacity is the next power of two that gives <= 50% load factor. */
addr_dedup_state_t *addr_dedup_create(int max_entries);

/* Returns 1 if the (k1, k2, k3) tuple is new in the current epoch (caller
 * should emit), 0 if already seen this epoch (caller should skip).
 *
 * If the table fills up at the current epoch, falls back to returning 1
 * (treat as new). This is conservative — we may over-emit but never under-
 * emit (correctness preserved at the cost of dedup effectiveness). */
int addr_dedup_check_and_mark(addr_dedup_state_t *s, uint64_t k1, uint64_t k2,
                              uint64_t k3);

/* Logically clears the table by bumping the internal epoch counter. */
void addr_dedup_reset(addr_dedup_state_t *s);

/* Number of times check_and_mark returned 0 since creation. */
uint64_t addr_dedup_hits(const addr_dedup_state_t *s);

void addr_dedup_destroy(addr_dedup_state_t *s);

#ifdef __cplusplus
}
#endif

#endif /* IM_ADDR_DEDUP_H */
