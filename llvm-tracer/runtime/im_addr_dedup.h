#ifndef IM_ADDR_DEDUP_H
#define IM_ADDR_DEDUP_H

#include <stddef.h>
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

/* Returns NULL on alloc failure. `max_entries` only sizes the INITIAL table; it is
 * not a bound. The table doubles whenever the live entries at the current epoch
 * would pass 50% load, so a low hint costs a few rehashes, never dedup accuracy. */
addr_dedup_state_t *addr_dedup_create(int max_entries);

/* Returns 1 if the (k1, k2, k3) tuple is new in the current epoch (caller
 * should emit), 0 if already seen this epoch (caller should skip).
 *
 * Only if growth fails (allocation refused) does a full table fall back to
 * returning 1 (treat as new): over-emit, never under-emit. That case is counted by
 * addr_dedup_saturations and both runtimes print it, because it silently degrades
 * every collapse built on this table. */
int addr_dedup_check_and_mark(addr_dedup_state_t *s, uint64_t k1, uint64_t k2,
                              uint64_t k3);

/* Logically clears the table by bumping the internal epoch counter. */
void addr_dedup_reset(addr_dedup_state_t *s);

/* Number of times check_and_mark returned 0 since creation. */
uint64_t addr_dedup_hits(const addr_dedup_state_t *s);

/* Claims that found no room and no growth. Must be 0; nonzero means the collapse
 * built on this table stopped collapsing partway through and the trace over-emits. */
uint64_t addr_dedup_saturations(const addr_dedup_state_t *s);

/* Current slot count, after any growth. */
int addr_dedup_capacity(const addr_dedup_state_t *s);

void addr_dedup_destroy(addr_dedup_state_t *s);

/* (lane, elem) -> slab slot for lane-placed tensors. Keyed on BOTH because a
 * partitioned tensor's elements are not disjoint per lane: a conv halo element is
 * consumed by two lanes and must be resident in both banks, so it owns a slot in each
 * slab. Keyed on the element alone it reused lane 0's slot numbers in lane 1 and
 * aliased two elements to one address. Open addressing, doubles on load; a dense
 * [lane][elem] table is 2 GB on the large matmul B. */
typedef struct slot_map slot_map_t;
slot_map_t *slot_map_create(size_t hint);
void slot_map_destroy(slot_map_t *m);
/* Slot for (lane, elem); inserts `fresh` when absent and sets *inserted to 1. Returns
 * -1 only when growth fails, and the caller must then leave the tuple as mapped. */
int32_t slot_map_get_or_put(slot_map_t *m, int lane, int elem, int32_t fresh,
                            int *inserted);
/* Test-and-set one marker bit (0..30) on an EXISTING (lane, elem) entry. Returns the
 * previous state, or -1 if the entry is absent. Used for "already read out": the
 * accumulator is loop-carried, so its partial sum is resident after the first touch. */
int slot_map_test_and_set(slot_map_t *m, int lane, int elem, int bit);
/* Forget every entry, keeping the allocation. */
void slot_map_clear(slot_map_t *m);
/* Slot for (lane, elem) if present, else -1. Never inserts. */
int32_t slot_map_peek(const slot_map_t *m, int lane, int elem);

#ifdef __cplusplus
}
#endif

#endif /* IM_ADDR_DEDUP_H */
