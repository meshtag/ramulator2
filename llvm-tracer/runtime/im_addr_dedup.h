#ifndef IM_ADDR_DEDUP_H
#define IM_ADDR_DEDUP_H

#include <stddef.h>
#include <stdint.h>

/*
 * im_addr_dedup: open-addressing hash set of (key1, key2, key3) tuples, packed to
 * 16, 32 and 16 bits. The PIM trace emitter uses it to collapse redundant accesses
 * into one trace line per unique key.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct addr_dedup_state addr_dedup_state_t;

/* Returns NULL on alloc failure. `max_entries` only sizes the INITIAL table; it is
 * not a bound. The table doubles whenever the live entries would pass 50% load, so a
 * low hint costs a few rehashes, never dedup accuracy. */
addr_dedup_state_t *addr_dedup_create(int max_entries);

/* Returns 1 if the (k1, k2, k3) tuple is new (caller should emit), 0 if already seen
 * (caller should skip).
 *
 * Only if growth fails (allocation refused) does a full table fall back to
 * returning 1 (treat as new): over-emit, never under-emit. That case is counted by
 * addr_dedup_saturations and the runtime prints it, because it silently degrades
 * every collapse built on this table. */
int addr_dedup_check_and_mark(addr_dedup_state_t *s, uint64_t k1, uint64_t k2,
                              uint64_t k3);

/* Claims that found no room and no growth. Must be 0; nonzero means the collapse
 * built on this table stopped collapsing partway through and the trace over-emits. */
uint64_t addr_dedup_saturations(const addr_dedup_state_t *s);

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

#ifdef __cplusplus
}
#endif

#endif /* IM_ADDR_DEDUP_H */
