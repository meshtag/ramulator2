#include "im_addr_dedup.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  uint64_t key;
  uint32_t epoch;
} dedup_slot_t;

struct addr_dedup_state {
  dedup_slot_t *slots;
  int capacity;       /* power of two */
  int mask;           /* capacity - 1 */
  uint32_t cur_epoch; /* 0 reserved as "empty"/uninitialized */
  uint64_t hits;
};

/* Pack three keys into one uint64. Layout (LSB→MSB):
 *   key1: 16 bits  (global_bank, max ~64k)
 *   key2: 32 bits  (sa*rows + row + optional bit, max ~4G)
 *   key3: 16 bits  (col)
 * Total: 64 bits. */
static inline uint64_t pack_keys(uint64_t k1, uint64_t k2, uint64_t k3) {
  return (k1 & 0xFFFFULL) | ((k2 & 0xFFFFFFFFULL) << 16) |
         ((k3 & 0xFFFFULL) << 48);
}

/* SplitMix64 finalizer — strong avalanche for 64→64 bit hashing. */
static inline uint64_t mix64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xFF51AFD7ED558CCDULL;
  x ^= x >> 33;
  x *= 0xC4CEB9FE1A85EC53ULL;
  x ^= x >> 33;
  return x;
}

addr_dedup_state_t *addr_dedup_create(int max_entries) {
  if (max_entries < 16)
    max_entries = 16;
  int capacity = 1;
  while (capacity < max_entries * 2)
    capacity <<= 1; /* 50% target load */

  addr_dedup_state_t *s =
      (addr_dedup_state_t *)malloc(sizeof(addr_dedup_state_t));
  if (!s)
    return NULL;
  s->slots = (dedup_slot_t *)calloc((size_t)capacity, sizeof(dedup_slot_t));
  if (!s->slots) {
    free(s);
    return NULL;
  }
  s->capacity = capacity;
  s->mask = capacity - 1;
  s->cur_epoch = 1; /* epoch 0 reserved as "empty" */
  s->hits = 0;
  return s;
}

int addr_dedup_check_and_mark(addr_dedup_state_t *s, uint64_t k1, uint64_t k2,
                              uint64_t k3) {
  if (!s)
    return 1; /* no dedup state — always emit */

  uint64_t key = pack_keys(k1, k2, k3);
  uint64_t h = mix64(key);
  int idx = (int)(h & (uint64_t)s->mask);

  for (int probe = 0; probe < s->capacity; probe++) {
    dedup_slot_t *slot = &s->slots[idx];
    if (slot->epoch == s->cur_epoch) {
      if (slot->key == key) {
        s->hits++;
        return 0; /* already seen this epoch */
      }
      /* live collision — keep probing */
    } else {
      /* empty or stale slot — claim it for this epoch */
      slot->key = key;
      slot->epoch = s->cur_epoch;
      return 1;
    }
    idx = (idx + 1) & s->mask;
  }

  /* Table fully occupied at current epoch — fall back to "new" (over-emit
   * but never under-emit). In practice the 50% load target keeps us far
   * from this case for any realistic per-program-id access count. */
  return 1;
}

void addr_dedup_reset(addr_dedup_state_t *s) {
  if (!s)
    return;
  s->cur_epoch++;
  if (s->cur_epoch == 0) {
    /* wraparound: physically wipe and restart epoch counter */
    memset(s->slots, 0, (size_t)s->capacity * sizeof(dedup_slot_t));
    s->cur_epoch = 1;
  }
}

uint64_t addr_dedup_hits(const addr_dedup_state_t *s) {
  return s ? s->hits : 0;
}

void addr_dedup_destroy(addr_dedup_state_t *s) {
  if (!s)
    return;
  free(s->slots);
  free(s);
}
