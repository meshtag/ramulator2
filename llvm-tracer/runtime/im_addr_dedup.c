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
  int live;           /* slots claimed at cur_epoch */
  uint64_t grows;     /* doublings; a nonzero value means the create() hint was low */
  uint64_t saturations; /* claims made with no room left: the caller over-emits */
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
  s->live = 0;
  s->grows = 0;
  s->saturations = 0;
  return s;
}

/* Double the table, rehashing only the entries live at cur_epoch (older epochs are
 * logically empty). Returns 0 if the allocation fails, leaving the table untouched.
 *
 * WHY THIS EXISTS: the capacity used to be fixed at create() from a caller hint, and
 * a full table falls back to "treat as new". The lockstep hint was sized to the PE
 * register file, which has nothing to do with how many distinct addresses one
 * program-id touches, so on large tiles the table filled partway through and the
 * collapse silently stopped collapsing. On matmul 128x3072x768 that inflated bank
 * reads 1,769,472 -> 2,668,032 and cycles 14,545,807 -> 21,150,357 (2026-09-10). */
static int addr_dedup_grow(addr_dedup_state_t *s) {
  int newcap = s->capacity * 2;
  if (newcap <= s->capacity)
    return 0; /* int overflow */
  dedup_slot_t *ns = (dedup_slot_t *)calloc((size_t)newcap, sizeof(dedup_slot_t));
  if (!ns)
    return 0;
  int newmask = newcap - 1;
  for (int i = 0; i < s->capacity; i++) {
    if (s->slots[i].epoch != s->cur_epoch)
      continue;
    uint64_t k = s->slots[i].key;
    int idx = (int)(mix64(k) & (uint64_t)newmask);
    while (ns[idx].epoch == s->cur_epoch)
      idx = (idx + 1) & newmask;
    ns[idx].key = k;
    ns[idx].epoch = s->cur_epoch;
  }
  free(s->slots);
  s->slots = ns;
  s->capacity = newcap;
  s->mask = newmask;
  s->grows++;
  return 1;
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
      /* empty or stale slot — claim it, growing first if we are at the load target */
      if (s->live + 1 > s->capacity / 2 && addr_dedup_grow(s)) {
        idx = (int)(h & (uint64_t)s->mask);
        for (;;) {
          dedup_slot_t *g = &s->slots[idx];
          if (g->epoch == s->cur_epoch) {
            if (g->key == key) { /* cannot happen: we probed to a free slot */
              s->hits++;
              return 0;
            }
            idx = (idx + 1) & s->mask;
            continue;
          }
          g->key = key;
          g->epoch = s->cur_epoch;
          s->live++;
          return 1;
        }
      }
      slot->key = key;
      slot->epoch = s->cur_epoch;
      s->live++;
      return 1;
    }
    idx = (idx + 1) & s->mask;
  }

  /* No room and growth failed (allocation refused). Fall back to "new": over-emit,
   * never under-emit. Counted, and both runtimes print the count, because a silent
   * fallback here reads as a faithful trace while the collapse has stopped working. */
  s->saturations++;
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
  s->live = 0;
}

uint64_t addr_dedup_hits(const addr_dedup_state_t *s) {
  return s ? s->hits : 0;
}

uint64_t addr_dedup_saturations(const addr_dedup_state_t *s) {
  return s ? s->saturations : 0;
}

int addr_dedup_capacity(const addr_dedup_state_t *s) {
  return s ? s->capacity : 0;
}

void addr_dedup_destroy(addr_dedup_state_t *s) {
  if (!s)
    return;
  free(s->slots);
  free(s);
}

/* ---- slot map --------------------------------------------------------------- */
struct slot_map {
  uint64_t *keys; /* stored key+1 so 0 marks empty */
  int32_t *vals;
  size_t cap, n;
};

static inline uint64_t slot_hash(uint64_t k) {
  k ^= k >> 33; k *= 0xFF51AFD7ED558CCDULL; k ^= k >> 33;
  k *= 0xC4CEB9FE1A85EC53ULL; k ^= k >> 33; return k;
}

slot_map_t *slot_map_create(size_t hint) {
  slot_map_t *m = (slot_map_t *)calloc(1, sizeof(*m));
  if (!m) return NULL;
  size_t cap = 1024;
  while (cap < hint * 2) cap <<= 1;
  m->cap = cap;
  m->keys = (uint64_t *)calloc(cap, sizeof(uint64_t));
  m->vals = (int32_t *)calloc(cap, sizeof(int32_t));
  if (!m->keys || !m->vals) { free(m->keys); free(m->vals); free(m); return NULL; }
  return m;
}

void slot_map_clear(slot_map_t *m) {
  if (!m)
    return;
  memset(m->keys, 0, m->cap * sizeof(*m->keys));
  memset(m->vals, 0, m->cap * sizeof(*m->vals));
  m->n = 0;
}

void slot_map_destroy(slot_map_t *m) {
  if (!m) return;
  free(m->keys); free(m->vals); free(m);
}

static int slot_map_grow(slot_map_t *m) {
  size_t ncap = m->cap << 1;
  uint64_t *nk = (uint64_t *)calloc(ncap, sizeof(uint64_t));
  int32_t *nv = (int32_t *)calloc(ncap, sizeof(int32_t));
  if (!nk || !nv) { free(nk); free(nv); return 0; }
  for (size_t i = 0; i < m->cap; i++) {
    if (!m->keys[i]) continue;
    size_t j = slot_hash(m->keys[i]) & (ncap - 1);
    while (nk[j]) j = (j + 1) & (ncap - 1);
    nk[j] = m->keys[i]; nv[j] = m->vals[i];
  }
  free(m->keys); free(m->vals);
  m->keys = nk; m->vals = nv; m->cap = ncap;
  return 1;
}

int32_t slot_map_get_or_put(slot_map_t *m, int lane, int elem, int32_t fresh,
                            int *inserted) {
  *inserted = 0;
  if (m->n * 2 >= m->cap && !slot_map_grow(m)) return -1;
  uint64_t key = (((uint64_t)(uint32_t)lane << 32) | (uint32_t)elem) + 1;
  size_t i = slot_hash(key) & (m->cap - 1);
  for (;;) {
    if (m->keys[i] == key) return m->vals[i] & 0x00FFFFFF; /* below the flag bits */
    if (m->keys[i] == 0) { m->keys[i] = key; m->vals[i] = fresh; m->n++; *inserted = 1; return fresh; }
    i = (i + 1) & (m->cap - 1);
  }
}

int slot_map_test_and_set(slot_map_t *m, int lane, int elem, int bit) {
  uint64_t key = (((uint64_t)(uint32_t)lane << 32) | (uint32_t)elem) + 1;
  size_t i = slot_hash(key) & (m->cap - 1);
  for (;;) {
    if (m->keys[i] == key) {
      int32_t mask = (int32_t)1 << bit;
      int prev = (m->vals[i] & mask) ? 1 : 0;
      m->vals[i] |= mask;
      return prev;
    }
    if (m->keys[i] == 0) return -1;
    i = (i + 1) & (m->cap - 1);
  }
}

int32_t slot_map_peek(const slot_map_t *m, int lane, int elem) {
  uint64_t key = (((uint64_t)(uint32_t)lane << 32) | (uint32_t)elem) + 1;
  size_t i = slot_hash(key) & (m->cap - 1);
  for (;;) {
    if (m->keys[i] == key) return m->vals[i] & 0x00FFFFFF;
    if (m->keys[i] == 0) return -1;
    i = (i + 1) & (m->cap - 1);
  }
}
