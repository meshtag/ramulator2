#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

/* ================================================================
 *  Configuration (overridable via environment variables)
 * ================================================================
 *
 *  RAMULATOR_TRACE_OUT     Output trace file path       (default: ramulator_trace.txt)
 *  CACHE_ENABLE            "1" to enable cache filter   (default: "1")
 *  CACHE_SIZE_KB           Total cache size in KB       (default: 1024 = 1 MB LLC)
 *  CACHE_ASSOC             Set associativity            (default: 16)
 *  CACHE_LINE_BYTES        Cache line size in bytes     (default: 64)
 * ================================================================ */

/* ────────────────────────────────────────────────────────────────
 *  Trace file
 * ──────────────────────────────────────────────────────────────── */
static FILE *trace_fp = NULL;
static const char *default_trace_path = "ramulator_trace.txt";

/* ────────────────────────────────────────────────────────────────
 *  Cache model — set-associative LRU
 *
 *  Each way stores:
 *    tag    – upper address bits identifying the line
 *    dirty  – written-to since last fill
 *    lru    – monotonic counter; highest = most-recently-used
 *    valid  – whether this way holds data
 * ──────────────────────────────────────────────────────────────── */
typedef struct {
    uint64_t tag;
    uint64_t lru;
    int      dirty;
    int      valid;
} cache_way_t;

static cache_way_t *cache        = NULL;   /* flat array: sets × assoc */
static int          cache_enable = 1;
static int          num_sets     = 0;
static int          assoc        = 0;
static int          line_bytes   = 64;
static int          index_bits   = 0;
static int          offset_bits  = 0;
static uint64_t     index_mask   = 0;
static uint64_t     lru_clock    = 0;      /* global clock for LRU ordering */

/* Stats */
static uint64_t stat_accesses    = 0;
static uint64_t stat_hits        = 0;
static uint64_t stat_misses      = 0;
static uint64_t stat_writebacks  = 0;
static uint64_t stat_dirty_evict = 0;

/* ──── helpers ──────────────────────────────────────────────────── */

static int ilog2(uint64_t v) {
    int r = 0;
    while (v >>= 1) r++;
    return r;
}

static inline uint64_t addr_tag(uint64_t addr) {
    return addr >> (offset_bits + index_bits);
}

static inline int addr_set(uint64_t addr) {
    return (int)((addr >> offset_bits) & index_mask);
}

static inline uint64_t align_addr(uint64_t addr) {
    return addr & ~((uint64_t)line_bytes - 1);
}

static inline cache_way_t *set_base(int set_idx) {
    return cache + (uint64_t)set_idx * assoc;
}

/*
 * Look up a cache line.  Returns the way pointer on hit, NULL on miss.
 * On hit: updates LRU and dirty flag.
 */
static cache_way_t *cache_lookup(uint64_t addr, int is_write) {
    uint64_t tag     = addr_tag(addr);
    int      set_idx = addr_set(addr);
    cache_way_t *set = set_base(set_idx);

    for (int i = 0; i < assoc; i++) {
        if (set[i].valid && set[i].tag == tag) {
            set[i].lru = ++lru_clock;
            if (is_write) set[i].dirty = 1;
            return &set[i];
        }
    }
    return NULL;
}

/*
 * Install a new line, evicting the LRU victim if needed.
 * Returns 1 if a dirty line was evicted (caller should emit a writeback).
 * Fills *evicted_addr with the writeback address when applicable.
 */
static int cache_install(uint64_t addr, int is_write, uint64_t *evicted_addr) {
    uint64_t tag     = addr_tag(addr);
    int      set_idx = addr_set(addr);
    cache_way_t *set = set_base(set_idx);

    /* find an invalid (empty) way first */
    for (int i = 0; i < assoc; i++) {
        if (!set[i].valid) {
            set[i].valid = 1;
            set[i].tag   = tag;
            set[i].dirty = is_write;
            set[i].lru   = ++lru_clock;
            return 0;
        }
    }

    /* all ways valid — find LRU victim (lowest lru counter) */
    int lru_idx = 0;
    uint64_t min_lru = set[0].lru;
    for (int i = 1; i < assoc; i++) {
        if (set[i].lru < min_lru) {
            min_lru = set[i].lru;
            lru_idx = i;
        }
    }

    int dirty_eviction = set[lru_idx].dirty;
    if (dirty_eviction) {
        /* reconstruct the evicted address from tag + set index */
        *evicted_addr = (set[lru_idx].tag << (offset_bits + index_bits))
                      | ((uint64_t)set_idx << offset_bits);
    }

    set[lru_idx].tag   = tag;
    set[lru_idx].dirty = is_write;
    set[lru_idx].lru   = ++lru_clock;

    return dirty_eviction;
}

/* ────────────────────────────────────────────────────────────────
 *  Core access function
 * ──────────────────────────────────────────────────────────────── */
static void trace_access(uint64_t addr, int is_write) {
    stat_accesses++;

    if (!cache_enable) {
        /* No cache — every access goes to DRAM */
        fprintf(trace_fp, "%s 0x%" PRIx64 "\n", is_write ? "ST" : "LD", addr);
        return;
    }

    cache_way_t *hit = cache_lookup(addr, is_write);
    if (hit) {
        stat_hits++;
        return;   /* cache hit — no DRAM traffic */
    }

    /* cache miss */
    stat_misses++;

    uint64_t evicted_addr = 0;
    int dirty_eviction = cache_install(addr, is_write, &evicted_addr);

    /* miss triggers a DRAM read (fill) */
    fprintf(trace_fp, "LD 0x%" PRIx64 "\n", addr);

    /* dirty eviction triggers a DRAM writeback */
    if (dirty_eviction) {
        stat_dirty_evict++;
        stat_writebacks++;
        fprintf(trace_fp, "ST 0x%" PRIx64 "\n", evicted_addr);
    }
}

/* ────────────────────────────────────────────────────────────────
 *  Public API called from instrumented IR
 * ──────────────────────────────────────────────────────────────── */

static int read_env_int(const char *name, int default_val) {
    const char *v = getenv(name);
    if (v) return atoi(v);
    return default_val;
}

void __mem_trace_init(void) {
    const char *path = getenv("RAMULATOR_TRACE_OUT");
    if (!path) path = default_trace_path;

    trace_fp = fopen(path, "w");
    if (!trace_fp) {
        fprintf(stderr, "[mem-trace] ERROR: cannot open %s for writing\n", path);
        exit(1);
    }

    /* cache configuration */
    cache_enable = read_env_int("CACHE_ENABLE", 1);
    if (!cache_enable) {
        fprintf(stderr, "[mem-trace] Cache DISABLED — all accesses go to trace\n");
        return;
    }

    int size_kb    = read_env_int("CACHE_SIZE_KB",   1024);
    assoc          = read_env_int("CACHE_ASSOC",     16);
    line_bytes     = read_env_int("CACHE_LINE_BYTES", 64);

    int total_bytes = size_kb * 1024;
    num_sets    = total_bytes / (line_bytes * assoc);
    offset_bits = ilog2(line_bytes);
    index_bits  = ilog2(num_sets);
    index_mask  = (uint64_t)num_sets - 1;

    cache = (cache_way_t *)calloc((uint64_t)num_sets * assoc, sizeof(cache_way_t));
    if (!cache) {
        fprintf(stderr, "[mem-trace] ERROR: failed to allocate cache (%d sets × %d ways)\n",
                num_sets, assoc);
        exit(1);
    }

    fprintf(stderr, "[mem-trace] Cache ENABLED: %d KB, %d-way, %d B lines, %d sets\n",
            size_kb, assoc, line_bytes, num_sets);
}

void __mem_trace_fini(void) {
    if (trace_fp) {
        fclose(trace_fp);
        trace_fp = NULL;
    }

    if (cache_enable) {
        fprintf(stderr, "\n[mem-trace] === Cache Statistics ===\n");
        fprintf(stderr, "[mem-trace]   Total accesses : %" PRIu64 "\n", stat_accesses);
        fprintf(stderr, "[mem-trace]   Hits           : %" PRIu64 " (%.2f%%)\n",
                stat_hits, stat_accesses ? 100.0 * stat_hits / stat_accesses : 0.0);
        fprintf(stderr, "[mem-trace]   Misses         : %" PRIu64 " (%.2f%%)\n",
                stat_misses, stat_accesses ? 100.0 * stat_misses / stat_accesses : 0.0);
        fprintf(stderr, "[mem-trace]   Dirty evictions: %" PRIu64 "\n", stat_dirty_evict);
        fprintf(stderr, "[mem-trace]   Writebacks     : %" PRIu64 "\n", stat_writebacks);
        fprintf(stderr, "[mem-trace]   DRAM requests  : %" PRIu64 " (misses + writebacks)\n",
                stat_misses + stat_writebacks);
    }

    if (cache) {
        free(cache);
        cache = NULL;
    }
}

void __mem_trace_load(void *addr, uint64_t size) {
    if (!trace_fp) return;

    uint64_t aligned_start = align_addr((uint64_t)addr);
    uint64_t aligned_end   = align_addr((uint64_t)addr + size - 1);

    for (uint64_t cl = aligned_start; cl <= aligned_end; cl += line_bytes) {
        trace_access(cl, 0);
    }
}

void __mem_trace_store(void *addr, uint64_t size) {
    if (!trace_fp) return;

    uint64_t aligned_start = align_addr((uint64_t)addr);
    uint64_t aligned_end   = align_addr((uint64_t)addr + size - 1);

    for (uint64_t cl = aligned_start; cl <= aligned_end; cl += line_bytes) {
        trace_access(cl, 1);
    }
}
