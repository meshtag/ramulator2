#include "pim_runtime.h"
#include "im_addr_dedup.h"
#include "im_runtime.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Compiler-emitted descriptor globals. */
#define PIM_LAYOUT_TABLE_DEFINE
#include "pim_layout_table.h"

/*
 * PIM Runtime Library
 *
 * Intercepts LLVM-instrumented loads/stores and translates them into
 * PIM trace operations (BR, BW, R, W) consumable by Ramulator2's
 * PimTrace frontend with the HBM3_PIM DRAM model.
 *
 * Trace format:  <OP> <ch>,<pch>,<bg>,<bank>,<sa>,<row>,<col>
 *   OP: R=read, W=write, BR=bank-read, BW=bank-write
 */

/* ================================================================
 *  HBM3_PIM organization constants (must match config)
 * ================================================================ */
static const int cfg_num_channels = 1;
static const int cfg_num_pch = 2;
static const int cfg_num_bg = 4;
static const int cfg_num_banks = 4; /* per bank group */
/* Copy of the organization the simulator decodes with, the HBM3_8Gb preset in
 * src/dram/impl/HBM3_PIM.cpp. Nothing checks that the two agree. The subarray count only
 * sets the capacity ceiling and takes no part in the address slicing. */
static const int cfg_num_sa = 64;
static const int cfg_num_rows = 512;
/* One column command moves internal_prefetch * dq = 2 * 128 = 256 bits on this part,
 * which is the HBM-PIM datapath, so the column index counts 256-bit words. Columns per
 * row halve to match, keeping the row at 8192 bits and the part's density unchanged.
 * These two MUST move together, and together with the simulator org (dq/column in the
 * rendered yaml), or the tracer addresses a row the machine does not have. */
static const int cfg_num_cols = 32;
static const int cfg_dq_bits = 256;

/* Width of one modelled value in bits, the shared spec's data_width (16 on this part).
 * Column packing follows it, values_per_col = dq_bits / data_width, and never the dtype
 * the host stores its arrays in. */
static const int cfg_data_width_bits = 16;

/* The interleaved map shifts by log2 of every level and of the values one column holds,
 * so a count that is not a power of two would silently corrupt it. */
#define PIM_POW2(v) ((v) > 0 && ((v) & ((v) - 1)) == 0)
_Static_assert(PIM_POW2(cfg_num_channels) && PIM_POW2(cfg_num_pch) &&
                   PIM_POW2(cfg_num_bg) && PIM_POW2(cfg_num_banks) &&
                   PIM_POW2(cfg_num_cols) && PIM_POW2(cfg_dq_bits / cfg_data_width_bits),
               "the interleaved map needs every level to be a power of two");

/* An HBM-PIM PE serves an even/odd bank pair (Lee et al. ISCA'21), so lane l owns banks 2l
 * and 2l+1, and pim_init refuses a kernel compiled for any other lane count. */
static const int cfg_num_flat_banks =
    cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
static const int cfg_num_lanes = cfg_num_flat_banks / 2;
_Static_assert(cfg_num_banks % 2 == 0, "a bank pair must stay inside one bank group");

/* From im_runtime: the replay lane of the current kernel call. */
extern int32_t __pim_get_bank_id(void);

/* The next free linear-element position. Tensors are
 * appended consecutively in the global linear-element address space and
 * aligned only to a values_per_col boundary (so a new tensor doesn't
 * partially share a packed DQ slot with the previous one).
 *
 * Tensors DO share physical rows — that is desirable: distinct tensors
 * occupy disjoint linear ranges, so they land at different bank slots
 * within a shared row, and the simulator services them in parallel
 * across banks.
 *
 * Worked example (vpc=4, num_cols=64, 4 banks/BG, 4 BGs, 2 pchs):
 *   A (256 elem): linear_base=0   → bank 0 of BG 0 of pch 0,
 *                                    row 0, cols 0..63 (full bank-row).
 *   B ( 72 elem): linear_base=256 → bank 1 of BG 0 of pch 0,
 *                                    row 0, cols 0..17 (= ceil(72/4)).
 *   C (1568 elem): linear_base=328 → starts at bank 1 of BG 0 of pch 0,
 *                                     row 0, col 18 (continuing B's row);
 *                                     spreads across the rest of BG 0,
 *                                     all of BG 1, and the first three
 *                                     banks of BG 2, all in row 0.
 * All three tensors share physical row 0 but live on different banks,
 * so a host-loop iteration that touches all three drives several banks
 * concurrently rather than serializing on one. */
static uint64_t g_interleaved_next_linear = 0;
/* Lane slabs take rows from the TOP of the per-bank row space; interleaved placement
 * grows from the bottom, and the two must never meet. */
static int g_lane_row_top = 0;
/* Words one half of a pair takes before the other: Samsung's kernels run 8 columns on
 * the even bank, then the same 8 on the odd one, one per GRF_A entry (PIMKernel.cpp). */
#define PIM_PAIR_BLOCK_WORDS 8

/* Compute log2(n) where n is a power of two. Returns -1 for n<=0. */
static int ilog2_pow2(int n) {
  if (n <= 0)
    return -1;
  int k = 0;
  while ((1 << k) < n)
    k++;
  return k;
}

/* ================================================================
 *  Tensor registry
 * ================================================================ */
#define MAX_TENSORS 16
#define MAX_BANKS 512
_Static_assert(cfg_num_lanes <= MAX_BANKS, "lane_ord holds one counter per lane");

typedef struct {
  void *base_addr;
  size_t total_bytes;
  int elem_size;
  int num_elements;
  pim_role_t role;

  int values_per_col;
  int values_per_row;

  /* Offset of element 0 in the global bit-interleaved address space. */
  uint64_t layout_linear_base;
  /* Lanes that receive a copy, the record's bank_replicated word as recv_lanes_of reads
   * it. 0 when the tensor is partitioned and -1 when the compiler stated nothing. */
  int recv_lanes;
  int lane_row_base;        /* first row of this tensor's per-lane slab, or -1 */
  int lane_row_count;
  int lane_ord[MAX_BANKS];  /* next free value slot per lane */
  slot_map_t *lane_slot;    /* (lane, elem) -> slot, set exactly when the tensor has a slab */
} tensor_info_t;

static tensor_info_t tensors[MAX_TENSORS];
static int num_tensors = 0;

/* Namespace bits for the lockstep-collapse key: tid 4b, ch 5b, pch 2b, write 1b,
 * bank parity 1b. lockstep_key hashes all 64 bits.
 *
 * An all-bank PIM command reaches the banks of one pseudochannel, so ch and pch are in the
 * key, and bank and bank group are not, which is the all-bank SIMD model. The write bit
 * keeps an accumulator store from folding into the load that read the same partial sum.
 * A PE reads one bank of its pair per command and the pair's halves share (row, col), so
 * the parity keeps the halves apart. */
static inline uint64_t lockstep_ns(const tensor_info_t *t, int ch, int pch,
                                   int is_write, int parity) {
  return ((uint64_t)(t - tensors) & 0xFULL) | (((uint64_t)ch & 0x1FULL) << 4) |
         (((uint64_t)pch & 0x3ULL) << 9) |
         ((uint64_t)(is_write ? 1 : 0) << 11) | ((uint64_t)(parity & 1) << 12);
}

/* ================================================================
 *  Runtime state
 * ================================================================ */
static FILE *trace_fp = NULL;

static pim_phase_t cur_phase = PIM_PHASE_IDLE;
static int initialized = 0;
static int finalized = 0;

/* Per-run counts and warn-once flags. pim_init clears both, because the harness inits
 * one library more than once per process (the ladder verifies, then traces), and each
 * run must report its own counts and WARNs. */
static struct {
  uint64_t bank_reads, bank_writes, reads, writes, ignored;
  /* Operand deliveries NOT charged because the compiler said the value is replicated
   * over only some of the banks. 0 whenever every split is single-axis, which is every
   * mapping our own picker emits. */
  uint64_t operand_repl_skips;
  /* Loads folded by the within-call DQ-word coalescer: lanes of ONE vector load that
   * land on the same DRAM column. */
  uint64_t load_coalesced;
  /* Distinct keys past the 64-entry per-call buffer, i.e. coalescing this call could
   * not complete. Must stay 0. Nonzero means a kernel emits a vector wider than the
   * buffer and some duplicate lanes were charged twice. */
  uint64_t coalesce_overflow;
  uint64_t lockstep_skips;
  uint64_t lane_placed;
  int layout_from_compiler;
} stats;
/* One flag per WARN condition, so one condition cannot hide another. */
static struct {
  int dispatch_overflow, lane_outside, slot_alloc, slab_overflow, no_layout_table,
      lane_count, grf_a;
} warned;

/* Lockstep collapse. The host replays each kernel once per bank, so one SIMD
 * instruction reaches the trace once per bank, while HBM-PIM issues it once to every
 * bank of a pseudochannel (OptiPIM's single_bank_opt). Replicas with the same
 * lockstep_ns, dispatch, lane ordinal, row and col fold into one event. Every role
 * takes it, operands included, whose replicas are one all-bank GRF write (Lee et al.
 * ISCA'21). The table grows on demand. */
static addr_dedup_state_t *g_lockstep_collapse = NULL;

/* The dispatch this runtime is currently seeing, taken from __pim_program_epoch.
 *
 * It is part of the lockstep KEY rather than a signal to reset the table. Those are
 * equivalent while the host drives one tile per program instance, because the bank
 * loop sits inside the instance loop, so an instance's keys are all adjacent. They
 * stop being equivalent for a persistent kernel, where the bank loop is OUTSIDE and
 * one replay walks every tile: bank 0 finishes tile T before bank 1 starts tile 0, so
 * resetting on a tile boundary would put the lane replays of one tile in different
 * epochs and the collapse would never fire at all. Keying works for both. */
static uint64_t cur_dispatch = 0;

/* Bits of the dispatch id given to the tile index, with the program epoch above them.
 * refresh_dispatch_id WARNs when a tile index does not fit. */
#define PIM_TILE_BITS 10

/* Per-lane access ordinal within a dispatch. Lanes replay one instruction stream, so the
 * k-th access of every lane is the same instruction. Without it the key cannot tell one
 * instruction seen on 16 lanes from 16 separate reads of one column, and the second case
 * folds away: scalar loads then cost what a vector load costs and vectorization reads 1.00x.
 * Advanced once per IR-level access, never per expanded element, so a 16-wide load keeps
 * one ordinal while 16 scalar loads take 16. */
static int g_ord_lane = -1;
static uint64_t g_ord_disp = UINT64_MAX;
static uint64_t g_access_ord = 0;

/* GRF_A, the register file an operand word is written into, has eight 256-bit entries per
 * PE (Lee et al. ISCA'21). g_grf_a holds the executing lane's entries, most recently used
 * first, and empties when the lane or the dispatch changes. */
#define PIM_GRF_A_ENTRIES 8
static uint64_t g_grf_a[PIM_GRF_A_ENTRIES];
static int g_grf_a_used = 0;
/* Every (lane, operand word, dispatch) delivered so far, so that a re-read is known. */
static addr_dedup_state_t *g_operand_words = NULL;

static void access_ord_next(void) {
  int lane = (int)__pim_get_bank_id();
  if (lane != g_ord_lane || cur_dispatch != g_ord_disp) {
    g_ord_lane = lane; g_ord_disp = cur_dispatch; g_access_ord = 0;
    g_grf_a_used = 0;
  }
  g_access_ord++;
}

/* The ordinal does not fit pack_keys' 16/32/16 layout beside row and col, so the five
 * fields are hashed to 64 bits and split back across them; that split is bijective. */
static inline uint64_t lk_mix(uint64_t x) {
  x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
  x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ULL;
  return x ^ (x >> 33);
}

static uint64_t lockstep_key(uint64_t ns, uint64_t disp, uint64_t ord, uint64_t row,
                             uint64_t col) {
  uint64_t h = lk_mix(ns ^ 0x9E3779B97F4A7C15ULL);
  h = lk_mix(h ^ disp);
  h = lk_mix(h ^ ord);
  return lk_mix(h ^ ((row << 16) | col));
}

/* Record decoding, the drift checks, tensor lookup, trace output and the MemTracePass
 * hooks, shared with simdram_runtime.c. */
#include "im_trace_common.h"

/* ================================================================
 *  Helpers
 * ================================================================ */

static void destroy_dedup_state(void) {
  addr_dedup_destroy(g_lockstep_collapse);
  g_lockstep_collapse = NULL;
  addr_dedup_destroy(g_operand_words);
  g_operand_words = NULL;
}

/* Sets cur_dispatch from the epoch the host signals through the program-id setters, so
 * the cadence does not depend on the order the replay walks the grid. Only a persistent
 * kernel spends bits on the tile index, so a gridded kernel's epoch keeps the whole id. */
static void refresh_dispatch_id(void) {
  uint64_t tile = __pim_tile_index();
  uint64_t epoch = __pim_program_epoch;
  if (__pim_persistent && tile >= (1ULL << PIM_TILE_BITS) && !warned.dispatch_overflow++)
    fprintf(stderr,
            "[pim-runtime] WARN: dispatch id does not fit (%llu tiles, epoch "
            "%llu, %d tile bits). Ids alias and distinct dispatches collapse.\n",
            (unsigned long long)tile, (unsigned long long)epoch, PIM_TILE_BITS);
  cur_dispatch = __pim_persistent ? ((epoch << PIM_TILE_BITS) | tile) : epoch;
}

static int compute_global_bank(int ch, int pch, int bg, int bank) {
  int banks_per_pch = cfg_num_bg * cfg_num_banks;
  int banks_per_ch = cfg_num_pch * banks_per_pch;
  return ch * banks_per_ch + pch * banks_per_pch + bg * cfg_num_banks + bank;
}

/* Bank h of lane l's pair. A pair stays inside one bank group, since a group holds an
 * even number of banks. */
static inline int lane_bank(int lane, int h) {
  return 2 * lane + h;
}

/* Bit-interleaved (scheme8-like) element-to-physical mapping.
 *
 * Address bit layout (LSB → MSB), each field is sized to the runtime
 * configuration via log2 of cfg_*. There is NO hardcoded bank count —
 * the field widths shift to match cfg_num_banks, cfg_num_bg, cfg_num_pch,
 * cfg_num_channels, cfg_num_cols, and values_per_col at runtime. All of
 * those must be powers of two, which the _Static_assert after the constants enforces.
 *
 *   [ within-DQ | col | bank-within-BG | BG | pch | chan | linear_row ]
 *
 * Sequential element indices first fill the within-DQ slot (vpc values),
 * then advance through cols, then through banks within a BG, then BGs,
 * then pch, then channels, finally advancing linear_row (sa*num_rows + row).
 *
 * Worked example for the constants above (vpc=16, num_cols=32, 4 banks/BG, 4 BGs,
 * 2 pchs, 1 channel):
 *   bits 0..3   within-DQ (16 values per 256-bit col)
 *   bits 4..8   col (32 cols)
 *   bits 9..10  bank within BG (4)
 *   bits 11..12 BG (4)
 *   bit  13     pch (2)
 *   (no chan bits for cfg_num_channels=1)
 *   bits 14..   linear_row
 * Elements 0..511 lie within bank 0 of BG 0 of pch 0 row 0, 512..1023 advance to
 * bank 1 of BG 0, and after all 4 banks of BG 0 (2048 elements) BG 1 follows.
 *
 * The geometry changes only by editing the constants, which must stay powers of two. */
static void map_element_interleaved(const tensor_info_t *t, int elem_idx,
                                    int *ch, int *pch, int *bg, int *bank,
                                    int *sa, int *row, int *col) {
  uint64_t linear = t->layout_linear_base + (uint64_t)elem_idx;

  int log2_vpc      = ilog2_pow2(t->values_per_col);
  int log2_cols     = ilog2_pow2(cfg_num_cols);
  int log2_banks_bg = ilog2_pow2(cfg_num_banks);
  int log2_bg       = ilog2_pow2(cfg_num_bg);
  int log2_pch      = ilog2_pow2(cfg_num_pch);
  int log2_chan     = ilog2_pow2(cfg_num_channels);

  /* Strip the within-DQ bits (multiple values share one physical column). */
  linear >>= log2_vpc;

  *col = (int)(linear & ((1ULL << log2_cols) - 1));
  linear >>= log2_cols;

  *bank = (int)(linear & ((1ULL << log2_banks_bg) - 1));
  linear >>= log2_banks_bg;

  *bg = (int)(linear & ((1ULL << log2_bg) - 1));
  linear >>= log2_bg;

  *pch = (int)(linear & ((1ULL << log2_pch) - 1));
  linear >>= log2_pch;

  *ch = (int)(linear & ((1ULL << log2_chan) - 1));
  linear >>= log2_chan;

  /* Whatever remains is the linear row index across all subarrays. */
  int linear_row = (int)linear;
  *sa = linear_row / cfg_num_rows;
  *row = linear_row % cfg_num_rows;
}

typedef struct {
  int elem_idx;
  int ch;
  int pch;
  int bg;
  int bank;
  int sa;
  int row;
  int col;
} pim_phys_loc_t;

/* Lane l may touch only its own pair of banks. The element goes in the next slot of the
 * executing lane's slab (OptiPIM's alloc_col++ made explicit), and the slot is remembered
 * so a re-read lands on the same address. A replicated tensor is whole in every lane's
 * slab, so its copy in each bank has addresses of its own. */
static void place_in_lane_slab(tensor_info_t *t, pim_phys_loc_t *loc) {
  int lane = __pim_get_bank_id();
  if (lane < 0 || lane >= cfg_num_lanes) {
    if (!warned.lane_outside++)
      fprintf(stderr, "[pim-runtime] WARN replay lane %d outside the %d lanes; kept "
                      "the interleaved home.\n", lane, cfg_num_lanes);
    return;
  }
  /* Every lane loads a replicated tensor in the same order, so one numbering serves
   * all copies: key on lane 0 and the map stays the size of the tensor, not 16x. */
  int key_lane = (t->recv_lanes > 0) ? 0 : lane;
  int inserted = 0;
  int32_t slot = slot_map_get_or_put(t->lane_slot, key_lane, loc->elem_idx,
                                     t->lane_ord[key_lane], &inserted);
  if (slot < 0) {
    if (!warned.slot_alloc++)
      fprintf(stderr, "[pim-runtime] WARN slab slot map allocation failed; kept the "
                      "interleaved home.\n");
    return;
  }
  if (inserted)
    t->lane_ord[key_lane]++;
  int vpc = t->values_per_col;
  int vpr = t->values_per_row;
  /* The halves of a pair take turns in blocks of words at matching (row, col). This is
   * the runtime's convention, not a compiler statement, and the block size moves cycles. */
  int w = slot / vpc;
  int h = (w / PIM_PAIR_BLOCK_WORDS) % 2;
  int bw = (w / (2 * PIM_PAIR_BLOCK_WORDS)) * PIM_PAIR_BLOCK_WORDS + w % PIM_PAIR_BLOCK_WORDS;
  int bslot = bw * vpc + slot % vpc;
  int r = bslot / vpr;
  if (r >= t->lane_row_count) {
    if (!warned.slab_overflow++)
      fprintf(stderr, "[pim-runtime] WARN tensor lane %d overflowed its %d-row slab; "
                      "rows now alias.\n", lane, t->lane_row_count);
    r = t->lane_row_count - 1;
  }
  int linear_row = t->lane_row_base + r;
  decompose_global_bank(lane_bank(lane, h), &loc->ch, &loc->pch, &loc->bg, &loc->bank);
  loc->sa = linear_row / cfg_num_rows;
  loc->row = linear_row % cfg_num_rows;
  loc->col = (bslot % vpr) / vpc;
  stats.lane_placed++;
}

/* The element's physical tuple: its slot in the executing lane's slab when the tensor has
 * a slab, else its interleaved home. Returns 0, counted as ignored, when addr is outside
 * the tensor. */
static int locate(tensor_info_t *t, uint64_t addr, pim_phys_loc_t *loc) {
  int elem_idx = (int)((addr - (uint64_t)t->base_addr) / t->elem_size);
  if (elem_idx < 0 || elem_idx >= t->num_elements) {
    stats.ignored++;
    return 0;
  }
  loc->elem_idx = elem_idx;
  map_element_interleaved(t, elem_idx, &loc->ch, &loc->pch, &loc->bg, &loc->bank,
                          &loc->sa, &loc->row, &loc->col);
  if (t->lane_slot)
    place_in_lane_slab(t, loc);
  return 1;
}

/* The trace op for one access, by direction and role:
 *   STREAMED load     -> BR (bank-read: data streams through the PE)
 *   OPERAND load      -> W  (write: data to the PE's registers over the bus)
 *   ACCUMULATOR load  -> R  (read: partial sum from the PE)
 *   ACCUMULATOR store -> BW (bank-write: result to the bank)
 * A store to a streamed or operand tensor emits nothing. */
static void emit_access(const tensor_info_t *t, const pim_phys_loc_t *loc, int is_write) {
  if (is_write) {
    if (t->role == PIM_ROLE_ACCUMULATOR) {
      emit_trace("BW", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa, loc->row, loc->col);
      stats.bank_writes++;
    }
    return;
  }
  switch (t->role) {
  case PIM_ROLE_STREAMED:
    emit_trace("BR", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa, loc->row, loc->col);
    stats.bank_reads++;
    break;
  case PIM_ROLE_OPERAND: {
    /* A bus write into the receiving PE's register file, so it is addressed at the replay
     * lane's bank rather than the element's home, as OptiPIM's codegen writes to
     * global_bank_id + i (fimdram.cpp:63-76). A value of a multi-axis split is replicated
     * only along the axes it misses, so only every (lanes / recv_lanes)-th lane gets it. */
    int stride = t->recv_lanes > 0 ? cfg_num_lanes / t->recv_lanes : 1;
    if (stride > 1 && (__pim_get_bank_id() % stride) != 0) {
      stats.operand_repl_skips++;
      break;
    }
    int dch, dpch, dbg, dbank;
    /* The pair half that holds the word, so W lands on the parity its key carries. */
    decompose_global_bank(lane_bank(__pim_get_bank_id(), loc->bank & 1), &dch, &dpch,
                          &dbg, &dbank);
    emit_trace("W", dch, dpch, dbg, dbank, loc->sa, loc->row, loc->col);
    stats.writes++;
    break;
  }
  case PIM_ROLE_ACCUMULATOR:
    emit_trace("R", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa, loc->row, loc->col);
    stats.reads++;
    break;
  }
}

/* ================================================================
 *  Public API
 * ================================================================ */

void pim_init(const char *trace_file) {
  /* If already initialized, close the existing trace and reinitialize. */
  if (initialized && trace_fp) {
    fclose(trace_fp);
    trace_fp = NULL;
  }
  destroy_dedup_state();

  if (!trace_file)
    trace_file = "pim_trace.txt";

  trace_fp = fopen(trace_file, "w");
  if (!trace_fp) {
    fprintf(stderr, "[pim-runtime] ERROR: cannot open %s\n", trace_file);
    exit(1);
  }

  check_compiler_dq_bits("pim-runtime", cfg_dq_bits);
  check_layout_rec_words("pim-runtime");
  /* Row-pack is the dcc-parity runtime's placement, so a kernel stamped for it is refused
   * rather than priced on the interleaved map. */
  if ((__pim_layout_scheme != 0 && __pim_layout_scheme != 2) ||
      (__pim_placement_align != 0 && __pim_placement_align != 1)) {
    fprintf(stderr, "[pim-runtime] ERROR: the kernel states scheme=%d align=%d, and this "
                    "runtime places interleaved (2) at DQ alignment (1) only\n",
            (int)__pim_layout_scheme, (int)__pim_placement_align);
    exit(1);
  }
  if ((int)__pim_lanes != cfg_num_lanes) {
    fprintf(stderr, "[pim-runtime] ERROR: the kernel was compiled for %d lanes, and this "
                    "runtime models a PE per bank pair, %d lanes on %d banks\n",
            (int)__pim_lanes, cfg_num_lanes, cfg_num_flat_banks);
    exit(1);
  }
  memset(&stats, 0, sizeof stats);
  memset(&warned, 0, sizeof warned);
  g_ord_lane = -1;
  g_ord_disp = UINT64_MAX;
  g_access_ord = 0;
  g_grf_a_used = 0;
  g_interleaved_next_linear = 0;
  g_lane_row_top = cfg_num_sa * cfg_num_rows;

  num_tensors = 0;
  cur_phase = PIM_PHASE_IDLE;
  /* This line must not say WARN: the ladder records every WARN line in lever_ladder.json,
   * and it refuses an HBM run whose stderr lacks this line. */
  fprintf(stderr, "[pim-runtime] bank pairs: %d lanes on %d banks, lane l owns banks "
                  "2l and 2l+1\n", cfg_num_lanes, cfg_num_flat_banks);

  /* HBM-PIM issues one all-bank command, but pipeline replays the kernel once per
   * (pid, bank), so the collapse undoes the replay. 136 entries per bank is only the
   * table's starting size. It grows on demand. */
  g_lockstep_collapse = addr_dedup_create(cfg_num_flat_banks * 136);
  g_operand_words = addr_dedup_create(1024);

  initialized = 1;
  finalized = 0;

  fprintf(stderr, "[pim-runtime] Initialized. Trace: %s\n", trace_file);
  fprintf(stderr,
          "[pim-runtime] HBM config: %d ch, %d pch, %d bg, %d banks/bg, "
          "%d sa, %d rows, %d cols, %d-bit DQ\n",
          cfg_num_channels, cfg_num_pch, cfg_num_bg, cfg_num_banks, cfg_num_sa,
          cfg_num_rows, cfg_num_cols, cfg_dq_bits);
}

/* ================================================================
 *  Compiler-emitted residency descriptor
 * ================================================================
 * emitPimLayoutTable (IMArtifact.cpp) puts these globals in the KERNEL
 * object. Kernel and runtime link into one dylib, so we just read them.
 *
 * Record layout in pim_layout_table.h, which the SIMDRAM runtime shares. The runtime
 * reads operand_arg, bank_replicated (0 when the tensor is bank-partitioned, otherwise
 * how many banks receive a copy, 1 meaning every bank, and -1 when the pass said
 * nothing) and is_store. The footprint words are unread here but keep the record width
 * equal to kRecWords. */

/* The record's bank_replicated word as a lane count. 1 means every lane, as does a count
 * of cfg_num_lanes or more, and a count between is a multi-axis split's receivers. 0 is a
 * partitioned tensor and -1 means the compiler said nothing. */
static int recv_lanes_of(int32_t word) {
  if (word <= 0)
    return word < 0 ? -1 : 0;
  return (word > 1 && word < cfg_num_lanes) ? (int)word : cfg_num_lanes;
}

/* Values one lane's slab holds. A replicated tensor is whole in every lane. A partitioned
 * lane can touch far more than its even share when lanes overlap, as a conv halo does, so
 * its slab holds the whole tensor up to PIM_SLAB_CAP values, and twice the even share once
 * that passes the cap. */
#define PIM_SLAB_CAP 65536
static int slab_values_per_lane(const tensor_info_t *t) {
  if (t->recv_lanes > 0)
    return t->num_elements;
  int share2 = 2 * ((t->num_elements + cfg_num_lanes - 1) / cfg_num_lanes);
  if (share2 > PIM_SLAB_CAP)
    return share2;
  return t->num_elements < PIM_SLAB_CAP ? t->num_elements : PIM_SLAB_CAP;
}

/* Reserves a slab of `values` per lane from the top of the row space, the same rows in
 * every bank. Rows are per bank, and each bank of a pair holds every other block of
 * words. When the rows do not fit, the tensor keeps its interleaved home and says so. */
static void reserve_lane_slab(tensor_info_t *t, int tensor_id, int values) {
  int vpc = t->values_per_col;
  int vpr = t->values_per_row;
  int span = 2 * PIM_PAIR_BLOCK_WORDS;
  int bank_words = ((values + vpc - 1) / vpc + span - 1) / span * PIM_PAIR_BLOCK_WORDS;
  int rows = (bank_words * vpc + vpr - 1) / vpr;
  if (rows < 1)
    rows = 1;
  if (rows > g_lane_row_top) {
    fprintf(stderr, "[pim-runtime] WARN tensor %d: lane slab of %d rows does not fit "
                    "the %d rows left; placement stays on the interleaved map.\n",
            tensor_id, rows, g_lane_row_top);
    return;
  }
  g_lane_row_top -= rows;
  t->lane_row_base = g_lane_row_top;
  t->lane_row_count = rows;
  t->lane_slot = slot_map_create((size_t)values / 2 + 1);
  if (!t->lane_slot) {
    t->lane_row_base = -1;
    fprintf(stderr, "[pim-runtime] WARN tensor %d: slot map allocation failed; "
                    "placement stays on the interleaved map.\n", tensor_id);
  }
}

/* Places one tensor and takes its role as the compiler's record says. A tensor without a
 * record keeps its interleaved home and the host's role, and a WARN says so, because that
 * placement is the runtime's choice rather than the compiler's. */
static void apply_compiler_layout(int tensor_id) {
  tensor_info_t *t = &tensors[tensor_id];
  const int32_t *rec = layout_record(tensor_id);
  if (!rec) {
    if ((int)__pim_layout_count <= 0) {
      if (!warned.no_layout_table++)
        fprintf(stderr, "[pim-runtime] WARN kernel carries no layout table; every tensor "
                        "stays on the interleaved map and the host roles stand.\n");
    } else if (t->role != PIM_ROLE_OPERAND) {
      fprintf(stderr, "[pim-runtime] WARN tensor %d: no layout record; placement stays "
                      "on the interleaved map.\n", tensor_id);
    }
    return;
  }
  if (!stats.layout_from_compiler++)
    fprintf(stderr, "[pim-runtime] compiler-emitted layout table found in the kernel "
                    "artifact (%d entries)\n", (int)__pim_layout_count);
  t->recv_lanes = recv_lanes_of(rec[PIM_LW_BANK_REPLICATED]);
  if (t->recv_lanes >= 0)
    reserve_lane_slab(t, tensor_id, slab_values_per_lane(t));
  t->role = (pim_role_t)role_from_record("pim-runtime", tensor_id, rec, t->role);
  if (t->recv_lanes < 0 && t->role != PIM_ROLE_OPERAND)
    fprintf(stderr, "[pim-runtime] WARN tensor %d (%s): compiler stated no partition "
                    "bit; placement stays on the interleaved map.\n", tensor_id,
            t->role == PIM_ROLE_ACCUMULATOR ? "ACCUMULATOR" : "STREAMED");
}

int pim_register_tensor(void *ptr, const int *dims, int ndims, int elem_size,
                        pim_role_t role) {
  if (num_tensors >= MAX_TENSORS) {
    fprintf(stderr, "[pim-runtime] ERROR: too many tensors (max %d)\n",
            MAX_TENSORS);
    return -1;
  }

  tensor_info_t *t = &tensors[num_tensors];
  t->base_addr = ptr;
  t->elem_size = elem_size;
  t->role = role;
  t->recv_lanes = -1;
  t->lane_row_base = -1;
  t->lane_row_count = 0;
  memset(t->lane_ord, 0, sizeof(t->lane_ord));
  if (t->lane_slot) { slot_map_destroy(t->lane_slot); t->lane_slot = NULL; }

  int total = 1;
  for (int i = 0; i < ndims && i < 4; i++) {
    total *= dims[i];
  }
  t->num_elements = total;
  t->total_bytes = (size_t)total * elem_size;

  /* Column packing follows the MODELLED data width, not the host array's dtype.
   * elem_size stays in use for address decoding (addr -> element index), which is
   * genuinely a property of how the host stores the array. */
  t->values_per_col = cfg_dq_bits / cfg_data_width_bits;
  t->values_per_row = cfg_num_cols * t->values_per_col;

  int rows_capacity = cfg_num_sa * cfg_num_rows;

  /* Interleaved (bit-interleaved scheme8-like) placement.
   *
   * Each tensor occupies a contiguous span of the global linear-element
   * address space; the bit decomposition in map_element_interleaved
   * places sequential elements across (within-DQ → col → bank → BG →
   * pch → chan → row) automatically.
   *
   * Sequential placement (only values_per_col alignment): consecutive
   * tensors share physical rows but land on different banks/cols, which
   * is desirable — small tensors get distinct banks naturally so the
   * simulator can service them in parallel without bank contention.
   * Aligning to a global-row boundary instead would force every tensor
   * to start at bank 0, serializing all small-tensor accesses there. */
  uint64_t elems_per_global_row =
      (uint64_t)t->values_per_row * (uint64_t)cfg_num_flat_banks;

  uint64_t align = (uint64_t)t->values_per_col;
  uint64_t base = g_interleaved_next_linear;
  if (align > 1 && base % align != 0) {
    base += align - (base % align);
  }
  t->layout_linear_base = base;
  g_interleaved_next_linear = base + (uint64_t)total;

  /* Sanity-check capacity: the highest bit we'll feed into linear_row
   * must fit within rows_capacity. The MSB of (base + total) above the
   * within-element bits gives the linear row count actually used. */
  uint64_t end = g_interleaved_next_linear;
  int log2_below_row = ilog2_pow2(t->values_per_col)
                     + ilog2_pow2(cfg_num_cols)
                     + ilog2_pow2(cfg_num_banks)
                     + ilog2_pow2(cfg_num_bg)
                     + ilog2_pow2(cfg_num_pch)
                     + ilog2_pow2(cfg_num_channels);
  uint64_t max_linear_row = end >> log2_below_row;
  if ((int)max_linear_row > rows_capacity) {
    fprintf(stderr,
            "[pim-runtime] ERROR: interleaved tensor %d needs linear_row "
            "up to %llu but capacity is %d\n",
            num_tensors, (unsigned long long)max_linear_row, rows_capacity);
    return -1;
  }

  fprintf(stderr,
          "[pim-runtime] Tensor %d: %s, %d elems (%zu bytes), interleaved "
          "linear_base=%llu (covers %llu global-row units)\n",
          num_tensors, role_name(role), total, t->total_bytes,
          (unsigned long long)t->layout_linear_base,
          (unsigned long long)((total + elems_per_global_row - 1) /
                               elems_per_global_row));
  int tid = num_tensors++;
  apply_compiler_layout(tid);
  return tid;
}

void pim_set_phase(pim_phase_t phase) {
  const char *names[] = {"IDLE", "COMPUTE"};
  if (phase <= PIM_PHASE_COMPUTE) {
    fprintf(stderr, "[pim-runtime] Phase -> %s\n", names[phase]);
  }
  cur_phase = phase;
}

void pim_finalize(void) {
  if (finalized)
    return;

  finalized = 1;

  if (trace_fp) {
    fclose(trace_fp);
    trace_fp = NULL;
  }

  fprintf(stderr, "\n[pim-runtime] === PIM Trace Statistics ===\n");
  fprintf(stderr, "[pim-runtime]   Bank-reads  (BR): %" PRIu64 "\n",
          stats.bank_reads);
  fprintf(stderr, "[pim-runtime]   Bank-writes (BW): %" PRIu64 "\n",
          stats.bank_writes);
  fprintf(stderr, "[pim-runtime]   Reads       (R) : %" PRIu64 "\n",
          stats.reads);
  fprintf(stderr, "[pim-runtime]   Writes      (W) : %" PRIu64 "\n",
          stats.writes);
  fprintf(stderr, "[pim-runtime]   Ignored        : %" PRIu64 "\n",
          stats.ignored);
  fprintf(stderr, "[pim-runtime]   Total PIM ops  : %" PRIu64 "\n",
          stats.bank_reads + stats.bank_writes + stats.reads + stats.writes);
  fprintf(stderr,
          "[pim-runtime]   Lockstep collapse skips         : %" PRIu64 "\n",
          stats.lockstep_skips);
  if (stats.operand_repl_skips)
    fprintf(stderr,
            "[pim-runtime]   Operand replication skips       : %" PRIu64
            " (compiler said fewer banks receive it)\n",
            stats.operand_repl_skips);
  fprintf(stderr, "[pim-runtime]   Lane-placed accesses            : %" PRIu64 "\n",
          stats.lane_placed);
  /* Slab occupancy per lane: how many values each lane placed in its own bank. Uneven
   * rows are masked or straddling lanes, not a bug; a zero row for an active lane is. */
  for (int i = 0; i < num_tensors; i++) {
    tensor_info_t *t = &tensors[i];
    if (!t->lane_slot)
      continue;
    fprintf(stderr, "[pim-runtime]   tensor %d slab slots per lane:", i);
    for (int l = 0; l < cfg_num_lanes; l++)
      fprintf(stderr, " %d", t->lane_ord[l]);
    fprintf(stderr, "  (rows %d..%d)\n", t->lane_row_base,
            t->lane_row_base + t->lane_row_count - 1);
  }
  {
    uint64_t sat = addr_dedup_saturations(g_lockstep_collapse);
    if (sat)
      fprintf(stderr,
              "[pim-runtime]   WARN dedup table saturated       : %" PRIu64
              " (collapse stopped; trace over-emits)\n", sat);
  }
  fprintf(stderr,
          "[pim-runtime]   Load coalesce (intra-vector)    : %" PRIu64 "\n",
          stats.load_coalesced);
  if (stats.coalesce_overflow)
    fprintf(stderr,
            "[pim-runtime]   WARN coalesce buffer overflow    : %" PRIu64
            " (an access touched more than PIM_COALESCE_MAX columns, so some of its "
            "word-mates were charged twice)\n",
            stats.coalesce_overflow);

  destroy_dedup_state();
  initialized = 0;
}

/* ================================================================
 *  LLVM pass hooks
 * ================================================================ */

/* The access's lockstep key. Dispatch and the lane's access ordinal ride in it, so the
 * bank replays of one access fold while two dispatches, or two accesses within a lane,
 * stay distinct. An operand stays in GRF for the whole dispatch, so its key drops the
 * access ordinal and cur_dispatch alone separates real re-deliveries. */
static uint64_t access_key(const tensor_info_t *t, const pim_phys_loc_t *loc, int is_write) {
  uint64_t ns = lockstep_ns(t, loc->ch, loc->pch, is_write, loc->bank & 1);
  uint64_t row = (uint64_t)loc->sa * (uint64_t)cfg_num_rows + (uint64_t)loc->row;
  uint64_t ord = (t->role == PIM_ROLE_OPERAND) ? 0 : g_access_ord;
  return lockstep_key(ns, cur_dispatch, ord, row, (uint64_t)loc->col);
}

/* The collapse prices a re-read of an operand word within a dispatch as a register hit.
 * That holds while fewer than PIM_GRF_A_ENTRIES other words came in since its last read,
 * so a re-read of a word that has left the entries is reported once. */
static void check_grf_a(uint64_t word) {
  int at = 0;
  while (at < g_grf_a_used && g_grf_a[at] != word)
    at++;
  if (at == g_grf_a_used) {
    uint64_t k = lk_mix(word ^ (uint64_t)g_ord_lane);
    if (!addr_dedup_check_and_mark(g_operand_words, k, k >> 16, k >> 48) &&
        !warned.grf_a++)
      fprintf(stderr, "[pim-runtime] WARN lane %d re-read an operand word that had left "
                      "its %d GRF_A entries, and the collapse prices the re-read as a "
                      "register hit.\n", g_ord_lane, PIM_GRF_A_ENTRIES);
    if (g_grf_a_used < PIM_GRF_A_ENTRIES)
      g_grf_a_used++;
    at = g_grf_a_used - 1;
  }
  for (; at > 0; at--)
    g_grf_a[at] = g_grf_a[at - 1];
  g_grf_a[0] = word;
}

/* Cap on distinct DRAM columns one instrumented access can coalesce over. */
#define PIM_COALESCE_MAX 64

/* Whether this call already sent the load word at loc, remembering it if not. A call is
 * one instruction, so a vector load costs one command per distinct column it touches,
 * while two separate loads of a word still pay twice. Every element took its slab slot
 * before this test, so a partitioned slab holds one slot per element and the key is the
 * placed address. */
static int coalesced(const pim_phys_loc_t *loc, uint64_t *seen, int *n_seen) {
  int gb = compute_global_bank(loc->ch, loc->pch, loc->bg, loc->bank);
  uint64_t key = ((uint64_t)gb & 0xFFFFULL) |
                 (((uint64_t)loc->sa & 0xFFULL) << 16) |
                 (((uint64_t)loc->row & 0xFFFFFFULL) << 24) |
                 (((uint64_t)loc->col & 0xFFFFULL) << 48);
  for (int j = 0; j < *n_seen; j++)
    if (seen[j] == key)
      return 1;
  if (*n_seen < PIM_COALESCE_MAX)
    seen[(*n_seen)++] = key;
  else
    stats.coalesce_overflow++;
  return 0;
}

/* One IR-level load or store over [base_addr, base_addr + size). Each element is located
 * once, a load word this call already sent is dropped, and of the lockstep replicas only
 * the first is emitted. Stores need no coalescing, since two lanes of one store at the
 * same address share the lockstep key. */
static void trace_access_range(uint64_t base_addr, uint64_t size, int is_write,
                               uint64_t compiler_lanes) {
  if (cur_phase != PIM_PHASE_COMPUTE) {
    stats.ignored++;
    return;
  }

  int tidx = find_tensor(base_addr);
  if (tidx < 0) {
    stats.ignored++;
    return;
  }
  tensor_info_t *t = &tensors[tidx];

  /* The ordinal keys on the dispatch, so the dispatch id is refreshed first. */
  refresh_dispatch_id();
  access_ord_next();

  int elem_size = t->elem_size > 0 ? t->elem_size : 1;
  uint64_t n_elements = size / (uint64_t)elem_size;
  if (n_elements == 0)
    n_elements = 1;
  check_access_lanes("pim-runtime", &warned.lane_count, compiler_lanes, size, elem_size,
                     n_elements);

  int coalesce = !is_write && n_elements > 1;
  uint64_t seen[PIM_COALESCE_MAX];
  int n_seen = 0;
  for (uint64_t i = 0; i < n_elements; i++) {
    uint64_t elem_addr = base_addr + i * (uint64_t)elem_size;
    /* A vector may run past its tensor into the next one. */
    if (elem_addr >= (uint64_t)t->base_addr + t->total_bytes) {
      int ntidx = find_tensor(elem_addr);
      if (ntidx < 0) {
        stats.ignored++;
        continue;
      }
      t = &tensors[ntidx];
    }
    pim_phys_loc_t loc;
    if (!locate(t, elem_addr, &loc))
      continue;
    if (coalesce && coalesced(&loc, seen, &n_seen)) {
      stats.load_coalesced++;
      continue;
    }
    uint64_t key = access_key(t, &loc, is_write);
    if (t->role == PIM_ROLE_OPERAND)
      check_grf_a(key);
    if (!addr_dedup_check_and_mark(g_lockstep_collapse, key, key >> 16, key >> 48)) {
      stats.lockstep_skips++;
      continue;
    }
    emit_access(t, &loc, is_write);
  }
}
