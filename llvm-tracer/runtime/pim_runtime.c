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
/* HAND-COPY of the DRAM organization the SIMULATOR decodes with. The tracer builds
 * addresses from these; ramulator reads them from the org preset named in the shared
 * spec (HBM3_8Gb -> {1, 2, 4, 4, 64 subarrays, 512 rows, 64 cols} in
 * third_party/ramulator2/src/dram/impl/HBM3_PIM.cpp). Nothing checks the two agree.
 * They did not until 2026-09-10: cfg_num_sa was 16, which is the HBM3_2Gb preset, so
 * the tracer modelled a quarter of the configured part's per-bank rows. Harmless as
 * it stood, because the subarray count only sets the capacity ceiling below and takes
 * no part in the address slicing, but a tensor needing more than 16*512 rows per bank
 * was refused on a limit the machine does not have. */
static const int cfg_num_sa = 64;
static const int cfg_num_rows = 512;
/* One column command moves internal_prefetch * dq = 2 * 128 = 256 bits on this part,
 * which is the HBM-PIM datapath, so the column index counts 256-bit words. Columns per
 * row halve to match, keeping the row at 8192 bits and the part's density unchanged.
 * These two MUST move together, and together with the simulator org (dq/column in the
 * rendered yaml), or the tracer addresses a row the machine does not have. */
static const int cfg_num_cols = 32;
static const int cfg_dq_bits = 256;
static void check_compiler_dq_bits(int cfg_bits);

/* Width of one modelled DATA VALUE in bits. This is a HARDWARE property (the shared
 * spec's data_width / pe_bits, 16 on this HBM3-PIM part), NOT the width of whatever
 * the host happens to store the test arrays in. Column packing must follow the
 * hardware: values_per_col = dq_bits / data_width. Deriving it from the numpy dtype
 * instead made us pack 4 int32 per column where the spec, and OptiPIM, pack 8, so we
 * paid double the bank-reads for the same tensor (2026-09-06). */
static const int cfg_data_width_bits = 16;

/* The interleaved map shifts by log2 of every level and of the values one column holds,
 * so a count that is not a power of two would silently corrupt it. */
#define PIM_POW2(v) ((v) > 0 && ((v) & ((v) - 1)) == 0)
_Static_assert(PIM_POW2(cfg_num_channels) && PIM_POW2(cfg_num_pch) &&
                   PIM_POW2(cfg_num_bg) && PIM_POW2(cfg_num_banks) &&
                   PIM_POW2(cfg_num_cols) && PIM_POW2(cfg_dq_bits / cfg_data_width_bits),
               "the interleaved map needs every level to be a power of two");

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
 * concurrently rather than serializing on one.
 *
 * An earlier draft aligned to a "global-row" boundary instead, which
 * forced every tensor's first element to bank 0 — that hot-spotted the
 * accumulator/operand/streamed tensors all on bank 0 and regressed
 * conv2d_1x8x16x16x3x3 by 3.2x. The current sequential-with-vpc-only
 * alignment is the version that produced the measured wins. */
static uint64_t g_interleaved_next_linear = 0;
/* Lane slabs take rows from the TOP of the per-bank row space; interleaved placement
 * grows from the bottom, and the two must never meet. */
static int g_lane_row_top = -1;
/* Banks per lane: 2 when the kernel has half as many lanes as the machine has banks,
 * because an HBM-PIM PE serves an even/odd bank pair. Set in pim_init. */
static int g_banks_per_lane = 1;
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
  /* The record's receive count (pim_layout_table.h), -1 when the compiler stated nothing.
   * Any stated value puts the tensor's accesses in the executing lane's slab, see
   * place_in_lane_slab, instead of the element's interleaved home. */
  int bank_replicated;
  int lane_row_base;        /* first row of this tensor's per-lane slab, or -1 */
  int lane_row_count;
  int lane_ord[MAX_BANKS];  /* next free value slot per lane */
  slot_map_t *lane_slot;    /* (lane, elem) -> slot, every tensor with a slab */
} tensor_info_t;

static tensor_info_t tensors[MAX_TENSORS];
static int num_tensors = 0;

/* Namespace bits for the lockstep-collapse key: tid 4b, ch 5b, pch 2b, write 1b,
 * bank parity 1b. lockstep_key hashes all 64 bits.
 *
 * ch/pch are in the key because an all-bank PIM command reaches the banks of ONE
 * pseudochannel; replicas in another pch or channel ride a separate command bus.
 * Bank/bankgroup stay OUT: collapsing those IS the all-bank SIMD model.
 * The write bit separates stores from loads, without which an ACCUMULATOR load
 * (the read half of a read-modify-write drain) marks the key the store then
 * consults, making partial-sum write-back free. Both found 2026-09-05.
 * Parity is 0 unless lanes own bank pairs: a PE reads one bank of its pair per
 * command, and the pair's two halves share (row, col), so without it they fold. */
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
      lane_count;
} warned;

/* Lockstep collapse. The host replays each kernel once per bank, so one SIMD
 * instruction reaches the trace once per bank, while HBM-PIM issues it once to every
 * bank of a pseudochannel (OptiPIM's single_bank_opt). Replicas with the same
 * lockstep_ns, dispatch, lane ordinal, row and col fold into one event. Every role
 * takes it, and the table grows on demand. */
static addr_dedup_state_t *g_lockstep_collapse = NULL;

/* The dispatch this runtime is currently seeing, taken from __pim_program_epoch.
 *
 * It is part of the lockstep KEY rather than a signal to reset the table. Those are
 * equivalent while the host drives one tile per program instance, because the bank
 * loop sits inside the instance loop, so an instance's keys are all adjacent. They
 * stop being equivalent for a persistent kernel, where the bank loop is OUTSIDE and
 * one replay walks every tile: bank 0 finishes tile T before bank 1 starts tile 0, so
 * resetting on a tile boundary would put the 32 replays of one tile in different
 * epochs and the collapse would never fire at all. Keying works for both. */
static uint64_t cur_dispatch = 0;

/* Bits of the dispatch field given to the tile index, with the program epoch
 * above it. Packed rather than summed: summing let instance p's tile 1 collide
 * with instance p+1's tile 0, so two distinct dispatches folded into one and the
 * trace under-emitted. The tile index is policed at pack time. */
#define PIM_TILE_BITS 10

/* Per-lane access ordinal within a dispatch. Lanes replay one instruction stream, so the
 * k-th access of every lane is the same instruction. Without it the key cannot tell one
 * instruction seen on 32 banks from 32 separate reads of one column, and the second case
 * folds away: scalar loads then cost what a vector load costs and vectorization reads 1.00x.
 * Advanced once per IR-level access, never per expanded element, so a 16-wide load keeps
 * one ordinal while 16 scalar loads take 16. */
static int g_ord_lane = -1;
static uint64_t g_ord_disp = UINT64_MAX;
static uint64_t g_access_ord = 0;

static void access_ord_next(void) {
  int lane = (int)__pim_get_bank_id();
  if (lane != g_ord_lane || cur_dispatch != g_ord_disp) {
    g_ord_lane = lane; g_ord_disp = cur_dispatch; g_access_ord = 0;
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

/* ================================================================
 *  Helpers
 * ================================================================ */

static void emit_trace(const char *op, int ch, int pch, int bg, int bank,
                       int sa, int row, int col) {
  fprintf(trace_fp, "%s %d,%d,%d,%d,%d,%d,%d\n", op, ch, pch, bg, bank, sa, row,
          col);
}

/*
 * Find which registered tensor contains 'addr'.
 * Returns tensor index or -1 if not found.
 */
// The function assumes that the tensor elements are placed contiguously in
// memory.
// TODO: Potential optimization: Use binary search in the address comparison
// calculation.
static int find_tensor(uint64_t addr) {
  for (int i = 0; i < num_tensors; i++) {
    uint64_t base = (uint64_t)tensors[i].base_addr;
    if (addr >= base && addr < base + tensors[i].total_bytes) {
      return i;
    }
  }
  return -1;
}

static void destroy_dedup_state() {
  if (g_lockstep_collapse) {
    addr_dedup_destroy(g_lockstep_collapse);
    g_lockstep_collapse = NULL;
  }
}

static void advance_program_epoch_if_needed(void) {
  /* Just record which dispatch we are in; the id goes into the lockstep key. See
   * cur_dispatch for why keying beats resetting. The epoch itself is signalled by the
   * host through the program-id setters rather than inferred from the ids, so the
   * cadence does not depend on the order the replay walks the grid. */
  {
    uint64_t tile = __pim_tile_index();
    uint64_t epoch = __pim_program_epoch;
    uint64_t packed = __pim_persistent ? ((epoch << PIM_TILE_BITS) | tile) : epoch;
    if (!warned.dispatch_overflow && __pim_persistent &&
        tile >= (1ULL << PIM_TILE_BITS)) {
      warned.dispatch_overflow = 1;
      fprintf(stderr,
              "[pim-runtime] WARN: dispatch id does not fit (%llu tiles, epoch "
              "%llu, %d tile bits). Ids alias and distinct dispatches collapse.\n",
              (unsigned long long)tile, (unsigned long long)epoch, PIM_TILE_BITS);
    }
    /* Only a persistent kernel spends bits on the tile index. A gridded one has
     * none, and shifting its epoch left by 10 overflowed the field at 128
     * instances. The artifact says which shape this is. */
    cur_dispatch = packed;
  }
}

static int compute_global_bank(int ch, int pch, int bg, int bank) {
  int banks_per_pch = cfg_num_bg * cfg_num_banks;
  int banks_per_ch = cfg_num_pch * banks_per_pch;
  return ch * banks_per_ch + pch * banks_per_pch + bg * cfg_num_banks + bank;
}

/* Inverse of compute_global_bank: flat replay bank id -> (ch,pch,bg,bank). */
static void decompose_global_bank(int g, int *ch, int *pch, int *bg, int *bank) {
  if (g < 0) g = 0;
  *bank = g % cfg_num_banks; g /= cfg_num_banks;
  *bg   = g % cfg_num_bg;    g /= cfg_num_bg;
  *pch  = g % cfg_num_pch;   g /= cfg_num_pch;
  *ch   = g;
}

/* Bank h of lane l's pair, or bank l itself when every bank has a PE. A pair stays
 * inside one bank group because pim_init requires an even bank count per group. */
static inline int lane_bank(int lane, int h) {
  return lane * g_banks_per_lane + (g_banks_per_lane > 1 ? h : 0);
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

/* Lanes are threads here: lane l is the PE of bank l, or of banks 2l and 2l+1 when
 * lanes own pairs, and may only touch its own banks. The
 * interleaved map changes bank every 512 elements while a lane's slab is 1024, so
 * lane 0's data sat in banks 0 and 1. Put the element in the executing lane's bank at
 * the next slot of that lane's slab (OptiPIM's alloc_col++ made explicit); the slot is
 * remembered so a re-read lands on the same address. A replicated tensor gets the same
 * treatment: its copy in bank b is a full slab, not the source word's (sa,row,col)
 * with the bank bits dropped, which aliased 4-8 words onto one address. */
static void place_in_lane_slab(tensor_info_t *t, pim_phys_loc_t *loc) {
  int lanes = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks /
              g_banks_per_lane;
  int lane = __pim_get_bank_id();
  if (lane < 0 || lane >= lanes || lane >= MAX_BANKS) {
    if (!warned.lane_outside++)
      fprintf(stderr, "[pim-runtime] WARN replay lane %d outside the %d lanes; kept "
                      "the interleaved home.\n", lane, lanes);
    return;
  }
  /* Every lane loads a replicated tensor in the same order, so one numbering serves
   * all 32 copies: key on lane 0 and the map stays the size of the tensor, not 32x. */
  int key_lane = (t->bank_replicated >= 1) ? 0 : lane;
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
  int blk = g_banks_per_lane > 1 ? PIM_PAIR_BLOCK_WORDS : 1;
  int h = (w / blk) % g_banks_per_lane;
  int bw = (w / (blk * g_banks_per_lane)) * blk + w % blk;
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

/* Resolve one logical element access to its physical HBM tuple.
 *
 * Returns 1 on success and fills `loc`. Returns 0 when the current phase
 * should not emit anything or the address falls outside the tensor. In both
 * failure cases, stats.ignored is updated here so callers can simply return. */
static int resolve_access_location(tensor_info_t *t, uint64_t addr,
                                   pim_phys_loc_t *loc) {
  if (cur_phase == PIM_PHASE_IDLE) {
    stats.ignored++;
    return 0;
  }

  int elem_idx = (int)((addr - (uint64_t)t->base_addr) / t->elem_size);
  if (elem_idx < 0 || elem_idx >= t->num_elements) {
    stats.ignored++;
    return 0;
  }

  loc->elem_idx = elem_idx;
  map_element_interleaved(t, elem_idx, &loc->ch, &loc->pch, &loc->bg, &loc->bank,
                          &loc->sa, &loc->row, &loc->col);
  if (cur_phase == PIM_PHASE_COMPUTE && t->bank_replicated >= 0 && t->lane_slot &&
      t->lane_row_base >= 0)
    place_in_lane_slab(t, loc);
  return 1;
}

/* Emit the trace op implied by the current phase, access direction, and tensor
 * role for an already-resolved physical tuple. */
static void emit_access_by_role_phase(tensor_info_t *t,
                                      const pim_phys_loc_t *loc, int is_write) {
  if (cur_phase == PIM_PHASE_COMPUTE) {
    if (!is_write) {
      switch (t->role) {
      case PIM_ROLE_STREAMED:
        emit_trace("BR", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa,
                   loc->row, loc->col);
        stats.bank_reads++;
        break;
      case PIM_ROLE_OPERAND: {
        /* An operand load is a bus-mediated write into the RECEIVING PE's register
         * file, so address it at the replay bank rather than at the source
         * element's home bank. Otherwise all N fanout writes pile onto one bank and
         * serialize, which over-charges us relative to OptiPIM, whose codegen writes
         * to global_bank_id + i (fimdram.cpp:63-76) and parallelizes. */
        /* HOW MANY BANKS ACTUALLY RECEIVE IT. bank_replicated is the compiler's count:
         * 1 means every bank (a scalar splat, or an artifact built before the count
         * existed), >1 means exactly that many. A value occupying one lane axis of a
         * multi-axis split is replicated only along the axes it misses, so charging one
         * write per replay lane billed OptiPIM's own 2-on-M-by-16-on-N mapping 16x over.
         * Identical to the old behaviour whenever the split has a single axis.
         * The count is in lanes, which is half the banks when lanes own pairs. */
        int nl = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks /
                 g_banks_per_lane;
        int recv = (t->bank_replicated > 1 && t->bank_replicated < nl)
                       ? t->bank_replicated : nl;
        int stride = (recv > 0) ? (nl / recv) : 1;
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
        emit_trace("R", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa,
                   loc->row, loc->col);
        stats.reads++;
        break;
      }
    } else {
      if (t->role == PIM_ROLE_ACCUMULATOR) {
        emit_trace("BW", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa,
                   loc->row, loc->col);
        stats.bank_writes++;
      }
      /* Stores to STREAMED/OPERAND in COMPUTE phase are ignored (read-only) */
    }
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

  check_compiler_dq_bits(cfg_dq_bits);
  /* Row-pack is the dcc-parity runtime's placement, so a kernel stamped for it is refused
   * rather than priced on the interleaved map. */
  if ((__pim_layout_scheme != 0 && __pim_layout_scheme != 2) ||
      (__pim_placement_align != 0 && __pim_placement_align != 1)) {
    fprintf(stderr, "[pim-runtime] ERROR: the kernel states scheme=%d align=%d, and this "
                    "runtime places interleaved (2) at DQ alignment (1) only\n",
            (int)__pim_layout_scheme, (int)__pim_placement_align);
    exit(1);
  }
  memset(&stats, 0, sizeof stats);
  memset(&warned, 0, sizeof warned);
  g_ord_lane = -1;
  g_ord_disp = UINT64_MAX;
  g_access_ord = 0;
  g_interleaved_next_linear = 0;
  g_lane_row_top = -1;

  num_tensors = 0;
  cur_phase = PIM_PHASE_IDLE;
  advance_program_epoch_if_needed();
  {
    /* Half as many lanes as banks is HBM-PIM's PE per bank pair (Lee et al. ISCA'21).
     * This line must not say WARN: the ladder records every WARN line in
     * lever_ladder.json. */
    int banks = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
    g_banks_per_lane = 1;
    if (__pim_lanes > 0 && 2 * (int)__pim_lanes == banks && cfg_num_banks % 2 == 0) {
      g_banks_per_lane = 2;
      fprintf(stderr, "[pim-runtime] bank pairs: %d lanes on %d banks, lane l owns banks "
                      "2l and 2l+1\n", (int)__pim_lanes, banks);
    } else if (__pim_lanes > 0 && (int)__pim_lanes != banks)
      fprintf(stderr, "[pim-runtime] WARN kernel compiled for %d lanes, machine has %d "
                      "banks; lane placement puts lane b in bank b and assumes they agree.\n",
              (int)__pim_lanes, banks);
  }

  /* HBM-PIM issues one all-bank command, but pass_ablation replays the kernel once per
   * (pid, bank), so the collapse undoes the replay. 136 entries per bank is only the
   * table's starting size. It grows on demand. */
  {
    int active_banks =
        cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
    g_lockstep_collapse = addr_dedup_create(active_banks * 136);
  }

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
 * emitPimLayoutTable (TritonIMToLLVM.cpp) puts these globals in the KERNEL
 * object. Kernel and runtime link into one dylib, so we just read them.
 *
 * Record layout in pim_layout_table.h, which the SIMDRAM runtime shares. The runtime
 * reads operand_arg, bank_replicated (0 when the tensor is bank-partitioned, otherwise
 * how many banks receive a copy, 1 meaning every bank, and -1 when the pass said
 * nothing) and is_store. operand_arg == tensor id, because pointer args are registered
 * first and in signature order. The footprint words are unread here but keep the
 * record width equal to kRecWords. */

/* The compiler bakes a bus width into every vector width it picks. Ours must match,
 * or the artifact was tuned for a machine this run is not modelling. */
static void check_compiler_dq_bits(int cfg_bits) {
  if (__pim_dq_bits && __pim_dq_bits != cfg_bits)
    fprintf(stderr,
            "[%s] WARN: bus-width disagreement. The compiler chose vector widths for "
            "a %d-bit bus; this run models %d-bit. Vector accesses are priced against "
            "a width the artifact never assumed.\n",
            "pim-runtime", (int)__pim_dq_bits, cfg_bits);
}

/* Refuse a descriptor whose record width is not the one we stride by. The compiler's
 * kRecWords and this runtime's PIM_LAYOUT_REC_WORDS are in separate submodules, so a
 * change to one alone misreads every record past the first, silently and plausibly. */
static void check_layout_rec_words(void) {
  static int checked = 0;
  if (checked)
    return;
  checked = 1;
  int emitted = (int)__pim_layout_rec_words;
  if (emitted > 0 && emitted != PIM_LAYOUT_REC_WORDS) {
    fprintf(stderr,
            "[pim-runtime] FATAL: layout record width mismatch. The compiler emitted "
            "%d words per record, this runtime strides by %d. The Triton and "
            "ramulator2 submodules have drifted; every record past the first would be "
            "misread. Rebuild both.\n",
            emitted, PIM_LAYOUT_REC_WORDS);
    abort();
  }
}

/* Honor the compiler's descriptor for one tensor. No-op if the kernel has none. */
static void apply_compiler_layout(int tensor_id) {
  check_layout_rec_words();
  int n = (int)__pim_layout_count;
  if (n <= 0) {
    /* The SIMDRAM twin of this silent return was a 64x under-charge (review 2026-09-12);
     * here the tensor merely stays on the interleaved map, but a placement decision made
     * by default rather than by the compiler must still be audible. */
    if (!warned.no_layout_table++)
      fprintf(stderr, "[pim-runtime] WARN kernel carries no layout table; every tensor "
                      "stays on the interleaved map and the host roles stand.\n");
    return;
  }
  const int32_t *table = __pim_layout_table;
  for (int i = 0; i < n; i++) {
    const int32_t *rec = table + (size_t)i * PIM_LAYOUT_REC_WORDS;
    if (rec[PIM_LW_OPERAND_ARG] != tensor_id)
      continue;
    if (stats.layout_from_compiler == 0)
      fprintf(
          stderr,
          "[pim-runtime] compiler-emitted layout table found in the kernel "
          "artifact (%d entries); host ctypes descriptor push is not used\n",
          n);
    /* COMPILER-DERIVED ROLE. Whether a tensor is replicated across banks
     * is a property of the LAYOUT, so the pass reads it off the encoding and we
     * honor it over the role the host asserted at registration. The host string
     * was written for one particular layout, so any layout change silently
     * inverted the charge: a transposed matmul accumulator measured 31x too fast
     * because both operands ended up in the free role (2026-09-06). The
     * accumulator is exempt -- a store's role is decided by what it is, not by
     * how it is spread. */
    tensor_info_t *_t = &tensors[tensor_id];
    /* Index through PIM_LW_*, never a literal: these shifted when the record lost
     * its reduction_col_axis word on 2026-09-10 and a raw rec[3] silently read
     * num_axes instead, turning every operand into a streamed tensor and dropping
     * the entire W charge (matmul 64x64x32 went 32,786 -> 246 cycles). */
    _t->bank_replicated = (int)rec[PIM_LW_BANK_REPLICATED];
    if (_t->bank_replicated >= 0) {
      /* Reserve a per-lane slab, the same rows in every bank. Partitioned: a lane can
       * touch far more than its even share when lanes overlap (a conv halo lane touched
       * 576 of a 4,096-element input), so a small tensor gets room for every element
       * per lane and only a large one falls back to twice the share. Replicated: every
       * lane holds the whole tensor, so the slab IS the tensor; the copy in bank b
       * used to keep the source word's (sa,row,col) and drop its bank bits, which
       * aliased 4-8 distinct words onto one address (matmul A: 8,192 writes, 2,048
       * addresses). When the rows do not fit, keep the interleaved home and say so. */
      int lanes = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks /
                  g_banks_per_lane;
      int vpr = _t->values_per_row;
      int share2 = 2 * ((_t->num_elements + lanes - 1) / lanes);
      int per_lane = _t->bank_replicated >= 1 ? _t->num_elements
                     : share2 > 65536 ? share2
                     : (_t->num_elements < 65536 ? _t->num_elements : 65536);
      int rows = (per_lane + vpr - 1) / vpr;
      if (g_banks_per_lane > 1) {
        /* Rows are per bank, and each bank of a pair holds every other block of words. */
        int vpc = _t->values_per_col;
        int span = PIM_PAIR_BLOCK_WORDS * g_banks_per_lane;
        int bank_words = ((per_lane + vpc - 1) / vpc + span - 1) / span *
                         PIM_PAIR_BLOCK_WORDS;
        rows = (bank_words * vpc + vpr - 1) / vpr;
      }
      if (rows < 1) rows = 1;
      if (g_lane_row_top < 0)
        g_lane_row_top = cfg_num_sa * cfg_num_rows;
      if (rows > g_lane_row_top) {
        fprintf(stderr, "[pim-runtime] WARN tensor %d: lane slab of %d rows does not fit "
                        "the %d rows left; placement stays on the interleaved map.\n",
                tensor_id, rows, g_lane_row_top);
      } else {
        g_lane_row_top -= rows;
        _t->lane_row_base = g_lane_row_top;
        _t->lane_row_count = rows;
        _t->lane_slot = slot_map_create((size_t)per_lane / 2 + 1);
        if (!_t->lane_slot) {
          _t->lane_row_base = -1;
          fprintf(stderr, "[pim-runtime] WARN tensor %d: slot map allocation failed; "
                          "placement stays on the interleaved map.\n", tensor_id);
        }
      }
    }
    /* THE ROLE IS THE COMPILER'S. is_store marks the output (sticky in the emitter, so a
     * tensor both loaded and stored is the output), bank_replicated says whether every
     * bank sees the same elements. The host string used to be the role and a layout
     * change silently inverted the charge (31x on a transposed accumulator); now it is
     * an input to check, and a disagreement prints a WARN, which the ladder records,
     * instead of costing a day. */
    if (rec[PIM_LW_IS_STORE] == 1 || rec[PIM_LW_BANK_REPLICATED] >= 0) {
      int derived = rec[PIM_LW_IS_STORE] == 1 ? PIM_ROLE_ACCUMULATOR
                    : rec[PIM_LW_BANK_REPLICATED] ? PIM_ROLE_OPERAND : PIM_ROLE_STREAMED;
      if (derived != _t->role) {
        static const char *names[] = {"STREAMED", "OPERAND", "ACCUMULATOR"};
        fprintf(stderr, "[pim-runtime] WARN tensor %d: host role %s, compiler layout says "
                        "%s; taking the compiler's.\n", tensor_id,
                (_t->role >= 0 && _t->role < 3) ? names[_t->role] : "?",
                names[derived]);
        _t->role = derived;
      }
    }
    if (_t->bank_replicated < 0 && _t->role != PIM_ROLE_OPERAND)
      fprintf(stderr, "[pim-runtime] WARN tensor %d (%s): compiler stated no partition "
                      "bit; placement stays on the interleaved map.\n", tensor_id,
              _t->role == PIM_ROLE_ACCUMULATOR ? "ACCUMULATOR" : "STREAMED");
    stats.layout_from_compiler++;
    return;
  }
  /* A table exists but has no record for this tensor: same gap, same warning. */
  if (tensors[tensor_id].role != PIM_ROLE_OPERAND)
    fprintf(stderr, "[pim-runtime] WARN tensor %d: no layout record; placement stays "
                    "on the interleaved map.\n", tensor_id);
}

/* Shared tail for every pim_register_tensor return path. */
static int pim_finish_register(void) {
  int tid = num_tensors++;
  apply_compiler_layout(tid);
  return tid;
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
  t->bank_replicated = -1;
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

  int total_flat_banks =
      cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
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
      (uint64_t)t->values_per_row * (uint64_t)total_flat_banks;

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

  const char *role_str = (role == PIM_ROLE_STREAMED)  ? "STREAMED"
                         : (role == PIM_ROLE_OPERAND) ? "OPERAND"
                                                      : "ACCUMULATOR";
  fprintf(stderr,
          "[pim-runtime] Tensor %d: %s, %d elems (%zu bytes), interleaved "
          "linear_base=%llu (covers %llu global-row units)\n",
          num_tensors, role_str, total, t->total_bytes,
          (unsigned long long)t->layout_linear_base,
          (unsigned long long)((total + elems_per_global_row - 1) /
                               elems_per_global_row));
  return pim_finish_register();
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
    if (t->bank_replicated < 0 || t->lane_row_base < 0)
      continue;
    int lanes = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks /
                g_banks_per_lane;
    fprintf(stderr, "[pim-runtime]   tensor %d slab slots per lane:", i);
    for (int l = 0; l < lanes && l < MAX_BANKS; l++)
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
            " (PIM_COALESCE_MAX is a compile-time #define, not an env var)\n",
            stats.coalesce_overflow);

  destroy_dedup_state();
  initialized = 0;
}

/* ================================================================
 *  LLVM pass hooks
 * ================================================================ */

/*
 * Translate a single memory access into PIM trace operations.
 *
 * COMPUTE phase (PIM PE active):
 *   STREAMED load   -> BR (bank-read: data streams through PE)
 *   OPERAND load    -> W  (write: data to PE register via bus)
 *   ACCUMULATOR load  -> R  (read: partial sum from PE)
 *   ACCUMULATOR store -> BW (bank-write: result to bank)
 *
 * IDLE phase: no trace emission.
 */
/* Process a single logical element access against an already-resolved tensor.
 * Returns nothing; updates stats + writes at most one trace line. */
static void pim_trace_access_one(tensor_info_t *t, uint64_t addr,
                                 int is_write) {
  pim_phys_loc_t loc;
  if (!resolve_access_location(t, addr, &loc)) {
    return;
  }

  /* OPERAND LOADS NOW COLLAPSE. 2026-09-18.
   *
   * They used to bypass, paying one write per RECEIVING BANK (32x here), to stay
   * "symmetric" with OptiPIM on the belief that "both stacks charge one write per
   * receiving bank". THAT PREMISE IS FALSE, measured on 40 shapes. OptiPIM's shared
   * config sets single_bank_opt, and their fimdram.cpp:277-282 breaks out of the spatial
   * loop after the leader bank, so their trace carries ONE bank's commands and the other
   * 31 are assumed to run in lockstep for free:
   *   matmul_1024x512x32  their writes 16,384 = banks x columns EXACTLY
   *   matmul_256x512x64   A is 16,384 elements, their writes 32,768 = ~2 per element
   *   38 conv shapes      writes per input element 0.06x-2.3x, clustered near 0.5x
   * Nothing near the 32x that per-receiving-bank charging produces.
   *
   * The hardware agrees with the collapse, not the exemption: real HBM-PIM has an
   * all-bank GRF broadcast (Lee et al. ISCA'21 III-A/III-B), which the old comment
   * conceded while giving it up anyway.
   *
   * Scope is unchanged and load-bearing: cur_dispatch rides in the key below, so bank
   * replicas of ONE instruction fold while a later program instance re-delivering the
   * same operand does not. Collapsing unscoped grants cross-pid operand residency the
   * register file cannot provide (measured 18.8x FASTER than OptiPIM, an artifact). */
  if (g_lockstep_collapse && cur_phase == PIM_PHASE_COMPUTE) {
    /* Scope the collapse to ONE pseudochannel. An all-bank PIM command reaches the
     * banks of a single (ch,pch); replicas in a different pch or channel ride a
     * separate command bus and need their own event. Omitting them collapsed
     * across independent buses and under-charged (2026-09-05 audit). Bank/bg stay
     * OUT of the key: collapsing those IS the all-bank SIMD model. */
    uint64_t tensor_key = lockstep_ns(t, loc.ch, loc.pch, is_write,
                                      g_banks_per_lane > 1 ? (loc.bank & 1) : 0);
    /* Dispatch and lane ordinal ride in the key, so the 32 bank replays of ONE access
     * collapse while two dispatches, or two accesses within a lane, stay distinct. */
    uint64_t key_row = (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    /* OPERANDS KEEP FOLDING WITHIN A DISPATCH. An operand is delivered into GRF and
     * stays for the dispatch, so a second access to its column is a register hit, not a
     * bus transaction; cur_dispatch already separates the re-delivery that IS physical.
     * A streamed load is consumed, so each access is its own command and takes the
     * ordinal. Without this split the ladder's operand writes rose 16x. */
    uint64_t ord = (t->role == PIM_ROLE_OPERAND) ? 0 : g_access_ord;
    uint64_t h = lockstep_key(tensor_key, cur_dispatch, ord, key_row,
                              (uint64_t)loc.col);
    if (!addr_dedup_check_and_mark(g_lockstep_collapse, h, h >> 16, h >> 48)) {
      stats.lockstep_skips++;
      return;
    }
  }

  emit_access_by_role_phase(t, &loc, is_write);
}

/* Cap on distinct DRAM columns one instrumented access can coalesce over. */
#define PIM_COALESCE_MAX 64

/* Process a vector or scalar access spanning [base_addr, base_addr+size).
 * MemTracePass calls this once per IR-level load/store; we expand to one
 * pim_trace_access_one per logical element so multi-bank/multi-row vector
 * loads are correctly accounted for, and the dedup primitive then collapses
 * intra-tile spatial redundancy. */
static void pim_trace_access_range(uint64_t base_addr, uint64_t size,
                                   int is_write, uint64_t compiler_lanes) {
  if (cur_phase == PIM_PHASE_IDLE) {
    stats.ignored++;
    return;
  }

  int tidx = find_tensor(base_addr);
  if (tidx < 0) {
    stats.ignored++;
    return;
  }
  tensor_info_t *t = &tensors[tidx];

  /* Track program-id and tile boundaries, which advance cur_dispatch in the lockstep
   * key. The ordinal advance follows the epoch because it keys on cur_dispatch. */
  if (cur_phase == PIM_PHASE_COMPUTE) {
    advance_program_epoch_if_needed();
    access_ord_next();
  }

  int elem_size = t->elem_size > 0 ? t->elem_size : 1;
  uint64_t n_elements = size / (uint64_t)elem_size;
  if (n_elements == 0)
    n_elements = 1;

  /* The compiler told us how many lanes this access moves. We derived the same
   * number from the HOST-registered element size. They are the same physical
   * quantity decided in two places, so say so when they differ instead of pricing
   * a width the artifact never emitted. Disagreement means the registered dtype
   * and the IR element type have parted company. */
  if (compiler_lanes && compiler_lanes != n_elements) {
    if (!warned.lane_count) {
      warned.lane_count = 1;
      fprintf(stderr,
              "[pim-runtime] WARN: lane-count disagreement. The compiler emitted "
              "%llu lanes for a %llu-byte access; elem_size=%d makes that %llu. "
              "Pricing the runtime's count.\n",
              (unsigned long long)compiler_lanes, (unsigned long long)size,
              elem_size, (unsigned long long)n_elements);
    }
  }

  /* WITHIN-CALL DQ-WORD COALESCING for loads. One __mem_trace_load call is ONE
   * machine instruction. A vector load of 8 fp16 values is one 128-bit bus
   * transaction, not eight, so it must cost one command per distinct DRAM column it
   * touches.
   *
   * LOADS ONLY. Stores need no twin: two lanes of one store at the same
   * (bank,sa,row,col) share the lockstep key, so the collapse downstream folds them.
   *
   * STRICTLY within the call, and that is the whole point. Two SEPARATE scalar loads
   * of the same word, issued far apart, still pay twice, because by then the word is
   * gone. That is the k-outer penalty and it is real physics: the k-outer matmul
   * measures 0 memo fires against the k-packed kernel's 7/8, an 8x difference that
   * the compiler earns by consuming word-mates together. Grouping across calls would
   * hand k-outer a discount it never earned.
   *
   * Inert for scalar loads: n_elements == 1 takes the fast path below untouched, so
   * this changes nothing until a kernel actually emits vector loads. */
  if (!is_write && n_elements > 1 && cur_phase == PIM_PHASE_COMPUTE) {
    uint64_t seen[PIM_COALESCE_MAX];
    int n_seen = 0;
    for (uint64_t i = 0; i < n_elements; i++) {
      uint64_t elem_addr = base_addr + i * (uint64_t)elem_size;
      if (elem_addr >= (uint64_t)t->base_addr + t->total_bytes) {
        int ntidx = find_tensor(elem_addr);
        if (ntidx < 0) {
          stats.ignored++;
          continue;
        }
        t = &tensors[ntidx];
      }
      int elem_idx =
          (int)((elem_addr - (uint64_t)t->base_addr) / t->elem_size);
      if (elem_idx < 0 || elem_idx >= t->num_elements) {
        stats.ignored++;
        continue;
      }
      /* Key on the PLACED tuple. Keying on the interleaved home dropped every
       * word-mate before it reached place_in_lane_slab, so a partitioned tensor's
       * slab held one slot per word instead of one per element and came out 2-8x
       * too few columns (matmul B read 16 BR where its 16 words per lane owe 32). */
      pim_phys_loc_t pl;
      pl.elem_idx = elem_idx;
      map_element_interleaved(t, elem_idx, &pl.ch, &pl.pch, &pl.bg, &pl.bank, &pl.sa,
                              &pl.row, &pl.col);
      if (t->bank_replicated >= 0 && t->lane_slot && t->lane_row_base >= 0)
        place_in_lane_slab(t, &pl);
      int gb = compute_global_bank(pl.ch, pl.pch, pl.bg, pl.bank);
      uint64_t key = ((uint64_t)gb & 0xFFFFULL) |
                     (((uint64_t)pl.sa & 0xFFULL) << 16) |
                     (((uint64_t)pl.row & 0xFFFFFFULL) << 24) |
                     (((uint64_t)pl.col & 0xFFFFULL) << 48);
      int dup = 0;
      for (int j = 0; j < n_seen; j++)
        if (seen[j] == key) { dup = 1; break; }
      if (dup) {
        stats.load_coalesced++;
        continue;
      }
      if (n_seen < PIM_COALESCE_MAX)
        seen[n_seen++] = key;
      else
        stats.coalesce_overflow++;
      pim_trace_access_one(t, elem_addr, is_write);
    }
    return;
  }

  for (uint64_t i = 0; i < n_elements; i++) {
    uint64_t elem_addr = base_addr + i * (uint64_t)elem_size;
    /* Re-resolve tensor if the vector spans tensor boundaries. Common case
     * (single-tensor vector) keeps the cached pointer. */
    if (elem_addr >= (uint64_t)t->base_addr + t->total_bytes) {
      int ntidx = find_tensor(elem_addr);
      if (ntidx < 0) {
        stats.ignored++;
        continue;
      }
      t = &tensors[ntidx];
    }
    pim_trace_access_one(t, elem_addr, is_write);
  }
}

void __mem_trace_load(void *addr, uint64_t size, uint64_t lanes) {
  if (!trace_fp)
    return;
  pim_trace_access_range((uint64_t)addr, size, 0, lanes);
}

void __mem_trace_store(void *addr, uint64_t size, uint64_t lanes) {
  if (!trace_fp)
    return;
  pim_trace_access_range((uint64_t)addr, size, 1, lanes);
}

