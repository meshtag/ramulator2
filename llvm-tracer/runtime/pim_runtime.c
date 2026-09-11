#include "pim_runtime.h"
#include "im_addr_dedup.h"
#include "im_runtime.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Compiler-emitted descriptor globals. Included with the other headers because the
 * env-reading code in pim_init references __pim_bg_interleave, which sits far above
 * where this used to be included. */
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
static int cfg_num_channels = 1;
static int cfg_num_pch = 2;
static int cfg_num_bg = 4;
static int cfg_num_banks = 4; /* per bank group */
/* HAND-COPY of the DRAM organization the SIMULATOR decodes with. The tracer builds
 * addresses from these; ramulator reads them from the org preset named in the shared
 * spec (HBM3_8Gb -> {1, 2, 4, 4, 64 subarrays, 512 rows, 64 cols} in
 * third_party/ramulator2/src/dram/impl/HBM3_PIM.cpp). Nothing checks the two agree.
 * They did not until 2026-09-10: cfg_num_sa was 16, which is the HBM3_2Gb preset, so
 * the tracer modelled a quarter of the configured part's per-bank rows. Harmless as
 * it stood, because the subarray count only sets the capacity ceiling below and takes
 * no part in the address slicing, but a tensor needing more than 16*512 rows per bank
 * was refused on a limit the machine does not have. */
static int cfg_num_sa = 64;
static int cfg_num_rows = 512;
static int cfg_num_cols = 64;
static int cfg_dq_bits = 128;
static void check_compiler_dq_bits(int cfg_bits);

/* Width of one modelled DATA VALUE in bits. This is a HARDWARE property (the shared
 * spec's data_width / pe_bits, 16 on this HBM3-PIM part), NOT the width of whatever
 * the host happens to store the test arrays in. Column packing must follow the
 * hardware: values_per_col = dq_bits / data_width. Deriving it from the numpy dtype
 * instead made us pack 4 int32 per column where the spec, and OptiPIM, pack 8, so we
 * paid double the bank-reads for the same tensor (2026-09-06). Override with
 * PIM_DATA_WIDTH_BITS. */
static int cfg_data_width_bits = 16;

/* Placement scheme. Selectable at init time via PIM_LAYOUT env var:
 *
 *   "striped"     — legacy. Compact: small tensors live in one bank.
 *                  Striped: large tensors stripe logical rows across
 *                  banks round-robin. Each "logical row" of a tensor is
 *                  owned by exactly one bank. Uses generic modulo
 *                  arithmetic, so it accepts non-power-of-2 bank/BG/col
 *                  counts.
 *
 *   "interleaved" (default) — bit-interleaved scheme8-style placement.
 *                  Sequential elements fill within-DQ → col →
 *                  bank-within-BG → BG → pch → row. Banks within a BG
 *                  hold *adjacent* element ranges at the same physical
 *                  row, enabling parallel servicing across the
 *                  cfg_num_banks members of a BG. Modeled after
 *                  Samsung's published HBM-PIM bit layout.
 *
 *                  Constraint: every level the bit-decomposition uses
 *                  (cfg_num_channels, cfg_num_pch, cfg_num_bg,
 *                  cfg_num_banks, cfg_num_cols, values_per_col) MUST
 *                  be a power of 2. pim_init enforces this and aborts
 *                  loudly otherwise. For non-power-of-2 configs (e.g.
 *                  exotic ablations with 3 or 5 banks per BG), use
 *                  PIM_LAYOUT=striped instead.
 *
 * The kernel is unaffected by the choice (it reads from host memory by
 * linear index); only the trace's reported (bank, row, col) tuples
 * change, which in turn changes how Ramulator schedules the simulation. */
typedef enum {
  PIM_LAYOUT_STRIPED = 0,
  PIM_LAYOUT_INTERLEAVED = 1,
} pim_layout_scheme_t;

static pim_layout_scheme_t cfg_layout_scheme = PIM_LAYOUT_STRIPED;

/* Where each tensor starts. DQ = align to the data-bus word only, so consecutive
 * tensors share rows but land on different banks; GLOBAL_ROW = align to a whole
 * global row, so every tensor starts at bank 0. The compiler states which
 * (__pim_placement_align); this is the runtime's fallback when it says nothing. */
typedef enum { PIM_ALIGN_DQ = 1, PIM_ALIGN_GLOBAL_ROW = 2 } pim_place_align_t;
static pim_place_align_t cfg_place_align = PIM_ALIGN_DQ;

/* Forward decl from im_runtime — provides the host-loop bank index for
 * the currently-executing kernel call. Used by duplicated-tensor reads
 * (PIM_DUPLICATE_BCAST=1) to find the host PE's local BG copy. */
extern int32_t __pim_get_bank_id(void);

/* In interleaved mode, the next free linear-element position. Tensors are
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
  // The maximum number of dimensions is 4 for now.
  int dims[4];
  int num_elements;
  pim_role_t role;

  /* Physical placement in HBM */
  int assigned_ch;
  int assigned_pch;
  int assigned_bg;
  int assigned_bank;
  int base_row; /* linear row across all subarrays in the bank */
  int striped;
  int total_flat_banks;
  int base_rows[MAX_BANKS];
  int values_per_col;
  int values_per_row;

  /* Placement scheme this tensor was registered under. For
   * PIM_LAYOUT_INTERLEAVED, layout_linear_base is the linear-element
   * offset into the global bit-interleaved address space; element-to-
   * physical mapping uses bit-shifts on (layout_linear_base + elem_idx). */
  pim_layout_scheme_t layout_scheme;
  uint64_t layout_linear_base;

  /* (bcast_scalar / duplicated / dup_row_base fields removed 2026-06-22 with
   * the broadcast-scalar / row-duplicate blank-dedup machinery.) */

  /* Reuse-as-layout descriptor from the compiler's
   * im-operand-residency-layout decisions. UNSET is the zero-initialized
   * default and means no residency. */

  /* Per-tensor diagnostic counters (Stage 0). Emitted = trace lines actually
   * written; dedup_skips = times check_and_mark returned 0 for this tensor;
   * range_calls = number of __mem_trace_load/store invocations into this
   * tensor (vs. expanded element accesses). */
  uint64_t emitted_br;
  uint64_t emitted_bw;
  uint64_t emitted_r;
  uint64_t emitted_w;
  uint64_t dedup_skips;
  uint64_t range_calls;
} tensor_info_t;

static tensor_info_t tensors[MAX_TENSORS];
static int num_tensors = 0;

/* Namespace bits for the lockstep-collapse key.
 *
 * CRITICAL: im_addr_dedup's pack_keys() keeps only the LOW 16 BITS of k1 (k2 gets
 * 32, k3 gets 16). Anything above bit 15 here is silently dropped, so this must
 * stay inside 16 bits. tid < MAX_TENSORS(16) = 4b, ch < 32 = 5b, pch < 4 = 2b,
 * write = 1b, total 12b.
 *
 * ch/pch are in the key because an all-bank PIM command reaches the banks of ONE
 * pseudochannel; replicas in another pch or channel ride a separate command bus.
 * Bank/bankgroup stay OUT: collapsing those IS the all-bank SIMD model.
 * The write bit separates stores from loads, without which an ACCUMULATOR load
 * (the read half of a read-modify-write drain) marks the key the store then
 * consults, making partial-sum write-back free. Both found 2026-09-05. */
static inline uint64_t lockstep_ns(const tensor_info_t *t, int ch, int pch,
                                   int is_write) {
  return ((uint64_t)(t - tensors) & 0xFULL) | (((uint64_t)ch & 0x1FULL) << 4) |
         (((uint64_t)pch & 0x3ULL) << 9) |
         ((uint64_t)(is_write ? 1 : 0) << 11);
}

/* ================================================================
 *  Runtime state
 * ================================================================ */
static FILE *trace_fp = NULL;

/* Bankgroup-interleave lever (PIM_BG_INTERLEAVE=1, default OFF). Places the
 * bankgroup bits BELOW the column bits in the interleaved element->physical map
 * so consecutive elements round-robin the bankgroups. Consecutive column
 * commands then hit DIFFERENT bankgroups (nCCDS spacing) instead of the same
 * one (nCCDL) -> up to ~2x column throughput on the streamed reduction operand.
 * Pure address-bit reorder over power-of-two hardware dims (bijective ->
 * correctness-invariant; only ramulator cycles change). The row bits are the
 * same high bits either way, so row-buffer locality is preserved. */
static int g_bg_interleave = 0;

/* (Runtime operand residency retired 2026-09-04: reuse is now expressed by the
 * kernel tile and, for broadcast operands, by the WB bus-broadcast collapse. The
 * per-role residency gates, the per-tensor LRU register cache, and the host/compiler
 * capacity plumbing were deleted with it. What remains is role -> BR/W/BW/WB charging
 * only. The layout_kind word went 2026-09-10 with the broadcast collapse it drove.) */
static pim_phase_t cur_phase = PIM_PHASE_IDLE;
static int initialized = 0;
static int finalized = 0;

/* Per-bank linear row allocation, indexed by flat bank id across all
 * ch/pch/bg/bank */
static int next_free_row[MAX_BANKS];

/* (g_next_free_dup_row / DUP row allocator removed 2026-06-22 with the
 * row-duplicate blank-dedup machinery.) */

/* Statistics */
static uint64_t stat_bank_reads = 0;
static uint64_t stat_bank_writes = 0;
static uint64_t stat_reads = 0;
static uint64_t stat_writes = 0;
static uint64_t stat_ignored = 0;
/* Loads folded by the within-call DQ-word coalescer: lanes of ONE vector load that
 * land on the same DRAM column. This is the whole mechanism now. The runtime memo that
 * used to infer the same grouping from access order was deleted 2026-09-09 once the
 * kernels expressed it in the IR (fp16 makes a 128-bit DQ word exactly 8 values). */
static uint64_t stat_load_coalesced = 0;
/* Distinct keys past the 64-entry per-call buffer, i.e. coalescing this call could
 * not complete. Must stay 0; nonzero means a kernel emits a vector wider than the
 * buffer and some duplicate lanes were charged twice. */
static uint64_t stat_coalesce_overflow = 0;

/* (g_bcast_scalar_enabled / broadcast-scalar enable flag removed 2026-06-22.) */

/* Per-pid residency reset (DEFAULT ON as of 2026-07-07). Real hardware has NO
 * cross-program-id operand reuse: GPU registers/shared-memory are strictly
 * intra-thread-block (grid-wide register reuse does not exist; cross-block reuse
 * is only opportunistic L2, which neither Aquabolt-XL nor SIMDRAM has), and the
 * PIM register file is reloaded per SIMD dispatch. Cross-tile operand reuse would
 * require a persistent kernel (one pid iterating many tiles = intra-pid reuse),
 * not automatic cross-pid residency. The broadcast collapse honors this: the
 * lockstep set resets per program-id, so a broadcast operand re-broadcasts once
 * per pid wave (no unphysical cross-pid reuse). */

/* (g_acc_resident_enabled + stat_acc_resident_skips removed 2026-06-26: the
 * accumulator-read reuse dedup was proven byte-identical inert — the psum is
 * loop-carried SSA in codegen and never round-trips to DRAM, so there are no
 * ACCUMULATOR loads to collapse. Accumulator residency is realized in compiler
 * codegen, not a runtime reuse skip.) */

/* (g_per_bg_bcast_enabled + g_duplicate_bcast_enabled removed 2026-06-22 with
 * the broadcast / row-duplicate blank-dedup machinery.) */

/* Lockstep collapse dedup (default ON, disable with PIM_LOCKSTEP_COLLAPSE=0).
 *
 * Models the fact that PIM-SIMD hardware fires ONE controller dispatch per
 * SIMD instruction; all N banks execute in parallel. Per-event trace
 * emission inflates each SIMD dispatch to N events (one per bank), which
 * Ramulator's controller dispatches FIFO at 1/cycle — overcounting dispatch
 * cost by N×.
 *
 * Dedup key: (tensor_id, sa, row, col). The existing g_dedup keys on
 * (gbank, row, col), which lets all N lockstep replicas through (different
 * gbanks). This NEW state keys on (sa, row, col) IGNORING (ch, pch, bg,
 * bank), so all replicas at the same physical (sa, row, col) within the
 * same tensor collapse to one event — matching OptiPIM's single_bank_opt.
 *
 * Scope: per-program-id (resets on pid change, just like g_dedup). The
 * tensor_id scoping prevents accidental collapse across different tensors
 * that happen to share (sa, row, col) on different banks.
 *
 * BYPASS: OPERAND-role loads skip this dedup. OptiPIM's PimCodeGen emits
 * per-PE register-load commands (priority-read + N writes) for operand
 * loads even with single_bank_opt; collapsing them to 1 under-models the
 * bus-mediated transfer cost. Letting the per-bank events through keeps
 * the per-PE register-load cost in our trace at OptiPIM's abstraction
 * level. STREAMED/ACCUMULATOR tensors still collapse — those are
 * genuine bank-parallel reads, not bus broadcasts. */
/* ============================================================================
 * RUNTIME ACCESS-COLLAPSE TAXONOMY — what suppresses a trace event, and why.
 * NONE of these are the (removed) "blank reuse dedups"; they are, distinctly:
 *
 *   1. LOCKSTEP COLLAPSE  (g_lockstep_collapse)  — EMULATION ARTIFACT, not reuse.
 *      Collapses the bank-replicas the host-replay over-emits (`for pid: for
 *      bank: kernel()` runs each kernel ~32x/phase) into one SIMD dispatch ==
 *      OptiPIM single_bank_opt. Faithful HW model; keep.
 *   (A BROADCAST COLLAPSE keyed on the logical element used to sit here. Its
 *      ROW_DUP layout_kind was deleted in 36a995c and the residual dead branch in
 *      2026-09-11, so a broadcast operand is now indistinguishable from any other
 *      operand load and pays per receiving bank. Reason is baseline symmetry, not
 *      hardware: see the note at the collapse site. Do not re-add without a
 *      matching change on OptiPIM's side, which is not ours to make.)
 *   (The per-pid accumulator store dedup that used to sit here was DELETED
 *      2026-09-09: it was subsumed by the lockstep collapse, whose gate is a
 *      superset and whose key is strictly coarser, so it could never fold anything
 *      lockstep would not. Verified fold-for-fold, cycles unchanged.)
 *
 * `addr_dedup_*` is a generic hash-set HELPER (im_addr_dedup.c) backing #1, and #2
 * through it. That is why the token "dedup" appears throughout the file. It is the
 * data structure's name, not N separate reuse dedups.
 * ============================================================================ */
static addr_dedup_state_t *g_lockstep_collapse = NULL;

/* Bus-word memo, per (tensor, receiving bank). One command moves one DQ word, so a
 * value already in the word just transferred needs no new command. A MEMO on the
 * immediately preceding access, never a set: a set that remembers every word ever
 * touched is a cache, and one here cut a small matmul 22-86x where the 8:1 bus can
 * justify 8x. This pays only when the tensor's LAYOUT makes lane-axis accesses
 * contiguous; with row-major A it fires 0/262144 times, which is correct.
 * DELETED 2026-09-09. The kernels now express the DQ word as a vector load, and
 * pim_trace_access_range coalesces within a single load
 * call, so this access-order memo had nothing left to do: matmul, matvec and conv all
 * measured byte-identical with it disabled before it was removed. */

/* OPTIPIM-COMPARABLE ADDRESSING (PIM_OPTIPIM_ADDRESSING=1, DEFAULT OFF).
 *
 * NOT FAITHFUL. This exists only so our trace can be compared like-for-like with
 * OptiPIM's, which pushes its output reads and operand writes at the unmodified
 * base address with row and column left at zero (fimdram.cpp:44 and :69). Every such
 * request therefore lands in one already-open row of one bank and never pays an
 * activation, which understates its cycles by whatever the real row-miss traffic
 * would cost. Measured directly against this simulator: a row-hit bank read is 1.00
 * cycle, a same-bank row miss 28.0.
 *
 * The DEFAULT is honest accounting: real addresses, real row misses. Turn this on to
 * answer "what would we score under their addressing", never to report a number. */
static int g_optipim_addressing = 0;

/* ACCUMULATOR SPILL (pim_set_acc_spill, default inactive).
 *
 * The 2-D matmul kernel holds acc as a BLOCK_M x BLOCK_N register tensor across the
 * whole K loop, i.e. BM*BN/banks values per PE against the PE register file.
 * When the tile overflows, hardware must write the excess back
 * to DRAM and re-read it every K step. Nothing in the kernel's memory trace shows
 * that, so an over-capacity tile used to look FREE, and the harness refused such
 * tiles rather than report a fictitious number.
 *
 * Refusing hides a real tradeoff: a larger tile costs spill traffic but saves operand
 * re-delivery, and which wins is an empirical question the search should decide. So
 * the overflow is charged instead: per K step, `overflow` values are written back and
 * read again, emitted as ordinary BW/BR at real accumulator addresses so Ramulator
 * prices the row behaviour rather than a scalar being added to a total.
 *
 * MODELLED, NOT OBSERVED. The kernel does not really perform these accesses; the
 * count comes from the tile geometry the host declares. That is a weaker footing than
 * the rest of the trace and it is why this is stated explicitly here. */
static void emit_acc_spill(void);
static int g_acc_spill_tensor = -1;
static int g_acc_spill_overflow = 0;   /* values per PE beyond the register file */
static int g_acc_spill_ksteps = 0;
static uint64_t stat_acc_spill_emitted = 0;
static int g_lockstep_enabled = 1;
static uint64_t stat_lockstep_skips = 0;

/* Per-program-id store dedup for ACCUMULATOR tensors.
 *
 * Modeling assumption:
 *   A Triton program-id is one logical unit of work (one output tile). In the
 *   common kernels we model here (matvec/matmul/conv/elemwise), the tile's
 *   partial sums live in registers during COMPUTE and are materialized to the
 *   ACCUMULATOR tensor once at the end of that program-id. Under that
 *   write-once-per-output-tuple model, multiple IR-level stores that resolve
 *   to the same physical (global_bank, linear_row, col) within one program-id
 *   are redundant from the DRAM-trace point of view and should count as one
 *   BW emission.
 *
 * Why duplicates happen even when the kernel is logically "write once":
 *   - Scalarization/vectorization can expose multiple IR stores that still
 *     target one physical column.
 *   - Adjacent vector lanes may collapse to the same physical tuple when
 *     values_per_col > 1.
 *   - A later store instruction in the same program-id may revisit a tuple a
 *     previous store instruction already materialized.
 *
 * What this does:
 *   The first store to a tuple in a program-id emits BW; later stores to that
 *   same tuple in the same program-id are dropped.
 *
 * When this is safe:
 *   For the current benchmark kernels, the kernels compute one output tile per
 *   program-id, keep the accumulator live, then drain it. They do not
 *   intentionally perform two semantically distinct writes to the same output
 *   tuple within one program-id.
 *
 * When this would be wrong:
 *   If a future kernel intentionally writes the same physical output location
 *   twice within one program-id and both writes should incur DRAM cost
 *   separately, this dedup would undercount. Examples include explicit
 *   spill/reload/drain patterns, multi-phase in-place updates, atomics, or any
 *   kernel whose memory semantics are not "final output materialization".
 *
 * Reset cadence:
 *   Reset on any program-id axis change (same cadence as g_dedup, the
 *   per-program-id load state).
 */

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

/* Bits reserved for the dispatch id above the linear row inside the lockstep k2.
 * k2 is 32 bits (im_addr_dedup.c pack_keys); the linear row needs
 * ceil(log2(cfg_num_sa * cfg_num_rows)) of them. Computed at init. */
static int g_dispatch_shift = 15;

/* Bits of the dispatch field given to the tile index, with the program epoch
 * above it. Packed rather than summed: summing let instance p's tile 1 collide
 * with instance p+1's tile 0, so two distinct dispatches folded into one and the
 * trace under-emitted. Both components are policed at pack time. */
#define PIM_TILE_BITS 10
static int g_warned_dispatch_overflow = 0;

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
    if (!g_warned_dispatch_overflow &&
        ((__pim_persistent && tile >= (1ULL << PIM_TILE_BITS)) ||
         packed >= (1ULL << (32 - g_dispatch_shift)))) {
      g_warned_dispatch_overflow = 1;
      fprintf(stderr,
              "[pim-runtime] WARN: dispatch id does not fit (%llu tiles, epoch "
              "%llu, %d bits). Ids alias and distinct dispatches collapse.\n",
              (unsigned long long)tile, (unsigned long long)epoch,
              32 - g_dispatch_shift);
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
  int nb = cfg_num_banks > 0 ? cfg_num_banks : 1;
  int nbg = cfg_num_bg > 0 ? cfg_num_bg : 1;
  int npch = cfg_num_pch > 0 ? cfg_num_pch : 1;
  if (g < 0) g = 0;
  *bank = g % nb;  g /= nb;
  *bg   = g % nbg; g /= nbg;
  *pch  = g % npch; g /= npch;
  *ch   = g;
}

static void decode_flat_bank(int global_bank, int *ch, int *pch, int *bg,
                             int *bank) {
  int banks_per_pch = cfg_num_bg * cfg_num_banks;
  int banks_per_ch = cfg_num_pch * banks_per_pch;

  *ch = global_bank / banks_per_ch;
  *pch = (global_bank / banks_per_pch) % cfg_num_pch;
  *bg = (global_bank / cfg_num_banks) % cfg_num_bg;
  *bank = global_bank % cfg_num_banks;
}

/* Bit-interleaved (scheme8-like) element-to-physical mapping.
 *
 * Address bit layout (LSB → MSB), each field is sized to the runtime
 * configuration via log2 of cfg_*. There is NO hardcoded bank count —
 * the field widths shift to match cfg_num_banks, cfg_num_bg, cfg_num_pch,
 * cfg_num_channels, cfg_num_cols, and values_per_col at runtime. All of
 * those must be powers of two; pim_init enforces this when interleaved
 * mode is selected.
 *
 *   [ within-DQ | col | bank-within-BG | BG | pch | chan | linear_row ]
 *
 * Sequential element indices first fill the within-DQ slot (vpc values),
 * then advance through cols, then through banks within a BG, then BGs,
 * then pch, then channels, finally advancing linear_row (sa*num_rows + row).
 *
 * Worked example for the project's default 1-channel HBM3 config
 * (vpc=4, num_cols=64, cfg_num_banks=4 banks/BG, cfg_num_bg=4 BGs,
 *  cfg_num_pch=2, cfg_num_channels=1):
 *   bits 0..1   within-DQ (4 values per col)
 *   bits 2..7   col (64 cols)
 *   bits 8..9   bank within BG (4)
 *   bits 10..11 BG (4)
 *   bit  12     pch (2)
 *   (no chan bits for cfg_num_channels=1)
 *   bits 13..   linear_row
 * Under that config: elements 0..255 lie within bank 0 of BG 0 of pch 0
 * row 0; 256..511 advance to bank 1 of BG 0; after all 4 banks of BG 0
 * (1024 elements) advance to BG 1; etc.
 *
 * Vary cfg_num_banks/cfg_num_bg/etc. for ablation and the bit slice
 * widths re-shape automatically — no code change required, only that
 * the new values are powers of two. */
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

  if (g_bg_interleave) {
    /* Bankgroup on the LOW bits: consecutive elements round-robin the
     * bankgroups so consecutive column commands hit different BGs (nCCDS)
     * rather than the same BG (nCCDL). The row bits (high) are unchanged, so
     * row-buffer locality is preserved; only col/bank/bg reassign among the
     * same low bits (a bijection -> correctness-invariant). */
    *bg = (int)(linear & ((1ULL << log2_bg) - 1));
    linear >>= log2_bg;

    *col = (int)(linear & ((1ULL << log2_cols) - 1));
    linear >>= log2_cols;

    *bank = (int)(linear & ((1ULL << log2_banks_bg) - 1));
    linear >>= log2_banks_bg;
  } else {
    *col = (int)(linear & ((1ULL << log2_cols) - 1));
    linear >>= log2_cols;

    *bank = (int)(linear & ((1ULL << log2_banks_bg) - 1));
    linear >>= log2_banks_bg;

    *bg = (int)(linear & ((1ULL << log2_bg) - 1));
    linear >>= log2_bg;
  }

  *pch = (int)(linear & ((1ULL << log2_pch) - 1));
  linear >>= log2_pch;

  *ch = (int)(linear & ((1ULL << log2_chan) - 1));
  linear >>= log2_chan;

  /* Whatever remains is the linear row index across all subarrays. */
  int linear_row = (int)linear;
  *sa = linear_row / cfg_num_rows;
  *row = linear_row % cfg_num_rows;
}

/*
 * Map a tensor element index to physical HBM coordinates.
 * Dispatches between the legacy striped scheme and the bit-interleaved
 * scheme based on the per-tensor layout_scheme field.
 */
static void map_element(const tensor_info_t *t, int elem_idx, int *ch, int *pch,
                        int *bg, int *bank, int *sa, int *row, int *col) {
  /* (per-BG duplicated-tensor routing removed 2026-06-22 with the
   * row-duplicate blank-dedup machinery; t->duplicated is always 0.) */
  if (t->layout_scheme == PIM_LAYOUT_INTERLEAVED) {
    map_element_interleaved(t, elem_idx, ch, pch, bg, bank, sa, row, col);
    return;
  }

  if (!t->striped) {
    *ch = t->assigned_ch;
    *pch = t->assigned_pch;
    *bg = t->assigned_bg;
    *bank = t->assigned_bank;

    int linear_row = t->base_row + (elem_idx / t->values_per_row);
    *sa = linear_row / cfg_num_rows;
    *row = linear_row % cfg_num_rows;
    *col = (elem_idx % t->values_per_row) / t->values_per_col;
    return;
  }

  int logical_row = elem_idx / t->values_per_row;
  int global_bank = logical_row % t->total_flat_banks;
  decode_flat_bank(global_bank, ch, pch, bg, bank);

  int linear_row =
      t->base_rows[global_bank] + (logical_row / t->total_flat_banks);
  *sa = linear_row / cfg_num_rows;
  *row = linear_row % cfg_num_rows;
  *col = (elem_idx % t->values_per_row) / t->values_per_col;
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

/* Resolve one logical element access to its physical HBM tuple.
 *
 * Returns 1 on success and fills `loc`. Returns 0 when the current phase
 * should not emit anything or the address falls outside the tensor. In both
 * failure cases, stat_ignored is updated here so callers can simply return. */
static int resolve_access_location(const tensor_info_t *t, uint64_t addr,
                                   pim_phys_loc_t *loc) {
  if (cur_phase == PIM_PHASE_IDLE) {
    stat_ignored++;
    return 0;
  }

  int elem_idx = (int)((addr - (uint64_t)t->base_addr) / t->elem_size);
  if (elem_idx < 0 || elem_idx >= t->num_elements) {
    stat_ignored++;
    return 0;
  }

  loc->elem_idx = elem_idx;
  map_element(t, elem_idx, &loc->ch, &loc->pch, &loc->bg, &loc->bank, &loc->sa,
              &loc->row, &loc->col);
  return 1;
}

/* Emit the trace op implied by the current phase, access direction, and tensor
 * role for an already-resolved physical tuple. */
static void emit_access_by_role_phase(tensor_info_t *t,
                                      const pim_phys_loc_t *loc, int is_write) {
  /* Comparison mode only, applied at the single funnel every record passes through so
   * it cannot be half-applied. See g_optipim_addressing. */
  pim_phys_loc_t _flat;
  if (g_optipim_addressing) {
    _flat = *loc;
    _flat.sa = 0;   /* a different subarray is a different row buffer, so pinning
                     * row alone still leaves misses; OptiPIM's requests all land in
                     * one open row per bank, which needs sa pinned too. */
    _flat.row = 0;
    _flat.col = 0;
    loc = &_flat;
  }
  if (cur_phase == PIM_PHASE_COMPUTE) {
    if (!is_write) {
      /* (duplicated bcast_scalar BR-override removed 2026-06-22.) */
      switch (t->role) {
      case PIM_ROLE_STREAMED:
        emit_trace("BR", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa,
                   loc->row, loc->col);
        stat_bank_reads++;
        t->emitted_br++;
        break;
      case PIM_ROLE_OPERAND: {
        /* An operand load is a bus-mediated write into the RECEIVING PE's register
         * file, so address it at the replay bank rather than at the source
         * element's home bank. Otherwise all N fanout writes pile onto one bank and
         * serialize, which over-charges us relative to OptiPIM, whose codegen writes
         * to global_bank_id + i (fimdram.cpp:63-76) and parallelizes.
         *
         * There is deliberately NO broadcast here. docs/path1-bank-model-scope.md
         * forbade costing an all-bank op as ONE dispatch: it drops below OptiPIM's
         * own single_bank_opt floor, and OptiPIM has no all-bank command to match it
         * (its broadcast shortcut is commented out). The BCAST_W path added
         * 2026-09-04 was that forbidden option and is reverted here (2026-09-05). */
        int dch, dpch, dbg, dbank;
        decompose_global_bank(__pim_get_bank_id(), &dch, &dpch, &dbg, &dbank);
        emit_trace("W", dch, dpch, dbg, dbank, loc->sa, loc->row, loc->col);
        stat_writes++;
        t->emitted_w++;
        break;
      }
      case PIM_ROLE_ACCUMULATOR:
        emit_trace("R", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa,
                   loc->row, loc->col);
        stat_reads++;
        t->emitted_r++;
        break;
      }
    } else {
      if (t->role == PIM_ROLE_ACCUMULATOR) {
        emit_trace("BW", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa,
                   loc->row, loc->col);
        stat_bank_writes++;
        t->emitted_bw++;
      }
      /* Stores to STREAMED/OPERAND in COMPUTE phase are ignored (read-only) */
    }
  } else if (cur_phase == PIM_PHASE_HOST) {
    if (!is_write) {
      emit_trace("R", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa, loc->row,
                 loc->col);
      stat_reads++;
      t->emitted_r++;
    } else {
      emit_trace("W", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa, loc->row,
                 loc->col);
      stat_writes++;
      t->emitted_w++;
    }
  }
}

/* (emit_per_bg_bcast_fanout + alloc_dup_rows_shared removed 2026-06-22 with the
 * broadcast/row-duplicate blank-dedup machinery.) */

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

  /* Read optional overrides from environment */
  const char *v;
  if ((v = getenv("PIM_NUM_CHANNELS")))
    cfg_num_channels = atoi(v);
  if ((v = getenv("PIM_NUM_PCH")))
    cfg_num_pch = atoi(v);
  if ((v = getenv("PIM_NUM_BG")))
    cfg_num_bg = atoi(v);
  if ((v = getenv("PIM_NUM_BANKS")))
    cfg_num_banks = atoi(v);
  if ((v = getenv("PIM_NUM_SA")))
    cfg_num_sa = atoi(v);
  if ((v = getenv("PIM_NUM_ROWS")))
    cfg_num_rows = atoi(v);
  if ((v = getenv("PIM_NUM_COLS")))
    cfg_num_cols = atoi(v);
  if ((v = getenv("PIM_DQ_BITS")))
    cfg_dq_bits = atoi(v);
  check_compiler_dq_bits(cfg_dq_bits);
  if ((v = getenv("PIM_DATA_WIDTH_BITS")))
    cfg_data_width_bits = atoi(v);
  if (cfg_data_width_bits < 1)
    cfg_data_width_bits = 1;

  /* Reduction-col-axis lever (default OFF). */

  /* Bankgroup-interleave lever (default OFF). The compiler decides; the env var is an
   * ablation override. It used to win silently, which is the one compiler/host conflict
   * in this file that neither logged nor aborted, so say so (2026-09-10). */
  if (__pim_bg_interleave)
    g_bg_interleave = 1;
  if ((v = getenv("PIM_BG_INTERLEAVE"))) {
    int env_bg = atoi(v) != 0;
    if (env_bg != g_bg_interleave)
      fprintf(stderr,
              "[pim-runtime] WARN: PIM_BG_INTERLEAVE=%d overrides the compiler's "
              "bg_interleave=%d. Addresses no longer follow the artifact.\n",
              env_bg, g_bg_interleave);
    g_bg_interleave = env_bg;
  }

  /* Placement scheme: PIM_LAYOUT=interleaved (default) | striped
   * Interleaved is bit-interleaved scheme8-like placement; sequential
   * elements walk (within-DQ → col → bank-within-BG → BG → pch → row),
   * giving small/medium tensors natural bank parallelism and avoiding
   * the single-bank hot-spot of the legacy compact path. Striped is the
   * legacy scheme (compact for small tensors, row-stripe for large)
   * retained for ablation and for non-power-of-2 configs. */
  /* Compiler decision first, env second and only as an announced override. Until
   * 2026-09-10 the scheme was a runtime default plus this env var and the artifact had
   * no say, which is the layout-decided-by-the-runtime pattern the project bans. */
  cfg_layout_scheme = PIM_LAYOUT_INTERLEAVED;
  if (__pim_layout_scheme == 1)
    cfg_layout_scheme = PIM_LAYOUT_STRIPED;
  else if (__pim_layout_scheme == 2)
    cfg_layout_scheme = PIM_LAYOUT_INTERLEAVED;
  const char *layout_env = getenv("PIM_LAYOUT");
  if (layout_env) {
    pim_layout_scheme_t want = (strcmp(layout_env, "striped") == 0)
                                   ? PIM_LAYOUT_STRIPED
                                   : PIM_LAYOUT_INTERLEAVED;
    if (__pim_layout_scheme && want != cfg_layout_scheme)
      fprintf(stderr,
              "[pim-runtime] WARN: PIM_LAYOUT=%s overrides the compiler's placement "
              "scheme. Addresses no longer follow the artifact.\n", layout_env);
    cfg_layout_scheme = want;
  }

  cfg_place_align = PIM_ALIGN_DQ;
  if (__pim_placement_align == 2)
    cfg_place_align = PIM_ALIGN_GLOBAL_ROW;
  {
    const char *ae = getenv("PIM_PLACEMENT_ALIGN");
    if (ae) {
      pim_place_align_t want = (strcmp(ae, "global-row") == 0)
                                   ? PIM_ALIGN_GLOBAL_ROW
                                   : PIM_ALIGN_DQ;
      if (__pim_placement_align && want != cfg_place_align)
        fprintf(stderr,
                "[pim-runtime] WARN: PIM_PLACEMENT_ALIGN=%s overrides the compiler's "
                "alignment rule.\n", ae);
      cfg_place_align = want;
    }
  }
  fprintf(stderr,
          "[pim-runtime] placement: scheme=%s align=%s (compiler said scheme=%d "
          "align=%d; 0 = nothing)\n",
          cfg_layout_scheme == PIM_LAYOUT_INTERLEAVED ? "interleaved" : "striped",
          cfg_place_align == PIM_ALIGN_GLOBAL_ROW ? "global-row" : "dq",
          (int)__pim_layout_scheme, (int)__pim_placement_align);
  g_interleaved_next_linear = 0;

  /* Power-of-2 guard for interleaved mode. The bit-interleaved address
   * decomposition shifts by log2(cfg_*) at every level; non-power-of-2
   * sizes silently corrupt the mapping. Fail loudly here rather than
   * produce garbage cycle counts later. */
  if (cfg_layout_scheme == PIM_LAYOUT_INTERLEAVED) {
    struct {
      const char *name;
      int value;
    } pow2_checks[] = {
        {"PIM_NUM_CHANNELS", cfg_num_channels},
        {"PIM_NUM_PCH",      cfg_num_pch},
        {"PIM_NUM_BG",       cfg_num_bg},
        {"PIM_NUM_BANKS",    cfg_num_banks},
        {"PIM_NUM_COLS",     cfg_num_cols},
    };
    for (size_t i = 0; i < sizeof(pow2_checks) / sizeof(pow2_checks[0]); i++) {
      int v = pow2_checks[i].value;
      if (v <= 0 || (v & (v - 1)) != 0) {
        fprintf(stderr,
                "[pim-runtime] ERROR: PIM_LAYOUT=interleaved requires every "
                "level to be a power of two, but %s=%d. Set PIM_LAYOUT=striped "
                "to use the modulo-based placement that accepts arbitrary "
                "counts.\n",
                pow2_checks[i].name, v);
        exit(1);
      }
    }
  }

  num_tensors = 0;
  cur_phase = PIM_PHASE_IDLE;
  memset(next_free_row, 0, sizeof(next_free_row));
  stat_bank_reads = stat_bank_writes = stat_reads = stat_writes = stat_ignored =
      0;
  {
    uint64_t tile = __pim_tile_index();
    uint64_t epoch = __pim_program_epoch;
    uint64_t packed = __pim_persistent ? ((epoch << PIM_TILE_BITS) | tile) : epoch;
    if (!g_warned_dispatch_overflow &&
        ((__pim_persistent && tile >= (1ULL << PIM_TILE_BITS)) ||
         packed >= (1ULL << (32 - g_dispatch_shift)))) {
      g_warned_dispatch_overflow = 1;
      fprintf(stderr,
              "[pim-runtime] WARN: dispatch id does not fit (%llu tiles, epoch "
              "%llu, %d bits). Ids alias and distinct dispatches collapse.\n",
              (unsigned long long)tile, (unsigned long long)epoch,
              32 - g_dispatch_shift);
    }
    /* Only a persistent kernel spends bits on the tile index. A gridded one has
     * none, and shifting its epoch left by 10 overflowed the field at 128
     * instances. The artifact says which shape this is. */
    cur_dispatch = packed;
  }
  /* Reserve the bits the linear row actually needs, then give the rest of k2 to the
   * dispatch id. Warn once if a run could ever have more dispatches than fit: the
   * ids would alias and two dispatches would wrongly collapse into one. */
  {
    long rows = (long)cfg_num_sa * (long)cfg_num_rows;
    int bits = 1;
    while ((1L << bits) < rows) bits++;
    g_dispatch_shift = bits;
    fprintf(stderr,
            "[pim-runtime] lockstep key: %d row bits, %d bits for the dispatch id "
            "(max %ld dispatches before aliasing)\n",
            bits, 32 - bits, 1L << (32 - bits));
  }

  /* (Store write-once and its IM_DEDUP gate removed 2026-09-09, subsumed by the
   * lockstep collapse. IM_DEDUP is no longer read; IM_DEDUP_CAP below is a
   * different variable and only picks a table's starting size.) */

  /* (PIM_BCAST_SCALAR / broadcast-scalar modeling removed 2026-06-22.) */

  /* (PIM_PER_BG_BCAST + PIM_DUPLICATE_BCAST / row-duplicate modeling removed
   * 2026-06-22 with the broadcast blank-dedup machinery.) */

  /* (Accumulator-residency runtime knob removed 2026-06-26 — accumulator reuse
   * is realized in codegen via the loop-carried-SSA psum; the runtime dedup was
   * proven inert. PIM_ACC_RESIDENT is no longer read.) */

  if ((v = getenv("PIM_OPTIPIM_ADDRESSING"))) {
    g_optipim_addressing = atoi(v) != 0;
    if (g_optipim_addressing)
      fprintf(stderr,
              "[pim-runtime] WARNING: PIM_OPTIPIM_ADDRESSING=1. Row/col forced to 0 to "
              "mirror OptiPIM's addressing. Cycles are NOT faithful and must not be "
              "reported as ours.\n");
  }

  /* Lockstep collapse knob (default ON). See g_lockstep_collapse declaration
   * for model. Set PIM_LOCKSTEP_COLLAPSE=0 to disable for ablation. */
  const char *lockstep_env = getenv("PIM_LOCKSTEP_COLLAPSE");
  g_lockstep_enabled = (lockstep_env && lockstep_env[0] == '0') ? 0 : 1;
  stat_lockstep_skips = 0;
  stat_coalesce_overflow = 0;
  fprintf(stderr,
          "[pim-runtime] lockstep_collapse=%d (set PIM_LOCKSTEP_COLLAPSE=0 to disable)\n",
          g_lockstep_enabled);

  /* Dedup-table capacities. Defaults are tuned for ~tens-of-thousands of
   * unique (bank,row,col) tuples per scope, which covers single-channel
   * HBM3-PIM problem sizes. Override via env vars when running larger
   * configurations:
   *   IM_DEDUP_CAP            — capacity for per-program-id load and store
   *                             scopes (default 16K).
   * If a table fills, the dedup primitive falls back to "treat as new"
   * (over-emit, never under-emit), so undersizing degrades performance
   * gracefully without affecting correctness. */
  {
    /* Faithful per-bank residency capacity. Operand reuse is served from a
     * PE-local resident buffer (~one row buffer per bank): a re-read of
     * resident data hits the buffer (no DRAM event), a re-read beyond capacity
     * re-fetches. The default capacity is therefore PHYSICAL — active_banks ×
     * per-bank buffer (row-buffer ≈ num_cols values-of-columns) — rather than
     * the old 256K "blank" cap that modeled an unphysical unbounded operand
     * cache.
     *
     * READ THIS BEFORE TRUSTING THE NUMBER BELOW. Since 2026-09-10 these values
     * only pick the table's STARTING size; addr_dedup grows on demand, so they no
     * longer bound anything. They used to: the lockstep table was capped at
     * active_banks x 136, a figure describing the PE register file and not the
     * number of distinct addresses one program-id touches. On matmul 128x3072x768
     * a program-id marks about 27,800 keys against 8,192 slots, the table filled,
     * check_and_mark started answering "new", and the collapse quietly stopped:
     * bank reads 1,769,472 -> 2,668,032, cycles 14,545,807 -> 21,150,357. Every
     * large-shape number produced before that date is inflated. */
    int active_banks =
        cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
    if (active_banks < 1) active_banks = 1;
    /* STARTING size for the dedup table, in entries per bank. 136 is a leftover
     * from when this was a residency budget bounded by the PE register file; the
     * residency model is gone and the tables grow on demand, so neither this nor
     * IM_RESIDENT_PER_BANK bounds anything or moves a cycle. */
    int resident_per_bank = 136;
    if ((v = getenv("IM_RESIDENT_PER_BANK"))) resident_per_bank = atoi(v);
    if (resident_per_bank < 1) resident_per_bank = 1;
    int physical_cap = active_banks * resident_per_bank;

    int perpid_cap = physical_cap;
    if ((v = getenv("IM_DEDUP_CAP"))) perpid_cap = atoi(v);
    if (perpid_cap < 16) perpid_cap = 16;
    fprintf(stderr,
            "[pim-runtime] dedup table start size: %d/bank x %d banks = %d "
            "(grows on demand; not a bound)\n",
            resident_per_bank, active_banks, physical_cap);

    // Lockstep collapse is FAITHFUL host-emulation-artifact correction: real
    // HBM-PIM issues one all-bank command, but launcher.py replays the kernel
    // per (pid,bank). It is the ONLY collapse left on the store path: the per-pid
    // store dedup went 2026-09-09 and the intra-vector store fold 2026-09-10, both
    // subsumed by this one. See docs/ablation-levers-plan.md.
    g_lockstep_collapse = g_lockstep_enabled ? addr_dedup_create(perpid_cap) : NULL;
  }

  initialized = 1;
  finalized = 0;

  fprintf(stderr, "[pim-runtime] Initialized. Trace: %s\n", trace_file);
  fprintf(stderr,
          "[pim-runtime] HBM config: %d ch, %d pch, %d bg, %d banks/bg, "
          "%d sa, %d rows, %d cols, %d-bit DQ  layout=%s\n",
          cfg_num_channels, cfg_num_pch, cfg_num_bg, cfg_num_banks, cfg_num_sa,
          cfg_num_rows, cfg_num_cols, cfg_dq_bits,
          cfg_layout_scheme == PIM_LAYOUT_INTERLEAVED ? "interleaved"
                                                      : "striped");
}

/* ================================================================
 *  Compiler-emitted residency descriptor
 * ================================================================
 * emitPimLayoutTable (TritonIMToLLVM.cpp) puts these globals in the KERNEL
 * object. Kernel and runtime link into one dylib, so we just read them.
 *
 * Record: [operand_arg, bank_replicated, num_axes, footprint...]. bank_replicated is the compiler-DERIVED role: 1 when
 * every bank sees the same elements (deliver to each PE = OPERAND), 0 when the
 * tensor is bank-partitioned (read bank-locally = STREAMED), -1 when the pass
 * said nothing and the host's role stands.
 * operand_arg == tensor id, because pointer args are registered first and in
 * signature order. bank_replicated is the only word the runtime acts on; the address
 * footprint words fed the retired residency capacity and are ignored,
 * but the record width is kept in sync with emitPimLayoutTable's kRecWords so the
 * table still strides correctly. The record layout itself lives in
 * pim_layout_table.h so the SIMDRAM runtime reads the same definition. */

static int stat_layout_from_compiler = 0;

/* Kernel scalar arguments, indexed the way the compiler numbers tt.func
 * arguments (pointers first, then scalars, no constexprs). Pointer slots go
 * unused. This is the host reporting its own launch, not deciding anything. */
#define PIM_MAX_KERNEL_SCALARS 64
static int32_t g_kernel_scalars[PIM_MAX_KERNEL_SCALARS];
static int g_num_kernel_scalars = 0;

void pim_set_kernel_scalars(const int32_t *vals, int n) {
  if (!vals || n < 0)
    return;
  if (n > PIM_MAX_KERNEL_SCALARS)
    n = PIM_MAX_KERNEL_SCALARS;
  for (int i = 0; i < n; i++)
    g_kernel_scalars[i] = vals[i];
  g_num_kernel_scalars = n;
}

/* Honor the compiler's descriptor for one tensor. No-op if the kernel has none.
 */
/* Refuse a descriptor whose record width is not the one we stride by. The compiler's
 * kRecWords and this runtime's PIM_LAYOUT_REC_WORDS are in separate submodules, so a
 * change to one alone misreads every record past the first, silently and plausibly. */
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

static void apply_compiler_layout(int tensor_id) {
  check_layout_rec_words();
  int n = (int)__pim_layout_count;
  if (n <= 0)
    return; /* weak fallback in force: no table in this kernel */
  const int32_t *table = __pim_layout_table;
  for (int i = 0; i < n; i++) {
    const int32_t *rec = table + (size_t)i * PIM_LAYOUT_REC_WORDS;
    if (rec[PIM_LW_OPERAND_ARG] != tensor_id)
      continue;
    if (stat_layout_from_compiler == 0)
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
    if (rec[PIM_LW_BANK_REPLICATED] >= 0 && _t->role != PIM_ROLE_ACCUMULATOR) {
      int derived =
          rec[PIM_LW_BANK_REPLICATED] ? PIM_ROLE_OPERAND : PIM_ROLE_STREAMED;
      if (derived != _t->role) {
        fprintf(stderr,
                "[pim-runtime] tensor %d: role %s -> %s (compiler-derived from "
                "the layout; host said otherwise)\n",
                tensor_id, _t->role == PIM_ROLE_OPERAND ? "OPERAND" : "STREAMED",
                derived == PIM_ROLE_OPERAND ? "OPERAND" : "STREAMED");
        _t->role = derived;
      }
    }
    stat_layout_from_compiler++;
    return;
  }
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

  int total = 1;
  for (int i = 0; i < ndims && i < 4; i++) {
    t->dims[i] = dims[i];
    total *= dims[i];
  }
  t->num_elements = total;
  t->total_bytes = (size_t)total * elem_size;
  memset(t->base_rows, 0, sizeof(t->base_rows));

  /* Column packing follows the MODELLED data width, not the host array's dtype.
   * elem_size stays in use for address decoding (addr -> element index), which is
   * genuinely a property of how the host stores the array. */
  t->values_per_col = cfg_dq_bits / cfg_data_width_bits;
  if (t->values_per_col < 1)
    t->values_per_col = 1;
  t->values_per_row = cfg_num_cols * t->values_per_col;

  int rows_needed = (total + t->values_per_row - 1) / t->values_per_row;
  int total_flat_banks =
      cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
  int rows_capacity = cfg_num_sa * cfg_num_rows;
  int banks_used = 0;
  int max_rows_per_bank = 0;

  t->total_flat_banks = total_flat_banks;
  t->assigned_ch = 0;
  t->assigned_pch = 0;
  t->assigned_bg = 0;
  t->assigned_bank = 0;
  t->base_row = 0;
  t->layout_scheme = cfg_layout_scheme;
  t->layout_linear_base = 0;

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
  if (cfg_layout_scheme == PIM_LAYOUT_INTERLEAVED) {
    /* Per-tensor power-of-2 guard for values_per_col (depends on
     * elem_size). Other levels are guarded once at pim_init. */
    if (t->values_per_col <= 0 ||
        (t->values_per_col & (t->values_per_col - 1)) != 0) {
      fprintf(stderr,
              "[pim-runtime] ERROR: tensor %d has values_per_col=%d which is "
              "not a power of two; PIM_LAYOUT=interleaved requires it. "
              "Either pick an elem_size that divides cfg_dq_bits cleanly, or "
              "set PIM_LAYOUT=striped.\n",
              num_tensors, t->values_per_col);
      return -1;
    }

    uint64_t elems_per_global_row =
        (uint64_t)t->values_per_row * (uint64_t)total_flat_banks;
    if (elems_per_global_row == 0)
      elems_per_global_row = 1;

    /* The alignment rule is the compiler's now (cfg_place_align). DQ alignment keeps
     * tensors off each other's column slots while letting them share rows on
     * different banks; global-row alignment starts every tensor at bank 0, which
     * serializes small tensors there and is the rule that measured 3.2x worse. */
    uint64_t align = (cfg_place_align == PIM_ALIGN_GLOBAL_ROW)
                         ? elems_per_global_row
                         : (uint64_t)t->values_per_col;
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

    t->striped = 0;
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

  if (rows_needed < total_flat_banks) {
    int global_bank = num_tensors % total_flat_banks;
    decode_flat_bank(global_bank, &t->assigned_ch, &t->assigned_pch,
                     &t->assigned_bg, &t->assigned_bank);
    if (next_free_row[global_bank] + rows_needed > rows_capacity) {
      fprintf(stderr,
              "[pim-runtime] ERROR: tensor %d needs %d rows but bank "
              "(ch=%d pch=%d bg=%d bank=%d) only has %d free of %d\n",
              num_tensors, rows_needed, t->assigned_ch, t->assigned_pch,
              t->assigned_bg, t->assigned_bank,
              rows_capacity - next_free_row[global_bank], rows_capacity);
      return -1;
    }
    t->base_row = next_free_row[global_bank];
    next_free_row[global_bank] += rows_needed;
    t->striped = 0;

    const char *role_str = (role == PIM_ROLE_STREAMED)  ? "STREAMED"
                           : (role == PIM_ROLE_OPERAND) ? "OPERAND"
                                                        : "ACCUMULATOR";
    int first_sa = t->base_row / cfg_num_rows;
    int last_sa = (t->base_row + rows_needed - 1) / cfg_num_rows;
    fprintf(stderr,
            "[pim-runtime] Tensor %d: %s, %d elems (%zu bytes), compact in "
            "ch=%d pch=%d bg=%d bank=%d sa=[%d..%d] rows=[%d..%d]\n",
            num_tensors, role_str, total, t->total_bytes, t->assigned_ch,
            t->assigned_pch, t->assigned_bg, t->assigned_bank, first_sa,
            last_sa, t->base_row, t->base_row + rows_needed - 1);
    return pim_finish_register();
  }

  t->striped = 1;

  for (int global_bank = 0; global_bank < total_flat_banks; global_bank++) {
    int row_count = 0;
    if (rows_needed > global_bank) {
      row_count =
          (rows_needed - global_bank + total_flat_banks - 1) / total_flat_banks;
    }
    t->base_rows[global_bank] = next_free_row[global_bank];
    if (row_count == 0) {
      continue;
    }

    if (next_free_row[global_bank] + row_count > rows_capacity) {
      int ch, pch, bg, bank;
      decode_flat_bank(global_bank, &ch, &pch, &bg, &bank);
      fprintf(stderr,
              "[pim-runtime] ERROR: tensor %d needs %d striped rows but bank "
              "(ch=%d pch=%d bg=%d bank=%d) only has %d free of %d\n",
              num_tensors, row_count, ch, pch, bg, bank,
              rows_capacity - next_free_row[global_bank], rows_capacity);
      return -1;
    }

    next_free_row[global_bank] += row_count;
    banks_used++;
    if (row_count > max_rows_per_bank) {
      max_rows_per_bank = row_count;
    }
  }

  const char *role_str = (role == PIM_ROLE_STREAMED)  ? "STREAMED"
                         : (role == PIM_ROLE_OPERAND) ? "OPERAND"
                                                      : "ACCUMULATOR";
  fprintf(stderr,
          "[pim-runtime] Tensor %d: %s, %d elems (%zu bytes), striped across "
          "%d banks, logical_rows=%d, max_rows/bank=%d\n",
          num_tensors, role_str, total, t->total_bytes, banks_used, rows_needed,
          max_rows_per_bank);

  return pim_finish_register();
}

/* (duplicate_bcast_tensor_per_bg removed 2026-06-22 with the row-duplicate
 * blank-dedup machinery.) */

void pim_set_tensor_broadcast_scalar(int tensor_id, int on) {
  /* REMOVED 2026-06-22: broadcast-scalar / row-duplicate was a blank reuse
   * dedup (it modeled OptiPIM-style row_duplicate replication at trace level).
   * Operand reuse is now realized in the IR (capacity-capped tiling) and the
   * residual is charged by the faithful per-bank LRU + lockstep. This setter is
   * a permanent no-op kept only for ABI compatibility with the harness bridge
   * (pass_ablation.py wraps the call in try/except). */
  (void)tensor_id;
  (void)on;
}

void pim_set_phase(pim_phase_t phase) {
  const char *names[] = {"IDLE", "COMPUTE", "HOST"};
  if (phase <= PIM_PHASE_HOST) {
    fprintf(stderr, "[pim-runtime] Phase -> %s\n", names[phase]);
  }
  /* Phase boundaries are the natural reset points for every dedup
   * scope: each phase is a distinct logical workload and any cached row
   * activations from one phase must not carry into the next. */
  if (phase != cur_phase) {
    /* Leaving COMPUTE: charge the modelled accumulator spill now, so it lands in
     * the phase whose work it belongs to. */
    if (cur_phase == PIM_PHASE_COMPUTE)
      emit_acc_spill();
  }
  cur_phase = phase;
}

/* Emit the modelled accumulator spill. Called once as COMPUTE ends, so it lands in
 * the same phase as the work it belongs to. Addresses walk the accumulator's own
 * elements per bank, so Ramulator sees real rows rather than a synthetic constant. */
static void emit_acc_spill(void) {
  if (g_acc_spill_tensor < 0 || g_acc_spill_tensor >= num_tensors)
    return;
  tensor_info_t *t = &tensors[g_acc_spill_tensor];
  int banks = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
  if (banks < 1)
    banks = 1;
  if (banks > MAX_BANKS)
    banks = MAX_BANKS;
  long long pairs = (long long)g_acc_spill_overflow * g_acc_spill_ksteps;
  for (int b = 0; b < banks; b++) {
    int ch, pch, bg, bank;
    decompose_global_bank(b, &ch, &pch, &bg, &bank);
    for (long long i = 0; i < pairs; i++) {
      int elem = (int)((i * banks + b) % (t->num_elements > 0 ? t->num_elements : 1));
      int e_ch, e_pch, e_bg, e_bank, sa, row, col;
      map_element(t, elem, &e_ch, &e_pch, &e_bg, &e_bank, &sa, &row, &col);
      emit_trace("BW", ch, pch, bg, bank, sa, row, col);
      emit_trace("BR", ch, pch, bg, bank, sa, row, col);
      stat_acc_spill_emitted += 2;
    }
  }
  fprintf(stderr, "[pim-runtime]   Accumulator spill (MODELLED): %" PRIu64 " records\n",
          stat_acc_spill_emitted);
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
          stat_bank_reads);
  fprintf(stderr, "[pim-runtime]   Bank-writes (BW): %" PRIu64 "\n",
          stat_bank_writes);
  fprintf(stderr, "[pim-runtime]   Reads       (R) : %" PRIu64 "\n",
          stat_reads);
  fprintf(stderr, "[pim-runtime]   Writes      (W) : %" PRIu64 "\n",
          stat_writes);
  fprintf(stderr, "[pim-runtime]   Ignored        : %" PRIu64 "\n",
          stat_ignored);
  fprintf(stderr, "[pim-runtime]   Total PIM ops  : %" PRIu64 "\n",
          stat_bank_reads + stat_bank_writes + stat_reads + stat_writes);
  fprintf(stderr,
          "[pim-runtime]   Lockstep collapse skips         : %" PRIu64 " (lockstep_collapse=%d)\n",
          stat_lockstep_skips, g_lockstep_enabled);
  {
    uint64_t sat = addr_dedup_saturations(g_lockstep_collapse);
    if (sat)
      fprintf(stderr,
              "[pim-runtime]   WARN dedup table saturated       : %" PRIu64
              " (collapse stopped; trace over-emits)\n", sat);
  }
  fprintf(stderr,
          "[pim-runtime]   Load coalesce (intra-vector)    : %" PRIu64 "\n",
          stat_load_coalesced);
  if (stat_coalesce_overflow)
    fprintf(stderr,
            "[pim-runtime]   WARN coalesce buffer overflow    : %" PRIu64
            " (raise PIM_COALESCE_MAX)\n",
            stat_coalesce_overflow);

  /* Per-tensor breakdown — Stage-0 diagnostics. Useful for figuring out
   * which tensor's accesses dominate the trace and where dedup is biting. */
  fprintf(stderr,
          "\n[pim-runtime] === Per-tensor breakdown ===\n"
          "[pim-runtime]   tid  role         range_calls   BR        BW        "
          "R         W         dedup_skips\n");
  for (int i = 0; i < num_tensors; i++) {
    tensor_info_t *t = &tensors[i];
    const char *role_str = (t->role == PIM_ROLE_STREAMED)  ? "STREAMED"
                           : (t->role == PIM_ROLE_OPERAND) ? "OPERAND "
                                                           : "ACCUMUL.";
    fprintf(stderr,
            "[pim-runtime]   %3d  %-9s    %10" PRIu64 "  %8" PRIu64
            "  %8" PRIu64 "  %8" PRIu64 "  %8" PRIu64 "  %12" PRIu64 "\n",
            i, role_str, t->range_calls, t->emitted_br, t->emitted_bw,
            t->emitted_r, t->emitted_w, t->dedup_skips);
  }

  destroy_dedup_state();
  initialized = 0;
}

/* ================================================================
 *  LLVM pass hooks
 * ================================================================ */

void __mem_trace_init(void) {
  /* No-op: PIM runtime expects user to call pim_init() explicitly
   * before registering tensors. If pim_init() is never called,
   * __mem_trace_load/store will skip (trace_fp == NULL). */
}

void __mem_trace_fini(void) { pim_finalize(); }

/*
 * Translate a single memory access into PIM trace operations.
 *
 * COMPUTE phase (PIM PE active):
 *   STREAMED load   -> BR (bank-read: data streams through PE)
 *   OPERAND load    -> W  (write: data to PE register via bus)
 *   ACCUMULATOR load  -> R  (read: partial sum from PE)
 *   ACCUMULATOR store -> BW (bank-write: result to bank)
 *
 * HOST phase: any load -> R, any store -> W (via bus)
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

  /* Operand register-residency reuse skip retired 2026-09-04: reuse is now the
   * kernel tile, and a broadcast operand collapses to one WB below. ACCUMULATOR
   * (psum) residency is realized in codegen (loop-carried SSA), never a runtime
   * skip. Stores are handled by the per-program-id write-once model below. */

  /* Lockstep dedup: collapse bank-replicated events at the same
   * (tensor_id, sa, row, col) within a program-id. Models 1 SIMD
   * dispatch per logical instruction in the bank-parallel hardware.
   *
   * Bypass for OPERAND loads (bcast or vector): OptiPIM's PimCodeGen
   * emits per-PE register-load commands (priority-read + N writes) for
   * EVERY operand load, not just broadcasts. The N writes model the
   * bus-mediated transfer of operand values into each PE's local
   * register file. Collapsing them to 1 under-models that bus
   * pipeline cost. Bypassing the dedup for OPERAND loads matches
   * OptiPIM's abstraction level for cycle accounting.
   *
   * STREAMED tensors (each PE reads its own bank-local slice via BR)
   * still collapse — those are genuine SIMD-bank-parallel bank-reads,
   * not bus broadcasts. */
  /* EVERY operand load bypasses the collapse and pays its per-bank write, broadcasts
   * included. Real HBM-PIM does have an all-bank GRF broadcast (Lee et al. ISCA'21
   * III-A/III-B), so this is a SYMMETRY choice, not a hardware limit: OptiPIM prices
   * input loading per PU inside its MILP objective, and taking the broadcast without
   * re-solving their optimizer would charge us for a dispatch the baseline never
   * gets. Both stacks now charge one write per receiving bank. */
  int is_operand_load = (!is_write && t->role == PIM_ROLE_OPERAND);
  if (g_lockstep_enabled && g_lockstep_collapse &&
      cur_phase == PIM_PHASE_COMPUTE && !is_operand_load) {
    /* Scope the collapse to ONE pseudochannel. An all-bank PIM command reaches the
     * banks of a single (ch,pch); replicas in a different pch or channel ride a
     * separate command bus and need their own event. Omitting them collapsed
     * across independent buses and under-charged (2026-09-05 audit). Bank/bg stay
     * OUT of the key: collapsing those IS the all-bank SIMD model. */
    uint64_t tensor_key = lockstep_ns(t, loc.ch, loc.pch, is_write);
    /* The dispatch id rides in k2 above the linear row, so two dispatches touching
     * the same physical tuple stay distinct while the 32 bank replays of ONE
     * dispatch still collapse. Replaces resetting the table per program instance. */
    uint64_t key_row = (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    uint64_t key_col = (uint64_t)loc.col;
    key_row |= cur_dispatch << g_dispatch_shift;
    if (!addr_dedup_check_and_mark(g_lockstep_collapse, tensor_key, key_row,
                                   key_col)) {
      stat_lockstep_skips++;
      t->dedup_skips++;
      return;
    }
  }

  /* Operand-delivery amortization deleted 2026-09-10. It granted an operand word
   * reuse ACROSS program instances, which the PE register file cannot do because it
   * reloads per SIMD dispatch. Removing it costs matvec 1.407x. A persistent kernel
   * is the legitimate way to get that reuse back. Measurements in the commit. */

  emit_access_by_role_phase(t, &loc, is_write);

  /* (per-BG broadcast fanout removed 2026-06-22 with the bcast_scalar/
   * row-duplicate machinery — faithful configs never set bcast_scalar.) */
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
    stat_ignored++;
    return;
  }

  int tidx = find_tensor(base_addr);
  if (tidx < 0) {
    stat_ignored++;
    return;
  }
  tensor_info_t *t = &tensors[tidx];
  t->range_calls++;

  /* Track program-id boundaries so the dedup table is reset per tile. */
  if (cur_phase == PIM_PHASE_COMPUTE) {
    advance_program_epoch_if_needed();
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
    static int warned = 0;
    if (!warned) {
      warned = 1;
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
   * LOADS ONLY, deliberately. The store twin was deleted 2026-09-10 as structurally
   * dead: within one call, two lanes at the same (bank,sa,row,col) always share the
   * lockstep key as well, since that key drops only bank and bank-group, so the
   * collapse downstream catches every fold the store side would have made. Loads
   * differ because an OPERAND load bypasses lockstep by design, which is what leaves
   * this copy load-bearing.
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
          stat_ignored++;
          continue;
        }
        t = &tensors[ntidx];
      }
      int elem_idx =
          (int)((elem_addr - (uint64_t)t->base_addr) / t->elem_size);
      if (elem_idx < 0 || elem_idx >= t->num_elements) {
        stat_ignored++;
        continue;
      }
      /* Keyed by the element's physical tuple, not by the receiving bank where an
       * OPERAND load's bus write actually lands. Re-keying was tried 2026-09-10 and
       * reverted as inert, but the premise has since changed: the conv weight load
       * used to arrive one element per call and skip this path entirely, and it now
       * vectorizes. Re-keying is untested under that shape. */
      int ch, pch, bg, bank, sa, row, col;
      map_element(t, elem_idx, &ch, &pch, &bg, &bank, &sa, &row, &col);
      int gb = compute_global_bank(ch, pch, bg, bank);
      uint64_t key = ((uint64_t)gb & 0xFFFFULL) |
                     (((uint64_t)sa & 0xFFULL) << 16) |
                     (((uint64_t)row & 0xFFFFFFULL) << 24) |
                     (((uint64_t)col & 0xFFFFULL) << 48);
      int dup = 0;
      for (int j = 0; j < n_seen; j++)
        if (seen[j] == key) { dup = 1; break; }
      if (dup) {
        stat_load_coalesced++;
        t->dedup_skips++;
        continue;
      }
      if (n_seen < PIM_COALESCE_MAX)
        seen[n_seen++] = key;
      else
        stat_coalesce_overflow++;
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
        stat_ignored++;
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

