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
/* One column command moves internal_prefetch * dq = 2 * 128 = 256 bits on this part,
 * which is the HBM-PIM datapath, so the column index counts 256-bit words. Columns per
 * row halve to match, keeping the row at 8192 bits and the part's density unchanged.
 * These two MUST move together, and together with the simulator org (dq/column in the
 * rendered yaml), or the tracer addresses a row the machine does not have. */
static int cfg_num_cols = 32;
static int cfg_dq_bits = 256;
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
/* Lane slabs take rows from the TOP of the per-bank row space; interleaved placement
 * grows from the bottom, and the two must never meet. */
static int g_lane_row_top = -1;
static uint64_t stat_lane_placed = 0;

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
  /* Compiler's partition bit: 1 replicated, 0 bank-partitioned, -1 not stated. Only 0
   * changes anything: that tensor's accesses go to the executing lane's own bank at a
   * per-lane slab address instead of the element's interleaved home, see
   * place_in_lane_slab. The lockstep key drops bank, so counts do not move; addresses do. */
  int bank_replicated;
  int lane_row_base;        /* first row of this tensor's per-lane slab, or -1 */
  int lane_row_count;
  int lane_ord[MAX_BANKS];  /* next free value slot per lane */
  slot_map_t *lane_slot;    /* (lane, elem) -> slot, partitioned tensors only */
  /* __pim_opt_lane_rep and __pim_opt_lane_ext, read by the counters and the native bug bits
   * to tell dropped slices apart, never to decide a count. -1 not stated. */
  int opt_lane_rep;
  int opt_lane_n;
  int opt_lane_dim[PIM_MAX_OPT_LANE_AXES];
  int opt_lane_ext[PIM_MAX_OPT_LANE_AXES];
  /* __pim_opt_residency: 1 delivered, 2 resident, 3 drained, -1 not stated. */
  int opt_res;

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

/* Namespace bits for the lockstep-collapse key: tid 4b, ch 5b, pch 2b, write 1b.
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

static inline uint64_t lk_mix(uint64_t x) {
  x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
  x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ULL;
  return x ^ (x >> 33);
}

/* The lockstep key hashed to 64 bits, as ramulator2_dcc keys its own less the access
 * ordinal. A packed key has room for 17 bits of the dispatch id. Passed as k1 = h,
 * k2 = h >> 16, k3 = h >> 48, which pack_keys reassembles into h. */
static uint64_t lockstep_key(uint64_t ns, uint64_t disp, uint64_t row, uint64_t col) {
  uint64_t h = lk_mix(ns ^ 0x9E3779B97F4A7C15ULL);
  h = lk_mix(h ^ disp);
  return lk_mix(h ^ ((row << 16) | col));
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
/* Operand deliveries NOT charged because the compiler said the value is replicated over
 * only some of the banks. 0 whenever every split is single-axis, which is every mapping
 * our own picker emits. */
static uint64_t stat_operand_repl_skips = 0;
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
 * trace under-emitted. The tile is policed at pack time. */
#define PIM_TILE_BITS 10
static int g_warned_dispatch_overflow = 0;

/* OPTIPIM'S codegen_new, NATIVELY (PIM_OPT_NATIVE=1, default off).
 *
 * Their HBM-PIM codegen (OptiPIM simulator fimdram.cpp:219-322 with single_bank_opt)
 * charges every temporal step of a mapping for the leader bank alone:
 *   W   E per bank to the first L banks, only when the step's input is new to the leader
 *   BR  E at the leader, a fresh row per step, columns 0..E-1
 *   R   E at the leader's base address
 * E = ceil(C/n): C is the contraction's iteration points per lane per step, charged by
 * position rather than by any tensor's footprint, n values per DQ word. L = min(active
 * banks, banks per pseudochannel). Here a step is one of OUR dispatches and novelty is the
 * leader lane's own input loads, so the step count and the novel steps come from the
 * kernel that ran. C and the roles are the compiler's (__pim_opt_*). Nothing else is
 * emitted while this is on. */
enum { OPT_NONE = 0, OPT_INPUT = 1, OPT_WEIGHT = 2, OPT_OUTPUT = 3 };
/* The compiler's residency stamp, __pim_opt_residency. */
enum { RES_DELIVERED = 1, RES_RESIDENT = 2, RES_DRAINED = 3 };
/* fp: the leader lane's distinct elements this step, indexed by role (input and weight
 * loaded, output stored), so the harness can hold the per-axis tile to their mapping. */
typedef struct { uint64_t epoch; int novel; int fp[4]; } opt_step_t;
static int g_opt_native = 0;
static int g_opt_role[MAX_TENSORS];
static uint8_t *g_opt_seen[MAX_TENSORS];   /* input elements the leader already holds */
static uint32_t *g_opt_stamp[MAX_TENSORS]; /* step that last touched each element */
static int stat_opt_fp_min[4], stat_opt_fp_max[4];
static uint8_t g_opt_active[MAX_BANKS];
static opt_step_t *g_opt_steps = NULL;
static size_t g_opt_n = 0, g_opt_cap = 0;
static uint64_t stat_opt_novel = 0, stat_opt_cmds = 0;
static int stat_opt_lanes = 0;

/* THEIR CONVENTIONS ON OUR STREAM (PIM_OPT_NATIVE = OPT_CONVENTIONS | bits). The
 * as-emitted pipeline runs unchanged, each dispatch's surviving commands are held, and at
 * the end of the dispatch every set bit applies one of codegen_new's conventions to them
 * before they are written. No bit set writes the as-emitted stream, every bit set writes
 * their stream. Grouped as the harness reports them:
 *   mapping       ROLE     their positional roles: the input written to the PEs (W), the
 *                          weight read in the bank (BR), whatever our layout made of them
 *                 ORDER    per step the input block bank by bank, then weight, then output
 *   physical      TILE_BR  the weight's bank reads at the MAC count, E per step
 *                 READOUT  the output's partials read out (R) from every bank that stored
 *                          them, in place of our write-back
 *   pessimistic   FAN      the input written to every bank that stores
 *                 TILE_WR  the input's writes and the read-out at E per step, the iteration
 *                          tile rather than the footprint
 *   under-charge  NOVEL    the input sent only on a step that brings the leader an element
 *                          it never held
 *                 R0       the output read out from the leader bank only
 *                 LEADER   only lane 0's commands, its input reaching the first 16 of the
 *                          storing banks
 *                 S1DROP   with one bank, the last iteration point dropped
 *   on its own    POS      their synthetic addresses: banks by place among the storing
 *                          banks, W and R at the bank's base, BR on row t mod rows
 * E = ceil(C/n), C the iteration points the compiler states, cycling over our addresses. */
enum { OPT_REPLAY = 1, OPT_CONVENTIONS = 2, CV_ROLE = 4, CV_READOUT = 8, CV_TILE_BR = 16,
       CV_FAN = 32, CV_ORDER = 64, CV_NOVEL = 128, CV_R0 = 256, CV_LEADER = 512,
       CV_POS = 1024, CV_S1DROP = 2048, CV_TILE_WR = 4096, CV_ALL = 8190 };
typedef struct { char op[4]; int a[7]; int origin, orole; uint32_t seq; } cv_ev_t;
static int g_cv_tid = -1;
static cv_ev_t *g_cv = NULL, *g_cv_out = NULL;
static size_t g_cv_n = 0, g_cv_cap = 0, g_cv_on = 0, g_cv_ocap = 0;
static uint8_t g_cv_store[MAX_BANKS];   /* lanes that stored in this dispatch */
static uint64_t stat_cv_staged = 0, stat_cv_lines = 0, stat_cv_novel_dropped = 0;
static uint64_t stat_cv_residue = 0, stat_cv_leader_dropped = 0, stat_cv_tile_empty = 0;
static uint64_t stat_cv_pos_unmapped = 0, stat_cv_unfanned = 0, stat_cv_lane0_idle = 0;
static int stat_cv_lanes_min = -1, stat_cv_lanes_max = -1;

/* ================================================================
 *  Helpers
 * ================================================================ */

static int g_cv_capture = 0;   /* set while a COMPUTE access is being emitted */
static void cv_stage(const char *op, int ch, int pch, int bg, int bank, int sa, int row,
                      int col);

static void emit_trace(const char *op, int ch, int pch, int bg, int bank,
                       int sa, int row, int col) {
  if (g_cv_capture) {
    cv_stage(op, ch, pch, bg, bank, sa, row, col);
    return;
  }
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
    if (!g_warned_dispatch_overflow && __pim_persistent &&
        tile >= (1ULL << PIM_TILE_BITS)) {
      g_warned_dispatch_overflow = 1;
      fprintf(stderr,
              "[pim-runtime] WARN: tile %llu does not fit the dispatch id's %d tile "
              "bits (epoch %llu). Ids alias and distinct dispatches collapse.\n",
              (unsigned long long)tile, PIM_TILE_BITS, (unsigned long long)epoch);
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

/* Banks are threads here: lane b is bank b's PE and may only touch bank b. The
 * interleaved map changes bank every 512 elements while a lane's slab is 1024, so
 * lane 0's data sat in banks 0 and 1. Put the element in the executing lane's bank at
 * the next slot of that lane's slab (OptiPIM's alloc_col++ made explicit); the slot is
 * remembered so a re-read lands on the same address. A replicated tensor gets the same
 * treatment: its copy in bank b is a full slab, not the source word's (sa,row,col)
 * with the bank bits dropped, which aliased 4-8 words onto one address. */
static int lane_slab_place(tensor_info_t *t, pim_phys_loc_t *loc) {
  static int warned = 0;
  int all_banks = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
  int lane = __pim_get_bank_id();
  if (lane < 0 || lane >= all_banks || lane >= MAX_BANKS) {
    if (!warned++)
      fprintf(stderr, "[pim-runtime] WARN replay lane %d outside the %d banks; kept "
                      "the interleaved home.\n", lane, all_banks);
    return 0;
  }
  /* Every lane loads a replicated tensor in the same order, so one numbering serves
   * all 32 copies: key on lane 0 and the map stays the size of the tensor, not 32x.
   * A resident tensor is numbered per lane instead. Lanes that split it on another axis
   * hold different slices, and only per-lane slots put each lane's k-th word in one place. */
  int key_lane = (t->bank_replicated >= 1 && t->opt_res != RES_RESIDENT) ? 0 : lane;
  int inserted = 0;
  int32_t slot = slot_map_get_or_put(t->lane_slot, key_lane, loc->elem_idx,
                                     t->lane_ord[key_lane], &inserted);
  if (slot < 0) {
    if (!warned++)
      fprintf(stderr, "[pim-runtime] WARN slab slot map allocation failed; kept the "
                      "interleaved home.\n");
    return 0;
  }
  if (inserted)
    t->lane_ord[key_lane]++;
  int vpc = t->values_per_col > 0 ? t->values_per_col : 16;
  int vpr = t->values_per_row > 0 ? t->values_per_row : 512;
  int r = slot / vpr;
  if (r >= t->lane_row_count) {
    if (!warned++)
      fprintf(stderr, "[pim-runtime] WARN tensor lane %d overflowed its %d-row slab; "
                      "rows now alias.\n", lane, t->lane_row_count);
    r = t->lane_row_count - 1;
  }
  int linear_row = t->lane_row_base + r;
  decompose_global_bank(lane, &loc->ch, &loc->pch, &loc->bg, &loc->bank);
  loc->sa = linear_row / cfg_num_rows;
  loc->row = linear_row % cfg_num_rows;
  loc->col = (slot % vpr) / vpc;
  return 1;
}

static void place_in_lane_slab(tensor_info_t *t, pim_phys_loc_t *loc) {
  if (lane_slab_place(t, loc))
    stat_lane_placed++;
}

/* Their alloc new as im-optipim-mac states it (__pim_opt_mac_geom): MAC j of step t sits at one
 * slot of the lane's own bank, the same for every tensor. 0 for a slot the bank lacks, so E
 * above the columns of a row is refused, never wrapped, and 2 where sa0 moved the row. */
static int arranged_place(int lane, int64_t t, int64_t j, int sa0, pim_phys_loc_t *loc) {
  const int32_t *g = __pim_opt_mac_geom;
  int64_t row = (int64_t)g[1] * t + (int64_t)g[2] * j;
  int64_t col = (int64_t)g[3] * t + (int64_t)g[4] * j;
  /* SA0, theirs: the row wraps at count[row], a subarray, not rows per bank (fimdram.cpp:294). */
  int moved = sa0 && row >= cfg_num_rows;
  if (moved) row %= cfg_num_rows;
  int banks = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
  if (t < 0 || lane < 0 || lane >= banks || row < 0 ||
      row >= (int64_t)cfg_num_sa * cfg_num_rows || col < 0 || col >= cfg_num_cols)
    return 0;
  decompose_global_bank(lane, &loc->ch, &loc->pch, &loc->bg, &loc->bank);
  loc->elem_idx = -1;
  loc->sa = (int)(row / cfg_num_rows);
  loc->row = (int)(row % cfg_num_rows);
  loc->col = (int)col;
  return 1 + moved;
}

/* Resolve one logical element access to its physical HBM tuple.
 *
 * Returns 1 on success and fills `loc`. Returns 0 when the current phase
 * should not emit anything or the address falls outside the tensor. In both
 * failure cases, stat_ignored is updated here so callers can simply return. */
static int resolve_access_location(tensor_info_t *t, uint64_t addr,
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
  if (cur_phase == PIM_PHASE_COMPUTE && t->bank_replicated >= 0 && t->lane_slot &&
      t->lane_row_base >= 0)
    place_in_lane_slab(t, loc);
  return 1;
}

/* RUN COUNTERS (PIM_OPT_COUNTERS=1), one cell per (dispatch, lane, tensor, direction), never
 * read by emission. Dispatches are numbered per run since __pim_program_epoch spans replays.
 * Each ISSUED access is FOLDED, SKIPPED (receiver stride), DROPPED (store to an input) or emitted. */
enum { CC_DISP, CC_LANE, CC_TID, CC_WRITE, CC_CALLS, CC_ELEMS, CC_WORDS, CC_ISSUED,
       CC_FOLDED, CC_SKIPPED, CC_DROPPED, CC_W, CC_BR, CC_R, CC_BW, CC_FIELDS };
enum { CD_EPOCH, CD_TILE, CD_PID_X, CD_PID_Y, CD_PID_Z, CD_FIELDS };
/* PIM_OPT_COUNTERS_DUMP=tid: each (lane, element, direction) of that tensor at its placed
 * address, on first touch. */
enum { CA_DISP, CA_LANE, CA_ELEM, CA_WRITE, CA_CH, CA_PCH, CA_BG, CA_BANK, CA_SA, CA_ROW,
       CA_COL, CA_FIELDS };
enum { CTR_CELLS, CTR_DISPATCHES, CTR_DUMP, CTR_TABLES };
_Static_assert(MAX_TENSORS <= 16 && MAX_BANKS <= 1024 && PIM_TILE_BITS <= 24,
               "the counter keys pack tid in 4 bits, lane in 10 and tile in 24");
#define CTR_WORDS_MAX 256
#define CTR_DUMP_MAX (1u << 21)

typedef struct { uint64_t *k; uint32_t *v; size_t cap, n; } ctr_map_t;
typedef struct { uint32_t *row; size_t n, cap; } ctr_rows_t;
static const int ctr_fields[CTR_TABLES] = {CC_FIELDS, CD_FIELDS, CA_FIELDS};
static ctr_rows_t g_ctr[CTR_TABLES];
static ctr_map_t g_ctr_map[CTR_TABLES];
static int g_ctr_on = 0, g_ctr_dump_tid = -1;
static uint64_t g_ctr_epoch0 = 0;
static int64_t g_ctr_disp = -1;
static uint64_t g_ctr_disp_key = UINT64_MAX, g_ctr_cell_key = UINT64_MAX;
static uint32_t g_ctr_cell_idx = 0;
/* W, BR, R, BW emitted outside any cell: the HOST phase and the modelled spill. */
static uint64_t g_ctr_outside[4];
static uint64_t g_ctr_word_overflow = 0, g_ctr_dump_overflow = 0;

static uint64_t ctr_mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

/* The value under key, or `fresh` stored there now. Keys are stored plus one, 0 is empty. */
static uint32_t ctr_map_get(ctr_map_t *m, uint64_t key, uint32_t fresh, int *inserted) {
  if (2 * (m->n + 1) > m->cap) {
    size_t cap = m->cap ? 2 * m->cap : 1024;
    uint64_t *k = (uint64_t *)calloc(cap, sizeof *k);
    uint32_t *v = (uint32_t *)malloc(cap * sizeof *v);
    if (!k || !v) { fprintf(stderr, "[pim-runtime] ERROR: counter map alloc\n"); exit(1); }
    for (size_t i = 0; i < m->cap; i++) {
      if (!m->k[i]) continue;
      size_t j = ctr_mix(m->k[i]) & (cap - 1);
      while (k[j]) j = (j + 1) & (cap - 1);
      k[j] = m->k[i];
      v[j] = m->v[i];
    }
    free(m->k);
    free(m->v);
    m->k = k; m->v = v; m->cap = cap;
  }
  uint64_t kk = key + 1;
  size_t j = ctr_mix(kk) & (m->cap - 1);
  for (; m->k[j]; j = (j + 1) & (m->cap - 1))
    if (m->k[j] == kk) { *inserted = 0; return m->v[j]; }
  m->k[j] = kk;
  m->v[j] = fresh;
  m->n++;
  *inserted = 1;
  return fresh;
}

static uint32_t *ctr_row_new(int table) {
  ctr_rows_t *t = &g_ctr[table];
  size_t f = (size_t)ctr_fields[table];
  if (t->n == t->cap) {
    t->cap = t->cap ? 2 * t->cap : 4096;
    t->row = (uint32_t *)realloc(t->row, t->cap * f * sizeof *t->row);
    if (!t->row) { fprintf(stderr, "[pim-runtime] ERROR: counter table alloc\n"); exit(1); }
  }
  uint32_t *r = t->row + t->n++ * f;
  memset(r, 0, f * sizeof *r);
  return r;
}

static void ctr_reset(void) {
  for (int i = 0; i < CTR_TABLES; i++) {
    free(g_ctr[i].row);
    free(g_ctr_map[i].k);
    free(g_ctr_map[i].v);
    memset(&g_ctr[i], 0, sizeof g_ctr[i]);
    memset(&g_ctr_map[i], 0, sizeof g_ctr_map[i]);
  }
  memset(g_ctr_outside, 0, sizeof g_ctr_outside);
  g_ctr_word_overflow = g_ctr_dump_overflow = 0;
  g_ctr_disp = -1;
  g_ctr_disp_key = g_ctr_cell_key = UINT64_MAX;
  g_ctr_epoch0 = __pim_program_epoch;
  const char *v = getenv("PIM_OPT_COUNTERS");
  g_ctr_on = v && atoi(v) != 0;
  v = getenv("PIM_OPT_COUNTERS_DUMP");
  g_ctr_dump_tid = g_ctr_on && v && *v ? atoi(v) : -1;
}

/* The dispatch this access belongs to, numbered within the run. */
static void ctr_dispatch(void) {
  uint64_t tile = __pim_tile_index(), ep = __pim_program_epoch - g_ctr_epoch0;
  uint64_t key = (ep << 24) | (tile & 0xFFFFFFULL);
  if (key == g_ctr_disp_key)
    return;
  g_ctr_disp_key = key;
  int ins;
  uint32_t ord = ctr_map_get(&g_ctr_map[CTR_DISPATCHES], key,
                             (uint32_t)g_ctr[CTR_DISPATCHES].n, &ins);
  if (ins) {
    uint32_t *r = ctr_row_new(CTR_DISPATCHES);
    r[CD_EPOCH] = (uint32_t)ep;
    r[CD_TILE] = (uint32_t)tile;
    r[CD_PID_X] = (uint32_t)__pim_get_program_id();
    r[CD_PID_Y] = (uint32_t)__pim_get_program_id_y();
    r[CD_PID_Z] = (uint32_t)__pim_get_program_id_z();
  }
  g_ctr_disp = ord;
}

static int64_t ctr_cell(int tid, int is_write) {
  if (!g_ctr_on || cur_phase != PIM_PHASE_COMPUTE || g_ctr_disp < 0)
    return -1;
  int lane = __pim_get_bank_id();
  uint64_t key = ((uint64_t)g_ctr_disp << 16) | ((uint64_t)(lane & 0x3FF) << 5) |
                 ((uint64_t)(tid & 0xF) << 1) | (uint64_t)(is_write != 0);
  if (key == g_ctr_cell_key)
    return g_ctr_cell_idx;
  int ins;
  uint32_t idx = ctr_map_get(&g_ctr_map[CTR_CELLS], key, (uint32_t)g_ctr[CTR_CELLS].n, &ins);
  if (ins) {
    uint32_t *r = ctr_row_new(CTR_CELLS);
    r[CC_DISP] = (uint32_t)g_ctr_disp;
    r[CC_LANE] = (uint32_t)lane;
    r[CC_TID] = (uint32_t)tid;
    r[CC_WRITE] = (uint32_t)(is_write != 0);
  }
  g_ctr_cell_key = key;
  g_ctr_cell_idx = idx;
  return idx;
}

static void ctr_add(const tensor_info_t *t, int is_write, int field) {
  int64_t c = ctr_cell((int)(t - tensors), is_write);
  if (c >= 0)
    g_ctr[CTR_CELLS].row[(size_t)c * CC_FIELDS + field]++;
}

static void ctr_dump(const pim_phys_loc_t *pl, int is_write) {
  int lane = __pim_get_bank_id();
  if (g_ctr[CTR_DUMP].n >= CTR_DUMP_MAX) {
    g_ctr_dump_overflow++;
    return;
  }
  uint64_t key = ((uint64_t)(lane & 0x3FF) << 33) | ((uint64_t)(uint32_t)pl->elem_idx << 1) |
                 (uint64_t)(is_write != 0);
  int ins;
  ctr_map_get(&g_ctr_map[CTR_DUMP], key, 0, &ins);
  if (!ins)
    return;
  uint32_t *r = ctr_row_new(CTR_DUMP);
  r[CA_DISP] = (uint32_t)g_ctr_disp;
  r[CA_LANE] = (uint32_t)lane;
  r[CA_ELEM] = (uint32_t)pl->elem_idx;
  r[CA_WRITE] = (uint32_t)(is_write != 0);
  r[CA_CH] = (uint32_t)pl->ch;
  r[CA_PCH] = (uint32_t)pl->pch;
  r[CA_BG] = (uint32_t)pl->bg;
  r[CA_BANK] = (uint32_t)pl->bank;
  r[CA_SA] = (uint32_t)pl->sa;
  r[CA_ROW] = (uint32_t)pl->row;
  r[CA_COL] = (uint32_t)pl->col;
}

/* Runs after the call was emitted. Under PIM_OPT_NATIVE=1 nothing was placed, so this places
 * the slots itself, which leaves .o1 alone but moves the slab-slot stderr line. */
static void ctr_call(uint64_t base_addr, uint64_t size, int is_write) {
  int tidx = find_tensor(base_addr);
  if (tidx < 0)
    return;
  tensor_info_t *t = &tensors[tidx];
  ctr_add(t, is_write, CC_CALLS);
  int es = t->elem_size > 0 ? t->elem_size : 1;
  uint64_t n = size / (uint64_t)es;
  if (n == 0)
    n = 1;
  uint64_t seen[CTR_WORDS_MAX];
  int n_seen = 0;
  for (uint64_t i = 0; i < n; i++) {
    uint64_t a = base_addr + i * (uint64_t)es;
    if (a >= (uint64_t)t->base_addr + t->total_bytes) {
      int nt = find_tensor(a);
      if (nt < 0)
        continue;
      t = &tensors[nt];
    }
    int elem = (int)((a - (uint64_t)t->base_addr) / t->elem_size);
    if (elem < 0 || elem >= t->num_elements)
      continue;
    pim_phys_loc_t pl;
    pl.elem_idx = elem;
    map_element(t, elem, &pl.ch, &pl.pch, &pl.bg, &pl.bank, &pl.sa, &pl.row, &pl.col);
    if (t->bank_replicated >= 0 && t->lane_slot && t->lane_row_base >= 0)
      lane_slab_place(t, &pl);
    ctr_add(t, is_write, CC_ELEMS);
    int gb = compute_global_bank(pl.ch, pl.pch, pl.bg, pl.bank);
    uint64_t key = ((uint64_t)gb & 0xFFFFULL) | (((uint64_t)pl.sa & 0xFFULL) << 16) |
                   (((uint64_t)pl.row & 0xFFFFFFULL) << 24) |
                   (((uint64_t)pl.col & 0xFFFFULL) << 48);
    int dup = 0;
    for (int j = 0; j < n_seen && !dup; j++)
      dup = seen[j] == key;
    if (!dup) {
      if (n_seen < CTR_WORDS_MAX)
        seen[n_seen++] = key;
      else
        g_ctr_word_overflow++;
      ctr_add(t, is_write, CC_WORDS);
    }
    if ((int)(t - tensors) == g_ctr_dump_tid)
      ctr_dump(&pl, is_write);
  }
}

int pim_ctr_enabled(void) { return g_ctr_on; }
int pim_ctr_dump_tensor(void) { return g_ctr_dump_tid; }
int pim_ctr_fields(int table) { return table >= 0 && table < CTR_TABLES ? ctr_fields[table] : 0; }
uint64_t pim_ctr_rows(int table) {
  return table >= 0 && table < CTR_TABLES ? (uint64_t)g_ctr[table].n : 0;
}
/* rows(table) x fields(table) uint32 into out. */
void pim_ctr_copy(int table, uint32_t *out) {
  if (table >= 0 && table < CTR_TABLES && g_ctr[table].n)
    memcpy(out, g_ctr[table].row, g_ctr[table].n * (size_t)ctr_fields[table] * sizeof *out);
}
uint64_t pim_ctr_outside(int cls) { return cls >= 0 && cls < 4 ? g_ctr_outside[cls] : 0; }
uint64_t pim_ctr_word_overflow(void) { return g_ctr_word_overflow; }
uint64_t pim_ctr_dump_overflow(void) { return g_ctr_dump_overflow; }
int pim_ctr_tensors(void) { return num_tensors; }
/* what: 0 their role (1 input, 2 weight, 3 output, 0 none), 1 our role, 2 bank_replicated,
 * 3 elements, 4 bytes per element, 5 replicated lane dims (-1 not stated), 6 lane axes,
 * 7 + 2i and 8 + 2i lane axis i's store dim and lanes, fastest first. */
int pim_ctr_tensor(int tid, int what) {
  if (tid < 0 || tid >= num_tensors) return -1;
  const tensor_info_t *t = &tensors[tid];
  switch (what) {
  case 0: return g_opt_role[tid];
  case 1: return (int)t->role;
  case 2: return t->bank_replicated;
  case 3: return t->num_elements;
  case 4: return t->elem_size;
  case 5: return t->opt_lane_rep;
  case 6: return t->opt_lane_n;
  }
  int i = (what - 7) / 2;
  if (what >= 7 && i < t->opt_lane_n)
    return (what - 7) % 2 ? t->opt_lane_ext[i] : t->opt_lane_dim[i];
  return -1;
}
/* The residency stamp as read at registration, -1 not stated. */
int pim_ctr_residency(int tid) {
  return tid >= 0 && tid < num_tensors ? tensors[tid].opt_res : -1;
}

/* im-optipim-mac's call. Per lane it counts the calls, the contractions they open (j == 0),
 * the shortest and longest run of j, and any j out of turn. Only the native emitter's NE_MAC
 * issues anything for it. */
static void ne_mac(int32_t j);
static uint64_t g_mac_calls[PIM_MAX_OPT_LANES], g_mac_runs[PIM_MAX_OPT_LANES];
static uint32_t g_mac_run[PIM_MAX_OPT_LANES], g_mac_min[PIM_MAX_OPT_LANES];
static uint32_t g_mac_max[PIM_MAX_OPT_LANES];
static uint64_t g_mac_unordered;

static void mac_close(int l) {
  if (!g_mac_run[l]) return;
  if (!g_mac_min[l] || g_mac_run[l] < g_mac_min[l]) g_mac_min[l] = g_mac_run[l];
  if (g_mac_run[l] > g_mac_max[l]) g_mac_max[l] = g_mac_run[l];
  g_mac_run[l] = 0;
}

static void mac_reset(void) {
  memset(g_mac_calls, 0, sizeof g_mac_calls);
  memset(g_mac_runs, 0, sizeof g_mac_runs);
  memset(g_mac_run, 0, sizeof g_mac_run);
  memset(g_mac_min, 0, sizeof g_mac_min);
  memset(g_mac_max, 0, sizeof g_mac_max);
  g_mac_unordered = 0;
}

void __pim_opt_mac(int32_t j) {
  ne_mac(j);
  int l = __pim_get_bank_id();
  if (l < 0 || l >= PIM_MAX_OPT_LANES) {
    g_mac_unordered++;
    return;
  }
  if (j == 0) {
    mac_close(l);
    g_mac_runs[l]++;
  }
  if (j < 0 || (uint32_t)j != g_mac_run[l]) g_mac_unordered++;
  g_mac_run[l]++;
  g_mac_calls[l]++;
}

/* what: 0 calls, 1 contractions, 2 shortest and 3 longest run, 4 the stated MACs per
 * contraction (-1 unstated). */
int64_t pim_opt_mac_count(int lane, int what) {
  if (lane < 0 || lane >= PIM_MAX_OPT_LANES) return -1;
  uint32_t lo = g_mac_min[lane], hi = g_mac_max[lane], open = g_mac_run[lane];
  if (open && (!lo || open < lo)) lo = open;
  if (open > hi) hi = open;
  switch (what) {
  case 0: return (int64_t)g_mac_calls[lane];
  case 1: return (int64_t)g_mac_runs[lane];
  case 2: return lo;
  case 3: return hi;
  case 4: return lane < __pim_opt_macs_count ? __pim_opt_macs[lane] : -1;
  }
  return -1;
}
uint64_t pim_opt_mac_unordered(void) { return g_mac_unordered; }

/* Replay coverage as the dcc-parity tree has it, marked before anything can drop an access:
 * every input element loaded and every output stored, which correct:true misses when the
 * reference equals the initial buffer. */
static uint8_t *g_cov[MAX_TENSORS][2];
static int g_cov_failures = -1;
static uint64_t g_cov_drops = 0;   /* COMPUTE stores to an input, a failure as in the dcc tree */

static void cov_mark(int tid, const tensor_info_t *t, uint64_t base, uint64_t n, int w) {
  if (t->num_elements <= 0) return;
  if (!g_cov[tid][w]) g_cov[tid][w] = (uint8_t *)calloc(((size_t)t->num_elements + 7) / 8, 1);
  if (!g_cov[tid][w]) { fprintf(stderr, "[pim-runtime] ERROR: coverage alloc\n"); exit(1); }
  uint64_t es = (uint64_t)(t->elem_size > 0 ? t->elem_size : 1);
  uint64_t e0 = (base - (uint64_t)(uintptr_t)t->base_addr) / es;
  for (uint64_t e = e0; e < e0 + n && e < (uint64_t)t->num_elements; e++)
    g_cov[tid][w][e >> 3] |= (uint8_t)(1u << (e & 7));
}

static int cov_count(const uint8_t *m, int n) {
  int c = 0;
  if (m) for (int e = 0; e < n; e++) c += (m[e >> 3] >> (e & 7)) & 1;
  return c;
}

static void cov_report(void) {
  g_cov_failures = 0;
  for (int i = 0; i < num_tensors; i++) {
    const tensor_info_t *t = &tensors[i];
    int n = t->num_elements;
    if (n <= 0) continue;
    int rd = cov_count(g_cov[i][0], n), wr = cov_count(g_cov[i][1], n);
    int acc = t->role == PIM_ROLE_ACCUMULATOR;
    int ok = acc ? wr == n : rd == n;
    fprintf(stderr, "[pim-runtime] coverage tensor %d %s: %d/%d read, %d/%d written%s\n",
            i, acc ? "accumulator" : "input", rd, n, wr, n, ok ? "" : "  NOT COVERED");
    g_cov_failures += !ok;
    free(g_cov[i][0]); free(g_cov[i][1]); g_cov[i][0] = g_cov[i][1] = NULL;
  }
  if (g_cov_drops) {
    fprintf(stderr, "[pim-runtime] coverage: %llu stores to an input dropped  NOT COVERED\n",
            (unsigned long long)g_cov_drops);
    g_cov_failures++;
  }
}

/* A run that never reached finalize must not leave its bits for the next one. */
static void cov_reset(void) {
  for (int i = 0; i < MAX_TENSORS; i++)
    for (int w = 0; w < 2; w++) { free(g_cov[i][w]); g_cov[i][w] = NULL; }
  g_cov_failures = -1;
  g_cov_drops = 0;
}

/* -1 before any finalize, else the number of tensors the run left uncovered. */
int pim_opt_coverage_failures(void) { return g_cov_failures; }

/* THE NATIVE EMITTER (PIM_OPT_NATIVE = OPT_EMIT | NE_ bits). One command per placed word a load
 * or store issued, W, BR or R by the residency stamp, or under NE_MAC the transfers of each
 * im-optipim-mac call. It reads no count opt_stats, opt_emit or cv_flush decide with. */
enum { OPT_EMIT = 8192, NE_ORDER = 16384, NE_MAC = 32768 };
/* Their codegen_new's conventions and bugs (fimdram.cpp:219-322, single_bank_opt), reproduced
 * on purpose so the README can disclose each. A bit clear is the reading without it, but for
 * FAN: the base already writes every bank, so the reading without FAN sets BCAST. */
enum {
  NE_PEBASE = 65536,      /* theirs: PE-register W and R as DRAM at the bank base (:26,44,69) */
  NE_SA0 = 131072,        /* theirs: weight row t mod 512 in subarray 0, count[row] (:291-298) */
  NE_FAN = 262144,        /* theirs: each bank written apart with the leader's slots (:55-75) */
  NE_CAP16 = 524288,      /* theirs: input to banks below 16, no pseudo-channel level (:227-233) */
  NE_LEADER_CH = 1048576, /* theirs: bank 0's bank reads stand for the channel (:274-283) */
  NE_R0 = 2097152,        /* theirs: only the leader's partials read out (:41-44,277-283) */
  NE_BUGS = NE_PEBASE | NE_SA0 | NE_FAN | NE_CAP16 | NE_LEADER_CH | NE_R0,
  /* The machine's all-bank write, which their codegen never issues (Lee ISCA'21 III-A): lanes of
   * one pseudo-channel holding one input slice take one write. FAN overrides it. */
  NE_BCAST = 4194304,
  NE_BITS = NE_ORDER | NE_MAC | NE_BUGS | NE_BCAST
};
enum { NE_IN, NE_MAC_PH, NE_OUT };   /* phases within a dispatch: input, MAC, read-out */
enum { NC_DISP, NC_LANE, NC_TID, NC_WRITE, NC_CALLS, NC_WORDS, NC_EMITTED, NC_FOLDED,
       NC_REFUSED, NC_DROPPED, NC_FIELDS };
enum { ND_EPOCH, ND_TILE, ND_PID_X, ND_PID_Y, ND_PID_Z, ND_T, ND_FIELDS };
enum { NS_LINES, NS_WORDS, NS_FOLDED, NS_RESIDUE, NS_UNSTATED, NS_CONFLICT, NS_SYNTHESIZED,
       NS_T_MISMATCH, NS_T_UNSTATED, NS_OFFBANK, NS_HOST, NS_OVERFLOW, NS_W, NS_BR, NS_R,
       NS_FIELDS };
/* NE_MAC, per (dispatch, lane): its MAC calls and stated count, the words its absorbed accesses
 * issued per phase, and the transfers its MACs issued and emitted per phase. */
enum { MC_DISP, MC_LANE, MC_MACS, MC_STAMP, MC_UNORDERED, MC_IN, MC_WT, MC_OUT, MC_W, MC_BR,
       MC_R, MC_W_EMITTED, MC_BR_EMITTED, MC_R_EMITTED, MC_FOLDED, MC_REFUSED, MC_DROPPED,
       MC_FIELDS };
/* Signed: an arranged counter, transfers less words absorbed, goes negative where a lane issued
 * more words than it ran MACs. */
enum { NM_MACS, NM_UNORDERED, NM_OFF_STAMP, NM_NO_WEIGHT, NM_UNABSORBED_W, NM_UNABSORBED_BR,
       NM_UNABSORBED_R, NM_UNPLACED, NM_ARRANGED_W, NM_ARRANGED_BR, NM_ARRANGED_R, NM_FIELDS };
/* Lane transfers before any fold, not lines. Per bug bit, those it dropped and how many held a
 * slice no kept transfer of the dispatch holds, a BR or R slice being the MAC's work. PEBASE and
 * SA0 drop nothing, and their moved count every line they place, already there or not. FAN
 * drops no transfer, it counts the writes carrying the leader's slots for the lane's own,
 * distinct where the two differ. lane_order counts lanes come after a higher one in a phase. */
enum { NB_PEBASE_DROPPED, NB_PEBASE_DISTINCT, NB_SA0_DROPPED, NB_SA0_DISTINCT, NB_FAN_DROPPED,
       NB_FAN_DISTINCT, NB_CAP16_DROPPED, NB_CAP16_DISTINCT, NB_LEADER_CH_DROPPED,
       NB_LEADER_CH_DISTINCT, NB_R0_DROPPED, NB_R0_DISTINCT, NB_PEBASE_MOVED, NB_SA0_MOVED,
       NB_UNSTATED, NB_BCAST_FOLDED, NB_LANE_ORDER, NB_FIELDS };
#define NE_WORDS_MAX 256
typedef struct { uint32_t disp, lane, seq; int ph; int a[7]; } ne_rec_t;
typedef struct { uint32_t *row; size_t n, cap; int f; } ne_tab_t;
static int g_ne_mask = 0;
static ne_tab_t g_ne_cells = {NULL, 0, 0, NC_FIELDS}, g_ne_disps = {NULL, 0, 0, ND_FIELDS},
                g_ne_idx = {NULL, 0, 0, 3}, g_ne_mcells = {NULL, 0, 0, MC_FIELDS};
static ctr_map_t g_ne_cell_map, g_ne_disp_map, g_ne_idx_map, g_ne_mcell_map, g_ne_lead_map,
                 g_ne_kept_map, g_ne_last_map;
static ne_rec_t *g_ne_rec = NULL;
static size_t g_ne_n = 0, g_ne_cap = 0, g_ne_cell0 = 0, g_ne_mcell0 = 0;
static addr_dedup_state_t *g_ne_fold = NULL;
static uint64_t g_ne_stat[NS_FIELDS];
static int64_t g_ne_mstat[NM_FIELDS], g_ne_bstat[NB_FIELDS];
static uint64_t g_ne_epoch0 = 0, g_ne_prog = UINT64_MAX, g_ne_disp_key = UINT64_MAX;
static uint32_t g_ne_disp = 0, g_ne_seq = 0;
/* im_optipim_hold_input, their unbounded input memo (their bug, disclosed): the input goes out
 * only where every held program id is below its bound. Honoured only on a walk with the held
 * ids slowest over exactly the stamped tiles, so every held tile was delivered fresh before. */
enum { NH_DROPPED, NH_HELD, NH_FRESH, NH_PROGRAMS, NH_FIELDS };
static int g_ne_hold = 0, g_ne_hold_tid = -1, g_ne_walk_n = 0;
static int64_t g_ne_hstat[NH_FIELDS], g_ne_walk[7];
static int32_t g_ne_pid_max[3];
static uint64_t g_ne_hold_prog = UINT64_MAX;

static uint32_t *ne_row(ne_tab_t *t) {
  if (t->n == t->cap) {
    t->cap = t->cap ? 2 * t->cap : 4096;
    t->row = (uint32_t *)realloc(t->row, t->cap * (size_t)t->f * sizeof *t->row);
    if (!t->row) { fprintf(stderr, "[pim-runtime] ERROR: native table alloc\n"); exit(1); }
  }
  uint32_t *r = t->row + t->n++ * (size_t)t->f;
  memset(r, 0, (size_t)t->f * sizeof *r);
  return r;
}

static void ne_map_free(ctr_map_t *m) {
  free(m->k);
  free(m->v);
  memset(m, 0, sizeof *m);
}

static void ne_reset(int mask) {
  ne_tab_t *tabs[] = {&g_ne_cells, &g_ne_disps, &g_ne_idx, &g_ne_mcells};
  for (int i = 0; i < 4; i++) { free(tabs[i]->row); tabs[i]->row = NULL; tabs[i]->n = tabs[i]->cap = 0; }
  ne_map_free(&g_ne_cell_map);
  ne_map_free(&g_ne_disp_map);
  ne_map_free(&g_ne_idx_map);
  ne_map_free(&g_ne_mcell_map);
  ne_map_free(&g_ne_lead_map);
  ne_map_free(&g_ne_kept_map);
  ne_map_free(&g_ne_last_map);
  free(g_ne_rec);
  g_ne_rec = NULL;
  g_ne_n = g_ne_cap = g_ne_cell0 = g_ne_mcell0 = 0;
  if (g_ne_fold) { addr_dedup_destroy(g_ne_fold); g_ne_fold = NULL; }
  memset(g_ne_stat, 0, sizeof g_ne_stat);
  memset(g_ne_mstat, 0, sizeof g_ne_mstat);
  memset(g_ne_bstat, 0, sizeof g_ne_bstat);
  g_ne_mask = mask;
  g_ne_epoch0 = __pim_program_epoch;
  g_ne_prog = g_ne_disp_key = UINT64_MAX;
  g_ne_disp = g_ne_seq = 0;
  g_ne_hold = mask && __pim_opt_hold_count > 0;
  g_ne_hold_tid = -1;
  g_ne_walk_n = 0;
  memset(g_ne_hstat, 0, sizeof g_ne_hstat);
  g_ne_pid_max[0] = g_ne_pid_max[1] = g_ne_pid_max[2] = -1;
  g_ne_hold_prog = UINT64_MAX;
  if (mask && !(g_ne_fold = addr_dedup_create(1 << 16))) {
    fprintf(stderr, "[pim-runtime] ERROR: native key table alloc\n");
    exit(1);
  }
}

static int ne_cmp(const void *x, const void *y) {
  const ne_rec_t *a = (const ne_rec_t *)x, *b = (const ne_rec_t *)y;
  if (a->disp != b->disp) return a->disp < b->disp ? -1 : 1;
  if ((g_ne_mask & NE_ORDER) && a->ph != b->ph) return a->ph - b->ph;
  if (a->lane != b->lane) return a->lane < b->lane ? -1 : 1;
  return a->seq < b->seq ? -1 : a->seq > b->seq;
}

static const uint32_t *g_ne_sort_rows;
static int ne_cell_cmp(const void *x, const void *y) {
  const uint32_t *a = g_ne_sort_rows + (size_t)*(const uint32_t *)x * NC_FIELDS;
  const uint32_t *b = g_ne_sort_rows + (size_t)*(const uint32_t *)y * NC_FIELDS;
  static const int order[4] = {NC_DISP, NC_TID, NC_WRITE, NC_LANE};
  for (int f = 0; f < 4; f++)
    if (a[order[f]] != b[order[f]]) return a[order[f]] < b[order[f]] ? -1 : 1;
  return 0;
}

/* Lane agreement over cells [lo, hi): per dispatch and tensor, every lane that issued words
 * issued as many as the lowest such lane. A lane idle with none is no residue. */
static void ne_residue(size_t lo, size_t hi) {
  if (hi <= lo) return;
  uint32_t *ix = (uint32_t *)malloc((hi - lo) * sizeof *ix);
  if (!ix) { fprintf(stderr, "[pim-runtime] ERROR: native residue alloc\n"); exit(1); }
  for (size_t i = lo; i < hi; i++) ix[i - lo] = (uint32_t)i;
  g_ne_sort_rows = g_ne_cells.row;
  qsort(ix, hi - lo, sizeof *ix, ne_cell_cmp);
  for (size_t i = 0; i < hi - lo;) {
    const uint32_t *g = g_ne_cells.row + (size_t)ix[i] * NC_FIELDS;
    uint32_t lead = 0;
    size_t j = i;
    for (; j < hi - lo; j++) {
      const uint32_t *r = g_ne_cells.row + (size_t)ix[j] * NC_FIELDS;
      if (r[NC_DISP] != g[NC_DISP] || r[NC_TID] != g[NC_TID] || r[NC_WRITE] != g[NC_WRITE])
        break;
      if (!r[NC_WORDS]) continue;
      if (!lead) lead = r[NC_WORDS];
      else if (r[NC_WORDS] != lead) g_ne_stat[NS_RESIDUE]++;
    }
    i = j;
  }
  free(ix);
}

static void ne_mac_issue(void);

/* Whole-program hold, written dispatch-major: within a dispatch by phase under ORDER, then
 * lane, then program order. */
static void ne_flush(void) {
  if (!g_ne_mask) return;
  ne_residue(g_ne_cell0, g_ne_cells.n);
  g_ne_cell0 = g_ne_cells.n;
  if (g_ne_mask & NE_MAC) ne_mac_issue();
  if (g_ne_n) qsort(g_ne_rec, g_ne_n, sizeof *g_ne_rec, ne_cmp);
  static const char *const op[3] = {"W", "BR", "R"};
  for (size_t i = 0; i < g_ne_n && trace_fp; i++) {
    const int *a = g_ne_rec[i].a;
    fprintf(trace_fp, "%s %d,%d,%d,%d,%d,%d,%d\n", op[g_ne_rec[i].ph], a[0], a[1], a[2], a[3],
            a[4], a[5], a[6]);
    g_ne_stat[NS_LINES]++;
    g_ne_stat[NS_W + g_ne_rec[i].ph]++;
  }
  g_ne_n = 0;
  ne_map_free(&g_ne_idx_map);
  ne_map_free(&g_ne_lead_map);
  ne_map_free(&g_ne_kept_map);
  ne_map_free(&g_ne_last_map);
  g_ne_idx.n = 0;
  addr_dedup_reset(g_ne_fold);
}

/* 1 when a held program id of the dispatch is at or past its bound, so the input stays put. */
static int ne_held(uint32_t disp) {
  if (g_ne_hold_tid < 0) return 0;
  const uint32_t *r = g_ne_disps.row + (size_t)disp * ND_FIELDS;
  for (int a = 0; a < 3; a++)
    if (__pim_opt_hold[2 + a] >= 0 && (int32_t)r[ND_PID_X + a] >= __pim_opt_hold[2 + a])
      return 1;
  return 0;
}

/* The stamped tiles against the fresh programs walked and the launch grid's extent over the
 * ids the hold does not hold. The walk check keeps every fresh program before a held one. */
static void ne_hold_tiles(void) {
  int64_t tiles = __pim_opt_hold[5], grid = 1;
  for (int a = 0; a < 3; a++)
    if (!((__pim_opt_hold[1] >> a) & 1)) grid *= (int64_t)g_ne_pid_max[a] + 1;
  if (g_ne_hstat[NH_PROGRAMS] == tiles && grid == tiles) return;
  fprintf(stderr, "[pim-runtime] ERROR: the held input (im_optipim_hold_input) states %" PRId64
                  " tiles, the launch delivered %" PRId64 " fresh programs over a grid of %"
                  PRId64 " on the ids it does not hold\n", tiles, g_ne_hstat[NH_PROGRAMS], grid);
  exit(1);
}

/* A new dispatch under the hold: the walk must rise in (held ids, other ids, tile), slowest
 * first, which their memo needs for a fresh predicate on the held ids alone. */
static void ne_hold_dispatch(uint32_t disp, uint64_t ep, const int32_t p[3], uint64_t tile) {
  if (g_ne_hold_tid < 0) {
    /* The stamp's arg indexes tensors[] as the layout and residency records do. */
    int arg = __pim_opt_hold[0];
    if (arg < 0 || arg >= num_tensors || tensors[arg].opt_res != RES_DELIVERED) {
      fprintf(stderr, "[pim-runtime] ERROR: the held input (im_optipim_hold_input) is arg %d, "
                      "which is not the tensor stamped delivered\n", arg);
      exit(1);
    }
    g_ne_hold_tid = arg;
  }
  int64_t k[7];
  for (int a = 2, i = 0; a >= 0; a--, i++) {
    int h = (__pim_opt_hold[1] >> a) & 1;
    k[i] = h ? p[a] : 0;
    k[3 + i] = h ? 0 : p[a];
  }
  k[6] = (int64_t)tile;
  int cmp = !g_ne_walk_n;
  for (int i = 0; i < 7 && !cmp; i++)
    cmp = k[i] > g_ne_walk[i] ? 1 : k[i] < g_ne_walk[i] ? -1 : 0;
  if (cmp <= 0) {
    char ids[4] = {0};
    for (int a = 2, n = 0; a >= 0; a--)
      if ((__pim_opt_hold[1] >> a) & 1) ids[n++] = "xyz"[a];
    fprintf(stderr, "[pim-runtime] ERROR: the held input (im_optipim_hold_input) is held across "
                    "program id %s, walked slowest, and the walk reaches pid %d,%d,%d tile %"
                    PRIu64 " out of that order\n", ids, p[0], p[1], p[2], tile);
    exit(1);
  }
  memcpy(g_ne_walk, k, sizeof k);
  g_ne_walk_n = 1;
  for (int a = 0; a < 3; a++)
    if (p[a] > g_ne_pid_max[a]) g_ne_pid_max[a] = p[a];
  int held = ne_held(disp);
  g_ne_hstat[held ? NH_HELD : NH_FRESH]++;
  if (ep != g_ne_hold_prog && !held) g_ne_hstat[NH_PROGRAMS]++;
  g_ne_hold_prog = ep;
}

/* The dispatch, (epoch, tile), numbered in replay order. Their step t is the compiler's affine
 * map of the program ids and tile, which must equal that ordinal. */
static uint32_t ne_dispatch(void) {
  uint64_t tile = __pim_tile_index(), ep = __pim_program_epoch - g_ne_epoch0;
  uint64_t key = (ep << 24) | (tile & 0xFFFFFFULL);
  if (key == g_ne_disp_key) return g_ne_disp;
  if (tile >> 24) g_ne_stat[NS_OVERFLOW]++;
  if (__pim_program_epoch != g_ne_prog) {
    ne_flush();
    g_ne_prog = __pim_program_epoch;
  }
  g_ne_disp_key = key;
  int ins;
  uint32_t ord = ctr_map_get(&g_ne_disp_map, key, (uint32_t)g_ne_disps.n, &ins);
  if (ins) {
    int32_t px = __pim_get_program_id(), py = __pim_get_program_id_y(),
            pz = __pim_get_program_id_z();
    uint32_t *r = ne_row(&g_ne_disps);
    r[ND_EPOCH] = (uint32_t)ep;
    r[ND_TILE] = (uint32_t)tile;
    r[ND_PID_X] = (uint32_t)px;
    r[ND_PID_Y] = (uint32_t)py;
    r[ND_PID_Z] = (uint32_t)pz;
    int64_t t = -1;
    if (__pim_opt_macs_count > 0) {
      const int32_t *s = __pim_opt_step;
      t = (int64_t)s[0] * px + (int64_t)s[1] * py + (int64_t)s[2] * pz +
          (int64_t)s[3] * (int64_t)tile + s[4];
      if (t != (int64_t)ord) g_ne_stat[NS_T_MISMATCH]++;
    } else {
      g_ne_stat[NS_T_UNSTATED]++;
    }
    r[ND_T] = (uint32_t)t;
    if (g_ne_hold) {
      const int32_t p[3] = {px, py, pz};
      ne_hold_dispatch(ord, ep, p, tile);
    }
  }
  g_ne_disp = ord;
  return ord;
}

static uint32_t *ne_cell(uint32_t disp, int lane, int tid, int is_write) {
  uint64_t key = ((uint64_t)disp << 16) | ((uint64_t)(lane & 0x3FF) << 5) |
                 ((uint64_t)(tid & 0xF) << 1) | (uint64_t)(is_write != 0);
  int ins;
  uint32_t i = ctr_map_get(&g_ne_cell_map, key, (uint32_t)g_ne_cells.n, &ins);
  if (!ins) return g_ne_cells.row + (size_t)i * NC_FIELDS;
  uint32_t *r = ne_row(&g_ne_cells);
  r[NC_DISP] = disp;
  r[NC_LANE] = (uint32_t)lane;
  r[NC_TID] = (uint32_t)tid;
  r[NC_WRITE] = (uint32_t)(is_write != 0);
  return r;
}

/* A lane's next index within one phase of one dispatch. */
static uint32_t ne_index(uint32_t disp, int lane, int ph) {
  int ins;
  uint32_t i = ctr_map_get(&g_ne_idx_map, ((uint64_t)disp << 10) | (uint64_t)(lane & 0x3FF),
                           (uint32_t)g_ne_idx.n, &ins);
  if (ins) ne_row(&g_ne_idx);
  return g_ne_idx.row[(size_t)i * 3 + ph]++;
}

/* Ported from ramulator2_dcc's lockstep key, plus the phase: lanes issuing one command at one
 * index fold, and a lane's own repeats never do. */
static uint64_t ne_key(uint64_t ns, uint64_t disp, int ph, uint64_t idx, uint64_t row,
                       uint64_t col) {
  uint64_t h = lk_mix(ns ^ 0x9E3779B97F4A7C15ULL);
  h = lk_mix(h ^ disp);
  h = lk_mix(h ^ (((uint64_t)ph << 32) | idx));
  return lk_mix(h ^ ((row << 16) | col));
}

static void ne_hold(uint32_t disp, int lane, int ph, const pim_phys_loc_t *pl) {
  if (g_ne_n == g_ne_cap) {
    g_ne_cap = g_ne_cap ? 2 * g_ne_cap : 4096;
    g_ne_rec = (ne_rec_t *)realloc(g_ne_rec, g_ne_cap * sizeof *g_ne_rec);
    if (!g_ne_rec) { fprintf(stderr, "[pim-runtime] ERROR: native hold alloc\n"); exit(1); }
  }
  g_ne_rec[g_ne_n++] = (ne_rec_t){disp, (uint32_t)lane, g_ne_seq++, ph,
                                  {pl->ch, pl->pch, pl->bg, pl->bank, pl->sa, pl->row, pl->col}};
}

/* BR folds lanes per pseudo-channel into one all-bank command, W and R are per bank. LEADER_CH
 * widens the BR fold to the channel, R0 folds R to the channel's leader, and a W slice from
 * BCAST folds W per pseudo-channel and slice. */
static uint64_t ne_ns(int tid, int ph, const pim_phys_loc_t *pl, int gb, int64_t w_slice) {
  uint64_t ns = (uint64_t)tid | ((uint64_t)ph << 4);
  if (ph == NE_MAC_PH ? (g_ne_mask & NE_LEADER_CH) : ph == NE_OUT && (g_ne_mask & NE_R0))
    return ns | (2ULL << 6) | ((uint64_t)pl->ch << 8);
  if (w_slice >= 0)
    return ns | (3ULL << 6) | ((uint64_t)(pl->ch * cfg_num_pch + pl->pch) << 8) |
           ((uint64_t)w_slice << 24);
  return ns | (ph == NE_MAC_PH ? (uint64_t)(pl->ch * cfg_num_pch + pl->pch) << 8
                               : (1ULL << 6) | ((uint64_t)gb << 8));
}

/* 1 when the command is held, 0 when it folds into an earlier lane's at its index. */
static int ne_issue(uint64_t ns, uint32_t disp, int lane, int ph, const pim_phys_loc_t *pl) {
  uint64_t h = ne_key(ns, disp, ph, ne_index(disp, lane, ph),
                      (uint64_t)pl->sa * (uint64_t)cfg_num_rows + (uint64_t)pl->row,
                      (uint64_t)pl->col);
  if (!addr_dedup_check_and_mark(g_ne_fold, h, h >> 16, h >> 48))
    return 0;
  ne_hold(disp, lane, ph, pl);
  return 1;
}

/* The one tensor stamped with residency res, -1 for none or several. */
static int ne_role_tid(int res) {
  int tid = -1;
  for (int i = 0; i < num_tensors; i++)
    if (tensors[i].opt_res == res) {
      if (tid >= 0) return -1;
      tid = i;
    }
  return tid;
}

/* The lane's coordinates on the lane axes outside mask rep, from im-operand-residency-layout's
 * lane statements, so two lanes with one key hold one slice. -1 when unstated. */
static int64_t ne_slice(const tensor_info_t *t, int rep, int lane) {
  if (rep < 0 || t->opt_lane_n <= 0) return -1;
  int64_t key = 0, mul = 1, rem = lane;
  for (int i = 0; i < t->opt_lane_n; i++) {
    int64_t n = t->opt_lane_ext[i] > 0 ? t->opt_lane_ext[i] : 1;
    if (!((rep >> t->opt_lane_dim[i]) & 1)) key += rem % n * mul;
    rem /= n;
    mul *= n;
  }
  return rem ? -1 : key;
}

/* The slice a transfer carries: the input's for W, and for BR and R the MAC's work, since a
 * read-out drains the lane's own partials. Lanes share work only along axes every role is
 * replicated on. */
static int64_t ne_slice_of(int ph, int tid, int lane) {
  if (tid < 0) return -1;
  int rep = tensors[tid].opt_lane_rep;
  for (int r = RES_DELIVERED; ph != NE_IN && r <= RES_DRAINED; r++) {
    int x = ne_role_tid(r);
    rep = x < 0 || tensors[x].opt_lane_rep < 0 || rep < 0 ? -1 : rep & tensors[x].opt_lane_rep;
  }
  return ne_slice(&tensors[tid], rep, lane);
}

static int ne_map_has(const ctr_map_t *m, uint64_t key) {
  if (!m->cap) return 0;
  uint64_t kk = key + 1;
  for (size_t j = ctr_mix(kk) & (m->cap - 1); m->k[j]; j = (j + 1) & (m->cap - 1))
    if (m->k[j] == kk) return 1;
  return 0;
}

static uint64_t ne_kept_key(uint32_t disp, int ph, int64_t slice) {
  return lk_mix((((uint64_t)disp << 2) | (uint64_t)ph) ^ lk_mix((uint64_t)slice)) >> 1;
}

/* ctr_map_get's value slot, so it can be changed. */
static uint32_t *ne_map_ref(ctr_map_t *m, uint64_t key, uint32_t fresh) {
  int ins;
  ctr_map_get(m, key, fresh, &ins);
  size_t j = ctr_mix(key + 1) & (m->cap - 1);
  while (m->k[j] != key + 1) j = (j + 1) & (m->cap - 1);
  return &m->v[j];
}

/* One lane transfer a bit dropped, distinct when no kept lane of the dispatch holds its slice. */
static void ne_drop(int nb, uint32_t disp, int ph, int64_t slice) {
  g_ne_bstat[nb]++;
  g_ne_bstat[nb + 1] += slice < 0 || !ne_map_has(&g_ne_kept_map, ne_kept_key(disp, ph, slice));
}

/* One transfer on its lane under the bits: 1 held, 0 folded into an earlier lane's at its
 * index, -1 dropped. The first lane to issue a phase is their leader bank and every kept lane
 * precedes a dropped one, while lanes arrive lowest first, which lane_order checks. */
static int ne_put(int tid, uint32_t disp, int lane, int ph, pim_phys_loc_t *pl, int gb) {
  const int m = g_ne_mask, per_pch = cfg_num_bg * cfg_num_banks;
  const int chan = ph == NE_MAC_PH ? (m & NE_LEADER_CH) != 0 : ph == NE_OUT && (m & NE_R0);
  const int fan = ph == NE_IN && (m & NE_FAN), cap = ph == NE_IN && (m & NE_CAP16);
  const int bcast = ph == NE_IN && (m & NE_BCAST) && !fan;
  int64_t slice = -1;
  if (fan || cap || chan || bcast) {
    slice = ne_slice_of(ph, tid, lane);
    g_ne_bstat[NB_UNSTATED] += slice < 0;
    uint32_t *last = ne_map_ref(&g_ne_last_map, ((uint64_t)disp << 2) | (uint64_t)ph,
                                (uint32_t)lane);
    if ((uint32_t)lane < *last) g_ne_bstat[NB_LANE_ORDER]++;
    else *last = (uint32_t)lane;
  }
  if (cap && lane >= per_pch) {
    ne_drop(NB_CAP16_DROPPED, disp, ph, slice);
    return -1;
  }
  if (ph != NE_MAC_PH && (m & NE_PEBASE))
    pl->sa = pl->row = pl->col = 0;
  int ins, lead = lane;
  if (fan || chan)
    lead = (int)ctr_map_get(&g_ne_lead_map, ((uint64_t)disp << 2) | (uint64_t)ph,
                            (uint32_t)lane, &ins);
  int held = ne_issue(ne_ns(tid, ph, pl, gb, bcast ? slice : -1), disp, lane, ph, pl);
  g_ne_bstat[NB_BCAST_FOLDED] += bcast && !held;
  if (!held && chan && (ph == NE_OUT || lane / per_pch != lead / per_pch)) {
    ne_drop(ph == NE_OUT ? NB_R0_DROPPED : NB_LEADER_CH_DROPPED, disp, ph, slice);
    return -1;
  }
  /* A trace carries no data: under FAN the write is the lane's, its slots the leader's. */
  if (fan && lane != lead) {
    g_ne_bstat[NB_FAN_DROPPED]++;
    g_ne_bstat[NB_FAN_DISTINCT] += slice < 0 || slice != ne_slice_of(ph, tid, lead);
  }
  if (cap || chan)
    ctr_map_get(&g_ne_kept_map, ne_kept_key(disp, ph, slice), 1, &ins);
  g_ne_bstat[NB_PEBASE_MOVED] += held && ph != NE_MAC_PH && (m & NE_PEBASE);
  return held;
}

static uint32_t *ne_mcell(uint32_t disp, int lane) {
  int ins;
  uint32_t i = ctr_map_get(&g_ne_mcell_map, ((uint64_t)disp << 10) | (uint64_t)(lane & 0x3FF),
                           (uint32_t)g_ne_mcells.n, &ins);
  if (!ins) return g_ne_mcells.row + (size_t)i * MC_FIELDS;
  uint32_t *r = ne_row(&g_ne_mcells);
  r[MC_DISP] = disp;
  r[MC_LANE] = (uint32_t)lane;
  r[MC_STAMP] = lane >= 0 && lane < __pim_opt_macs_count && lane < PIM_MAX_OPT_LANES
                    ? (uint32_t)__pim_opt_macs[lane] : UINT32_MAX;
  return r;
}

/* NE_MAC: one MAC of the lane's contraction, issued at the flush, once every access of its
 * dispatch is in. */
static void ne_mac(int32_t j) {
  if (!(g_ne_mask & NE_MAC) || !trace_fp)
    return;
  if (cur_phase != PIM_PHASE_COMPUTE) {
    g_ne_stat[NS_HOST] += cur_phase == PIM_PHASE_HOST;
    return;
  }
  uint32_t disp = ne_dispatch();
  uint32_t *m = ne_mcell(disp, __pim_get_bank_id());
  if (j < 0 || (uint32_t)j != m[MC_MACS]) {
    m[MC_UNORDERED]++;
    g_ne_mstat[NM_UNORDERED]++;
  }
  m[MC_MACS]++;
  g_ne_mstat[NM_MACS]++;
}

/* One load or store call: one command per distinct placed word it touches, or under NE_MAC
 * none, its words absorbed by the lane's MACs. A word of a tensor with no stamp, against its
 * stamp, or off the lane's bank is refused. */
static void ne_access(uint64_t base, uint64_t size, int is_write) {
  if (cur_phase != PIM_PHASE_COMPUTE) {
    g_ne_stat[NS_HOST] += cur_phase == PIM_PHASE_HOST;
    return;
  }
  int tidx = find_tensor(base);
  if (tidx < 0) return;
  tensor_info_t *t = &tensors[tidx];
  uint64_t es = (uint64_t)(t->elem_size > 0 ? t->elem_size : 1);
  uint64_t n = size / es ? size / es : 1;
  cov_mark(tidx, t, base, n, is_write);
  uint32_t disp = ne_dispatch();
  int lane = __pim_get_bank_id();
  uint32_t *cell = ne_cell(disp, lane, tidx, is_write), *mcell = NULL;
  cell[NC_CALLS]++;
  const int held = ne_held(disp);
  int cell_tid = tidx;
  uint64_t seen[NE_WORDS_MAX];
  int n_seen = 0;
  for (uint64_t i = 0; i < n; i++) {
    uint64_t a = base + i * es;
    if (a >= (uint64_t)t->base_addr + t->total_bytes) {
      int nt = find_tensor(a);
      if (nt < 0) continue;
      t = &tensors[nt];
    }
    int elem = (int)((a - (uint64_t)t->base_addr) / t->elem_size);
    if (elem < 0 || elem >= t->num_elements) continue;
    pim_phys_loc_t pl;
    pl.elem_idx = elem;
    map_element(t, elem, &pl.ch, &pl.pch, &pl.bg, &pl.bank, &pl.sa, &pl.row, &pl.col);
    int placed = t->bank_replicated >= 0 && t->lane_slot && t->lane_row_base >= 0 &&
                 lane_slab_place(t, &pl);
    int gb = compute_global_bank(pl.ch, pl.pch, pl.bg, pl.bank);
    uint64_t key = ((uint64_t)gb & 0xFFFFULL) | (((uint64_t)pl.sa & 0xFFULL) << 16) |
                   (((uint64_t)pl.row & 0xFFFFFFULL) << 24) |
                   (((uint64_t)pl.col & 0xFFFFULL) << 48);
    int dup = 0;
    for (int j = 0; j < n_seen && !dup; j++) dup = seen[j] == key;
    if (dup) continue;
    if (n_seen < NE_WORDS_MAX) seen[n_seen++] = key;
    else g_ne_stat[NS_OVERFLOW]++;
    int tid = (int)(t - tensors);
    if (tid != cell_tid) {
      cell = ne_cell(disp, lane, tid, is_write);
      cell_tid = tid;
    }
    cell[NC_WORDS]++;
    g_ne_stat[NS_WORDS]++;
    int ph = t->opt_res == RES_DELIVERED ? NE_IN : t->opt_res == RES_RESIDENT ? NE_MAC_PH
             : t->opt_res == RES_DRAINED ? NE_OUT : -1;
    int bad = ph < 0 ? NS_UNSTATED : (ph == NE_OUT) != (is_write != 0) ? NS_CONFLICT
              : !placed || gb != lane ? NS_OFFBANK : -1;
    if (bad >= 0) {
      g_ne_stat[bad]++;
      cell[NC_REFUSED]++;
      if (bad == NS_CONFLICT && is_write) g_cov_drops++;
      continue;
    }
    /* Held, the whole input stays in every lane's PE, before any MAC absorbs it. */
    if (held && tid == g_ne_hold_tid) {
      cell[NC_DROPPED]++;
      g_ne_hstat[NH_DROPPED]++;
      continue;
    }
    if (g_ne_mask & NE_MAC) {
      if (!mcell) mcell = ne_mcell(disp, lane);
      mcell[MC_IN + ph]++;
      continue;
    }
    int got = ne_put(tid, disp, lane, ph, &pl, gb);
    if (got < 0) {
      cell[NC_DROPPED]++;
    } else if (!got) {
      cell[NC_FOLDED]++;
      g_ne_stat[NS_FOLDED]++;
    } else {
      cell[NC_EMITTED]++;
    }
  }
}

static int ne_mcell_cmp(const void *x, const void *y) {
  const uint32_t *a = g_ne_mcells.row + (size_t)*(const uint32_t *)x * MC_FIELDS;
  const uint32_t *b = g_ne_mcells.row + (size_t)*(const uint32_t *)y * MC_FIELDS;
  if (a[MC_DISP] != b[MC_DISP]) return a[MC_DISP] < b[MC_DISP] ? -1 : 1;
  return a[MC_LANE] < b[MC_LANE] ? -1 : a[MC_LANE] > b[MC_LANE];
}

/* Lowest lane first, so the first lane to read a weight slot leads its pseudo-channel. MAC j
 * issues W where its lane delivered the input this dispatch, then BR and R, at slot (t, j).
 * Refused: no weight access, a count off the stamp or out of turn, an access no MAC absorbs. */
static void ne_mac_issue(void) {
  size_t lo = g_ne_mcell0, hi = g_ne_mcells.n;
  g_ne_mcell0 = hi;
  if (hi <= lo) return;
  uint32_t *ix = (uint32_t *)malloc((hi - lo) * sizeof *ix);
  if (!ix) { fprintf(stderr, "[pim-runtime] ERROR: native MAC alloc\n"); exit(1); }
  for (size_t i = lo; i < hi; i++) ix[i - lo] = (uint32_t)i;
  qsort(ix, hi - lo, sizeof *ix, ne_mcell_cmp);
  const int tid[3] = {ne_role_tid(RES_DELIVERED), ne_role_tid(RES_RESIDENT),
                      ne_role_tid(RES_DRAINED)};
  uint32_t disp_prev = UINT32_MAX, lead = 0;
  for (size_t i = 0; i < hi - lo; i++) {
    uint32_t *m = g_ne_mcells.row + (size_t)ix[i] * MC_FIELDS;
    uint32_t disp = m[MC_DISP], macs = m[MC_MACS];
    int lane = (int)m[MC_LANE];
    if (disp != disp_prev) { disp_prev = disp; lead = 0; }
    if (macs && !lead) lead = macs;
    else if (macs && macs != lead) g_ne_stat[NS_RESIDUE]++;
    if (!macs) {
      g_ne_mstat[NM_UNABSORBED_W] += m[MC_IN];
      g_ne_mstat[NM_UNABSORBED_BR] += m[MC_WT];
      g_ne_mstat[NM_UNABSORBED_R] += m[MC_OUT];
    } else {
      g_ne_mstat[NM_OFF_STAMP] += macs != m[MC_STAMP];
      if (!m[MC_WT]) g_ne_mstat[NM_NO_WEIGHT] += macs;
      int refuse = m[MC_UNORDERED] || macs != m[MC_STAMP] || !m[MC_WT];
      int64_t t = (int32_t)g_ne_disps.row[(size_t)disp * ND_FIELDS + ND_T];
      for (uint32_t j = 0; j < macs; j++)
        for (int ph = NE_IN; ph <= NE_OUT; ph++) {
          if (ph == NE_IN && !m[MC_IN]) continue;
          m[MC_W + ph]++;
          pim_phys_loc_t pl;
          int at = refuse || tid[ph] < 0 ? 0 : arranged_place(lane, t, j, ph == NE_MAC_PH &&
                                                              (g_ne_mask & NE_SA0), &pl);
          if (!at) {
            m[MC_REFUSED]++;
            g_ne_mstat[NM_UNPLACED] += !refuse;
            continue;
          }
          int got = ne_put(tid[ph], disp, lane, ph, &pl, lane);
          if (got > 0) {
            m[MC_W_EMITTED + ph]++;
            g_ne_bstat[NB_SA0_MOVED] += at == 2;
          } else {
            m[got ? MC_DROPPED : MC_FOLDED]++;
          }
        }
    }
    g_ne_mstat[NM_ARRANGED_W] += (int64_t)m[MC_W] - m[MC_IN];
    g_ne_mstat[NM_ARRANGED_BR] += (int64_t)m[MC_BR] - m[MC_WT];
    g_ne_mstat[NM_ARRANGED_R] += (int64_t)m[MC_R] - m[MC_OUT];
  }
  free(ix);
}

/* Flushes the last program and checks the two tallies: lines written against what the access
 * path and the MACs emitted. A difference, and a modelled spill, are commands nothing issued. */
static void ne_finish(void) {
  ne_flush();
  if (g_ne_hold) {
    ne_hold_tiles();
    fprintf(stderr, "[pim-runtime] native hold: %" PRId64 " input words held on %" PRId64
                    " dispatches, delivered on %" PRId64 ", %" PRId64 " fresh programs of %d "
                    "tiles stated\n", g_ne_hstat[NH_DROPPED], g_ne_hstat[NH_HELD],
            g_ne_hstat[NH_FRESH], g_ne_hstat[NH_PROGRAMS], (int)__pim_opt_hold[5]);
  }
  /* A key table that could not grow emits where it should have folded. */
  g_ne_stat[NS_OVERFLOW] += addr_dedup_saturations(g_ne_fold);
  uint64_t emitted = 0;
  for (size_t i = 0; i < g_ne_cells.n; i++)
    emitted += g_ne_cells.row[i * NC_FIELDS + NC_EMITTED];
  for (size_t i = 0; i < g_ne_mcells.n; i++)
    for (int ph = NE_IN; ph <= NE_OUT; ph++)
      emitted += g_ne_mcells.row[i * MC_FIELDS + MC_W_EMITTED + ph];
  g_ne_stat[NS_SYNTHESIZED] += g_ne_stat[NS_LINES] > emitted ? g_ne_stat[NS_LINES] - emitted
                                                              : emitted - g_ne_stat[NS_LINES];
  if (g_acc_spill_tensor >= 0)
    g_ne_stat[NS_SYNTHESIZED] += 2ULL * (uint64_t)g_acc_spill_overflow *
                                 (uint64_t)g_acc_spill_ksteps *
                                 (uint64_t)(cfg_num_channels * cfg_num_pch * cfg_num_bg *
                                            cfg_num_banks);
  fprintf(stderr, "[pim-runtime] native emitter: %" PRIu64 " lines (W %" PRIu64 ", BR %" PRIu64
                  ", R %" PRIu64 ") from %" PRIu64 " words, %" PRIu64 " folded, residue %" PRIu64
                  ", unstated %" PRIu64 ", conflict %" PRIu64 ", off-bank %" PRIu64
                  ", synthesized %" PRIu64 ", t mismatch %" PRIu64 " (unstated on %" PRIu64
                  " dispatches)\n",
          g_ne_stat[NS_LINES], g_ne_stat[NS_W], g_ne_stat[NS_BR], g_ne_stat[NS_R],
          g_ne_stat[NS_WORDS], g_ne_stat[NS_FOLDED], g_ne_stat[NS_RESIDUE],
          g_ne_stat[NS_UNSTATED], g_ne_stat[NS_CONFLICT], g_ne_stat[NS_OFFBANK],
          g_ne_stat[NS_SYNTHESIZED], g_ne_stat[NS_T_MISMATCH], g_ne_stat[NS_T_UNSTATED]);
  if (g_ne_mask & NE_MAC)
    fprintf(stderr, "[pim-runtime] native MACs: %" PRId64 " calls, %" PRId64 " out of turn, %"
                    PRId64 " lane-dispatches off the stamp, %" PRId64 " without a weight access, "
                    "unabsorbed words W %" PRId64 " BR %" PRId64 " R %" PRId64 ", %" PRId64
                    " unplaced, arranged W %" PRId64 " BR %" PRId64 " R %" PRId64 "\n",
            g_ne_mstat[NM_MACS], g_ne_mstat[NM_UNORDERED], g_ne_mstat[NM_OFF_STAMP],
            g_ne_mstat[NM_NO_WEIGHT], g_ne_mstat[NM_UNABSORBED_W], g_ne_mstat[NM_UNABSORBED_BR],
            g_ne_mstat[NM_UNABSORBED_R], g_ne_mstat[NM_UNPLACED], g_ne_mstat[NM_ARRANGED_W],
            g_ne_mstat[NM_ARRANGED_BR], g_ne_mstat[NM_ARRANGED_R]);
  if (g_ne_mask & (NE_BUGS | NE_BCAST))
    fprintf(stderr, "[pim-runtime] native bug bits: dropped, distinct FAN %" PRId64 ", %" PRId64
                    " CAP16 %" PRId64 ", %" PRId64 " LEADER_CH %" PRId64 ", %" PRId64 " R0 %"
                    PRId64 ", %" PRId64 ", moved PEBASE %" PRId64 " SA0 %" PRId64 ", %" PRId64
                    " slices unstated, BCAST folded %" PRId64 ", %" PRId64 " out of lane order\n",
            g_ne_bstat[NB_FAN_DROPPED], g_ne_bstat[NB_FAN_DISTINCT], g_ne_bstat[NB_CAP16_DROPPED],
            g_ne_bstat[NB_CAP16_DISTINCT], g_ne_bstat[NB_LEADER_CH_DROPPED],
            g_ne_bstat[NB_LEADER_CH_DISTINCT], g_ne_bstat[NB_R0_DROPPED],
            g_ne_bstat[NB_R0_DISTINCT], g_ne_bstat[NB_PEBASE_MOVED], g_ne_bstat[NB_SA0_MOVED],
            g_ne_bstat[NB_UNSTATED], g_ne_bstat[NB_BCAST_FOLDED], g_ne_bstat[NB_LANE_ORDER]);
}

/* table 0 cells [disp, lane, tid, write, calls, words, emitted, folded, refused, dropped], 1
 * dispatches [epoch, tile, pid x, y, z, t or -1], 2 MAC cells (MC_ fields, NE_MAC only). */
int pim_ne_fields(int table) {
  return table == 0 ? NC_FIELDS : table == 1 ? ND_FIELDS : table == 2 ? MC_FIELDS : 0;
}
static const ne_tab_t *ne_table(int table) {
  return table == 0 ? &g_ne_cells : table == 1 ? &g_ne_disps : table == 2 ? &g_ne_mcells : NULL;
}
uint64_t pim_ne_rows(int table) { return ne_table(table) ? ne_table(table)->n : 0; }
void pim_ne_copy(int table, uint32_t *out) {
  const ne_tab_t *t = ne_table(table);
  if (t && t->n) memcpy(out, t->row, t->n * (size_t)t->f * sizeof *out);
}
int pim_ne_mac_stats(void) { return NM_FIELDS; }
int64_t pim_ne_mac_stat(int what) { return what >= 0 && what < NM_FIELDS ? g_ne_mstat[what] : 0; }
const char *pim_ne_mac_stat_name(int what) {
  static const char *const names[NM_FIELDS] = {
      [NM_MACS] = "macs", [NM_UNORDERED] = "unordered", [NM_OFF_STAMP] = "off_stamp",
      [NM_NO_WEIGHT] = "no_weight", [NM_UNABSORBED_W] = "unabsorbed_W",
      [NM_UNABSORBED_BR] = "unabsorbed_BR", [NM_UNABSORBED_R] = "unabsorbed_R",
      [NM_UNPLACED] = "unplaced", [NM_ARRANGED_W] = "arranged_W",
      [NM_ARRANGED_BR] = "arranged_BR", [NM_ARRANGED_R] = "arranged_R"};
  return what >= 0 && what < NM_FIELDS && names[what] ? names[what] : "";
}
int pim_ne_bit_stats(void) { return NB_FIELDS; }
int64_t pim_ne_bit_stat(int what) { return what >= 0 && what < NB_FIELDS ? g_ne_bstat[what] : 0; }
const char *pim_ne_bit_stat_name(int what) {
  static const char *const names[NB_FIELDS] = {
      [NB_PEBASE_DROPPED] = "PEBASE_dropped", [NB_PEBASE_DISTINCT] = "PEBASE_distinct",
      [NB_SA0_DROPPED] = "SA0_dropped", [NB_SA0_DISTINCT] = "SA0_distinct",
      [NB_FAN_DROPPED] = "FAN_dropped", [NB_FAN_DISTINCT] = "FAN_distinct",
      [NB_CAP16_DROPPED] = "CAP16_dropped", [NB_CAP16_DISTINCT] = "CAP16_distinct",
      [NB_LEADER_CH_DROPPED] = "LEADER_CH_dropped",
      [NB_LEADER_CH_DISTINCT] = "LEADER_CH_distinct", [NB_R0_DROPPED] = "R0_dropped",
      [NB_R0_DISTINCT] = "R0_distinct", [NB_PEBASE_MOVED] = "PEBASE_moved",
      [NB_SA0_MOVED] = "SA0_moved", [NB_UNSTATED] = "lanes_unstated",
      [NB_BCAST_FOLDED] = "BCAST_folded", [NB_LANE_ORDER] = "lane_order"};
  return what >= 0 && what < NB_FIELDS && names[what] ? names[what] : "";
}
int pim_ne_hold_stats(void) { return NH_FIELDS; }
int64_t pim_ne_hold_stat(int what) { return what >= 0 && what < NH_FIELDS ? g_ne_hstat[what] : 0; }
const char *pim_ne_hold_stat_name(int what) {
  static const char *const names[NH_FIELDS] = {
      [NH_DROPPED] = "dropped", [NH_HELD] = "held", [NH_FRESH] = "fresh",
      [NH_PROGRAMS] = "programs"};
  return what >= 0 && what < NH_FIELDS && names[what] ? names[what] : "";
}
int pim_ne_stats(void) { return NS_FIELDS; }
uint64_t pim_ne_stat(int what) { return what >= 0 && what < NS_FIELDS ? g_ne_stat[what] : 0; }
/* By name, so a reordered NS_ enum cannot relabel a refusal tally in the harness. */
const char *pim_ne_stat_name(int what) {
  static const char *const names[NS_FIELDS] = {
      [NS_LINES] = "lines", [NS_WORDS] = "words", [NS_FOLDED] = "folded",
      [NS_RESIDUE] = "residue", [NS_UNSTATED] = "unstated", [NS_CONFLICT] = "conflict",
      [NS_SYNTHESIZED] = "synthesized", [NS_T_MISMATCH] = "t_mismatch",
      [NS_T_UNSTATED] = "t_unstated", [NS_OFFBANK] = "offbank", [NS_HOST] = "host",
      [NS_OVERFLOW] = "overflow", [NS_W] = "W", [NS_BR] = "BR", [NS_R] = "R"};
  return what >= 0 && what < NS_FIELDS && names[what] ? names[what] : "";
}
int pim_ne_mask(void) { return g_ne_mask; }

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
    g_cv_tid = (int)(t - tensors);
    g_cv_capture = (g_opt_native & OPT_CONVENTIONS) != 0;
    if (!is_write) {
      /* (duplicated bcast_scalar BR-override removed 2026-06-22.) */
      switch (t->role) {
      case PIM_ROLE_STREAMED:
        emit_trace("BR", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa,
                   loc->row, loc->col);
        stat_bank_reads++;
        t->emitted_br++;
        ctr_add(t, 0, CC_BR);
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
        /* HOW MANY BANKS ACTUALLY RECEIVE IT. bank_replicated is the compiler's count:
         * 1 means every bank (a scalar splat, or an artifact built before the count
         * existed), >1 means exactly that many. A value occupying one lane axis of a
         * multi-axis split is replicated only along the axes it misses, so charging one
         * write per replay lane billed OptiPIM's own 2-on-M-by-16-on-N mapping 16x over.
         * Identical to the old behaviour whenever the split has a single axis. */
        int all_b = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
        int recv = (t->bank_replicated > 1 && t->bank_replicated < all_b)
                       ? t->bank_replicated : all_b;
        int stride = (recv > 0) ? (all_b / recv) : 1;
        if (stride > 1 && (__pim_get_bank_id() % stride) != 0) {
          stat_operand_repl_skips++;
          ctr_add(t, 0, CC_SKIPPED);
          break;
        }
        int dch, dpch, dbg, dbank;
        decompose_global_bank(__pim_get_bank_id(), &dch, &dpch, &dbg, &dbank);
        emit_trace("W", dch, dpch, dbg, dbank, loc->sa, loc->row, loc->col);
        stat_writes++;
        t->emitted_w++;
        ctr_add(t, 0, CC_W);
        break;
      }
      case PIM_ROLE_ACCUMULATOR:
        emit_trace("R", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa,
                   loc->row, loc->col);
        stat_reads++;
        t->emitted_r++;
        ctr_add(t, 0, CC_R);
        break;
      }
    } else {
      if (t->role == PIM_ROLE_ACCUMULATOR) {
        emit_trace("BW", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa,
                   loc->row, loc->col);
        stat_bank_writes++;
        t->emitted_bw++;
        ctr_add(t, 1, CC_BW);
      } else {
        ctr_add(t, 1, CC_DROPPED);
        g_cov_drops++;
      }
      /* Stores to STREAMED/OPERAND in COMPUTE phase are ignored (read-only) */
    }
    g_cv_capture = 0;
  } else if (cur_phase == PIM_PHASE_HOST) {
    if (!is_write) {
      emit_trace("R", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa, loc->row,
                 loc->col);
      stat_reads++;
      t->emitted_r++;
      g_ctr_outside[2]++;
    } else {
      emit_trace("W", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa, loc->row,
                 loc->col);
      stat_writes++;
      t->emitted_w++;
      g_ctr_outside[0]++;
    }
  }
}

/* (emit_per_bg_bcast_fanout + alloc_dup_rows_shared removed 2026-06-22 with the
 * broadcast/row-duplicate blank-dedup machinery.) */

/* ================================================================
 *  Public API
 * ================================================================ */

static void opt_reset(void) {
  for (int i = 0; i < MAX_TENSORS; i++) {
    free(g_opt_seen[i]);
    g_opt_seen[i] = NULL;
    free(g_opt_stamp[i]);
    g_opt_stamp[i] = NULL;
    g_opt_role[i] = OPT_NONE;
  }
  memset(g_opt_active, 0, sizeof g_opt_active);
  free(g_opt_steps);
  g_opt_steps = NULL;
  g_opt_n = g_opt_cap = 0;
  stat_opt_novel = stat_opt_cmds = 0;
  stat_opt_lanes = 0;
  memset(stat_opt_fp_min, 0, sizeof stat_opt_fp_min);
  memset(stat_opt_fp_max, 0, sizeof stat_opt_fp_max);
  g_cv_n = g_cv_on = 0;
  memset(g_cv_store, 0, sizeof g_cv_store);
  stat_cv_staged = stat_cv_lines = stat_cv_novel_dropped = stat_cv_residue = 0;
  stat_cv_leader_dropped = stat_cv_tile_empty = stat_cv_pos_unmapped = 0;
  stat_cv_unfanned = stat_cv_lane0_idle = 0;
  stat_cv_lanes_min = stat_cv_lanes_max = -1;
  g_cv_tid = -1;
  g_cv_capture = 0;
}

/* One instrumented access under PIM_OPT_NATIVE: open a step on a new dispatch, and mark
 * it novel when the leader lane loads an input element it has not held before. Their
 * memo is keyed per PE slot (fimdram.cpp:301-306); ours is per element, which agrees
 * whenever a slot sees one element per step. A bank is in their mapping when it computes
 * outputs, so a lane counts once it stores: a padded lane still loads a replicated input
 * that has no axis to mask it by. */
static void opt_note_range(tensor_info_t *t, uint64_t base, uint64_t size, int is_write) {
  int lane = __pim_get_bank_id();
  if (is_write && lane >= 0 && lane < MAX_BANKS)
    g_opt_active[lane] = 1;
  if (g_opt_n == 0 || g_opt_steps[g_opt_n - 1].epoch != __pim_program_epoch) {
    if (g_opt_n == g_opt_cap) {
      g_opt_cap = g_opt_cap ? 2 * g_opt_cap : 4096;
      g_opt_steps = (opt_step_t *)realloc(g_opt_steps, g_opt_cap * sizeof *g_opt_steps);
      if (!g_opt_steps) { fprintf(stderr, "[pim-runtime] ERROR: step alloc\n"); exit(1); }
    }
    g_opt_steps[g_opt_n++] = (opt_step_t){__pim_program_epoch, 0, {0, 0, 0, 0}};
  }
  int tid = (int)(t - tensors);
  int role = g_opt_role[tid];
  /* the leader's input and weight loads and its output stores */
  if (lane != 0 || role == OPT_NONE || (role == OPT_OUTPUT) != (is_write != 0))
    return;
  if (!g_opt_stamp[tid]) {
    g_opt_stamp[tid] = (uint32_t *)calloc((size_t)t->num_elements, sizeof(uint32_t));
    if (!g_opt_stamp[tid]) { fprintf(stderr, "[pim-runtime] ERROR: stamp alloc\n"); exit(1); }
  }
  if (role == OPT_INPUT && !g_opt_seen[tid]) {
    g_opt_seen[tid] = (uint8_t *)calloc((size_t)t->num_elements, 1);
    if (!g_opt_seen[tid]) { fprintf(stderr, "[pim-runtime] ERROR: memo alloc\n"); exit(1); }
  }
  opt_step_t *st = &g_opt_steps[g_opt_n - 1];
  uint32_t step = (uint32_t)g_opt_n;
  int es = t->elem_size > 0 ? t->elem_size : 1;
  uint64_t n = size / (uint64_t)es;
  if (n == 0) n = 1;
  for (uint64_t i = 0; i < n; i++) {
    int64_t idx = (int64_t)((base + i * (uint64_t)es - (uint64_t)t->base_addr) / (uint64_t)es);
    if (idx < 0 || idx >= t->num_elements)
      continue;
    if (g_opt_stamp[tid][idx] != step) {
      g_opt_stamp[tid][idx] = step;
      st->fp[role]++;
    }
    if (role == OPT_INPUT && !g_opt_seen[tid][idx]) {
      g_opt_seen[tid][idx] = 1;
      st->novel = 1;
    }
  }
}

/* The run's banks, commands per tensor per step and per-role footprints, for the record. */
static int opt_stats(int *L_out) {
  int S = 0;
  for (int b = 0; b < MAX_BANKS; b++)
    S += g_opt_active[b];
  int per_pch = cfg_num_bg * cfg_num_banks;
  if (L_out) *L_out = S < per_pch ? S : per_pch;
  int n = cfg_dq_bits / cfg_data_width_bits;
  if (n < 1) n = 1;
  int C = (int)__pim_opt_cells;
  /* Their flush drops the slot that triggers it, which with one bank is the last. */
  int kept = S > 1 ? C : C - 1;
  int E = kept > 0 ? (kept + n - 1) / n : 0;
  stat_opt_lanes = S;
  stat_opt_cmds = (uint64_t)E;
  stat_opt_novel = 0;
  for (size_t i = 0; i < g_opt_n; i++)
    stat_opt_novel += g_opt_steps[i].novel != 0;
  for (int r = 1; r < 4; r++) {
    stat_opt_fp_min[r] = g_opt_n ? g_opt_steps[0].fp[r] : 0;
    stat_opt_fp_max[r] = stat_opt_fp_min[r];
    for (size_t i = 1; i < g_opt_n; i++) {
      int v = g_opt_steps[i].fp[r];
      if (v < stat_opt_fp_min[r]) stat_opt_fp_min[r] = v;
      if (v > stat_opt_fp_max[r]) stat_opt_fp_max[r] = v;
    }
  }
  return E;
}

/* Their per-step stream, in their order: the W block, the BR block, the R block. */
static void opt_emit(void) {
  int L = 0, E = opt_stats(&L);
  for (size_t i = 0; i < g_opt_n; i++) {
    int row = (int)(i % (size_t)cfg_num_rows);
    if (g_opt_steps[i].novel) {
      for (int b = 0; b < L; b++) {
        int ch, pch, bg, ba;
        decompose_global_bank(b, &ch, &pch, &bg, &ba);
        for (int e = 0; e < E; e++)
          emit_trace("W", ch, pch, bg, ba, 0, 0, 0);
        stat_writes += (uint64_t)E;
      }
    }
    for (int w = 0; w < E; w++)
      emit_trace("BR", 0, 0, 0, 0, 0, row, w);
    for (int w = 0; w < E; w++)
      emit_trace("R", 0, 0, 0, 0, 0, 0, 0);
    stat_bank_reads += (uint64_t)E;
    stat_reads += (uint64_t)E;
  }
}

uint64_t pim_opt_steps(void) { return (uint64_t)g_opt_n; }
uint64_t pim_opt_novel(void) { return stat_opt_novel; }
uint64_t pim_opt_lanes(void) { return (uint64_t)stat_opt_lanes; }
uint64_t pim_opt_cmds(void) { return stat_opt_cmds; }
uint64_t pim_opt_cells(void) { return (uint64_t)__pim_opt_cells; }
uint64_t pim_opt_decided(void) { return (uint64_t)__pim_opt_decided; }
/* The leader's distinct elements per step, fewest and most over the run, by role. */
uint64_t pim_opt_fp_min(int role) { return role > 0 && role < 4 ? (uint64_t)stat_opt_fp_min[role] : 0; }
uint64_t pim_opt_fp_max(int role) { return role > 0 && role < 4 ? (uint64_t)stat_opt_fp_max[role] : 0; }

/* ---- their conventions applied to our own held commands ---- */
static void cv_stage(const char *op, int ch, int pch, int bg, int bank, int sa, int row,
                     int col) {
  if (g_cv_n == g_cv_cap) {
    g_cv_cap = g_cv_cap ? 2 * g_cv_cap : 4096;
    g_cv = (cv_ev_t *)realloc(g_cv, g_cv_cap * sizeof *g_cv);
    if (!g_cv) { fprintf(stderr, "[pim-runtime] ERROR: convention buffer alloc\n"); exit(1); }
  }
  cv_ev_t *e = &g_cv[g_cv_n];
  snprintf(e->op, sizeof e->op, "%s", op);
  int a[7] = {ch, pch, bg, bank, sa, row, col};
  memcpy(e->a, a, sizeof a);
  e->origin = __pim_get_bank_id();
  e->orole = g_cv_tid >= 0 && g_cv_tid < MAX_TENSORS ? g_opt_role[g_cv_tid] : OPT_NONE;
  e->seq = (uint32_t)g_cv_n++;
  stat_cv_staged++;
}

static void cv_out(const cv_ev_t *e) {
  if (g_cv_on == g_cv_ocap) {
    g_cv_ocap = g_cv_ocap ? 2 * g_cv_ocap : 4096;
    g_cv_out = (cv_ev_t *)realloc(g_cv_out, g_cv_ocap * sizeof *g_cv_out);
    if (!g_cv_out) { fprintf(stderr, "[pim-runtime] ERROR: convention buffer alloc\n"); exit(1); }
  }
  g_cv_out[g_cv_on] = *e;
  g_cv_out[g_cv_on].seq = (uint32_t)g_cv_on;
  g_cv_on++;
}

/* One pass: the output list becomes the held list. */
static void cv_swap(void) {
  cv_ev_t *p = g_cv; size_t c = g_cv_cap;
  g_cv = g_cv_out; g_cv_cap = g_cv_ocap; g_cv_n = g_cv_on;
  g_cv_out = p; g_cv_ocap = c; g_cv_on = 0;
}

static int cv_group(const cv_ev_t *e) { return e->a[0] * cfg_num_pch + e->a[1]; }
static int cv_bank(const cv_ev_t *e) { return compute_global_bank(e->a[0], e->a[1], e->a[2], e->a[3]); }
static int cv_class(const cv_ev_t *e) {
  return e->orole == OPT_INPUT ? 0 : e->orole == OPT_WEIGHT ? 1 : e->orole == OPT_OUTPUT ? 2 : 3;
}
static int cv_cmp(const void *x, const void *y) {
  const cv_ev_t *a = (const cv_ev_t *)x, *b = (const cv_ev_t *)y;
  int ca = cv_class(a), cb = cv_class(b);
  if (ca != cb) return ca - cb;
  if (ca == 0 && cv_bank(a) != cv_bank(b)) return cv_bank(a) - cv_bank(b);
  return a->seq < b->seq ? -1 : a->seq > b->seq;
}

/* Copy e to every bank of its group that stored in this dispatch, in bank order, or under
 * LEADER to the first `cap` storing banks, where their broadcast stops. A copy that finds no
 * storing bank is counted: the rung would have lost it. */
static void cv_fan(const cv_ev_t *e, int per, int cap) {
  int g = cv_group(e), lo = cap ? 0 : g * per, hi = cap ? MAX_BANKS : (g + 1) * per, n = 0;
  for (int b = lo; b < hi && b < MAX_BANKS && (!cap || n < cap); b++) {
    if (!g_cv_store[b]) continue;
    cv_ev_t f = *e;
    decompose_global_bank(b, &f.a[0], &f.a[1], &f.a[2], &f.a[3]);
    cv_out(&f);
    n++;
  }
  if (!n) stat_cv_unfanned++;
}

static void cv_flush(void) {
  if (!(g_opt_native & OPT_CONVENTIONS) || !trace_fp) return;
  int m = g_opt_native, per = cfg_num_bg * cfg_num_banks;
  size_t t = g_opt_n ? g_opt_n - 1 : 0;
  int novel = g_opt_n ? g_opt_steps[t].novel : 0;
  int S = 0, ngroups = cfg_num_channels * cfg_num_pch, leader[64];
  for (int b = 0; b < MAX_BANKS; b++) S += g_cv_store[b];
  if (g_cv_n) {
    if (stat_cv_lanes_min < 0 || S < stat_cv_lanes_min) stat_cv_lanes_min = S;
    if (S > stat_cv_lanes_max) stat_cv_lanes_max = S;
    /* Their leader is bank 0. With lane 0 idle no rung could put their commands there. */
    if (!g_cv_store[0]) stat_cv_lane0_idle++;
  }
  if (ngroups > 64) ngroups = 64;
  for (int g = 0; g < ngroups; g++) {
    leader[g] = g * per;
    for (int b = g * per; b < (g + 1) * per && b < MAX_BANKS; b++)
      if (g_cv_store[b]) { leader[g] = b; break; }
  }
  int nv = cfg_dq_bits / cfg_data_width_bits;
  if (nv < 1) nv = 1;
  int C = (int)__pim_opt_cells, kept = (m & CV_S1DROP) && S == 1 ? C - 1 : C;
  int E = kept > 0 ? (kept + nv - 1) / nv : 0;
  /* ROLE and READOUT retag, NOVEL drops the input of a step with nothing new. */
  for (size_t i = 0; i < g_cv_n; i++) {
    cv_ev_t e = g_cv[i];
    int store = !strcmp(e.op, "BW");
    if ((m & CV_ROLE) && !store && e.orole == OPT_INPUT) snprintf(e.op, sizeof e.op, "W");
    if ((m & CV_ROLE) && !store && e.orole == OPT_WEIGHT) snprintf(e.op, sizeof e.op, "BR");
    if ((m & CV_READOUT) && store && e.orole == OPT_OUTPUT) snprintf(e.op, sizeof e.op, "R");
    if ((m & CV_NOVEL) && !novel && e.orole == OPT_INPUT) { stat_cv_novel_dropped++; continue; }
    cv_out(&e);
  }
  cv_swap();
  /* TILE_BR (the weight) and TILE_WR (input and output): each (group, role) block becomes
   * E commands at the leader, over our addresses. */
  if (m & (CV_TILE_BR | CV_TILE_WR)) {
    uint8_t done[64][4] = {{0}};
    for (size_t i = 0; i < g_cv_n; i++) {
      const cv_ev_t *e = &g_cv[i];
      int g = cv_group(e), r = e->orole;
      int tiled = r == OPT_WEIGHT ? (m & CV_TILE_BR) : (r == OPT_INPUT || r == OPT_OUTPUT)
                                                           ? (m & CV_TILE_WR) : 0;
      if (!tiled || g >= 64) { cv_out(e); continue; }
      if (done[g][r]) continue;
      done[g][r] = 1;
      size_t k = 0;
      for (size_t j = i; j < g_cv_n; j++)
        if (cv_group(&g_cv[j]) == g && g_cv[j].orole == r) k++;
      for (int w = 0; w < E; w++) {
        size_t want = (size_t)w % k;
        for (size_t j = i, c = 0; j < g_cv_n; j++) {
          if (cv_group(&g_cv[j]) != g || g_cv[j].orole != r) continue;
          if (c++ == want) { cv_ev_t f = g_cv[j]; f.origin = leader[g]; cv_out(&f); break; }
        }
      }
    }
    /* A tiled role a step should carry and does not: the top rung would not be theirs. */
    for (int g = 0; g < ngroups; g++) {
      if (!(done[g][1] | done[g][2] | done[g][3])) continue;
      for (int r = 1; r < 4; r++) {
        int tiled = r == OPT_WEIGHT ? (m & CV_TILE_BR) : (m & CV_TILE_WR);
        if (tiled && !done[g][r] && !(r == OPT_INPUT && (m & CV_NOVEL) && !novel))
          stat_cv_tile_empty++;
      }
    }
    cv_swap();
  }
  /* LEADER, then FAN and READOUT's per-bank copies from each group's leader, then R0. */
  for (size_t i = 0; i < g_cv_n; i++) {
    const cv_ev_t *e = &g_cv[i];
    int g = cv_group(e);
    if ((m & CV_LEADER) && e->origin != 0) { stat_cv_leader_dropped++; continue; }
    int fan = (m & CV_FAN) && e->orole == OPT_INPUT;
    int readout = (m & CV_READOUT) && e->orole == OPT_OUTPUT && !strcmp(e->op, "R");
    if ((fan || readout) && g < 64 && e->origin != leader[g]) { stat_cv_residue++; continue; }
    if (fan) {
      cv_fan(e, per, (m & CV_LEADER) ? (S < per ? S : per) : 0);
      continue;
    }
    if (readout) {
      size_t from = g_cv_on;
      cv_fan(e, per, 0);
      if (m & CV_R0) {   /* keep the leader bank's copy only */
        size_t w = from;
        for (size_t j = from; j < g_cv_on; j++)
          if (g < 64 && cv_bank(&g_cv_out[j]) == leader[g]) g_cv_out[w++] = g_cv_out[j];
        g_cv_on = w;
      }
      continue;
    }
    cv_out(e);
  }
  cv_swap();
  /* POS: banks by place among the storing banks (the rest after them), the weight on row t
   * of their subarray, everything else at its bank's base. */
  if (m & CV_POS) {
    int col[64] = {0}, place[MAX_BANKS], n = 0;
    for (int b = 0; b < MAX_BANKS; b++)
      if (g_cv_store[b]) place[b] = n++;
    for (int b = 0; b < MAX_BANKS; b++)
      if (!g_cv_store[b]) place[b] = n++;
    for (size_t i = 0; i < g_cv_n; i++) {
      cv_ev_t *e = &g_cv[i];
      int g = cv_group(e), gb = cv_bank(e);
      if (gb >= 0 && gb < MAX_BANKS)
        decompose_global_bank(place[gb], &e->a[0], &e->a[1], &e->a[2], &e->a[3]);
      e->a[4] = e->a[5] = e->a[6] = 0;
      if (e->orole == OPT_WEIGHT && g < 64) {
        e->a[5] = (int)(t % (size_t)cfg_num_rows);
        e->a[6] = col[g]++;
        if (e->a[6] >= cfg_num_cols) stat_cv_pos_unmapped++;
      }
    }
  }
  if (m & CV_ORDER) {
    for (size_t i = 0; i < g_cv_n; i++) g_cv[i].seq = (uint32_t)i;
    qsort(g_cv, g_cv_n, sizeof *g_cv, cv_cmp);
  }
  for (size_t i = 0; i < g_cv_n; i++) {
    const int *a = g_cv[i].a;
    fprintf(trace_fp, "%s %d,%d,%d,%d,%d,%d,%d\n", g_cv[i].op, a[0], a[1], a[2], a[3],
            a[4], a[5], a[6]);
  }
  stat_cv_lines += g_cv_n;
  g_cv_n = 0;
  memset(g_cv_store, 0, sizeof g_cv_store);
}

uint64_t pim_cv_staged(void) { return stat_cv_staged; }
uint64_t pim_cv_lines(void) { return stat_cv_lines; }
uint64_t pim_cv_novel_dropped(void) { return stat_cv_novel_dropped; }
uint64_t pim_cv_residue(void) { return stat_cv_residue; }
uint64_t pim_cv_leader_dropped(void) { return stat_cv_leader_dropped; }
uint64_t pim_cv_tile_empty(void) { return stat_cv_tile_empty; }
uint64_t pim_cv_pos_unmapped(void) { return stat_cv_pos_unmapped; }
uint64_t pim_cv_unfanned(void) { return stat_cv_unfanned; }
uint64_t pim_cv_lane0_idle(void) { return stat_cv_lane0_idle; }
uint64_t pim_cv_lanes_min(void) { return stat_cv_lanes_min < 0 ? 0 : (uint64_t)stat_cv_lanes_min; }
uint64_t pim_cv_lanes_max(void) { return stat_cv_lanes_max < 0 ? 0 : (uint64_t)stat_cv_lanes_max; }

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
  /* Every replay of a layer shares this library, so whatever one run counted must not
   * reach the next. */
  opt_reset();
  ctr_reset();
  cov_reset();
  mac_reset();
  stat_operand_repl_skips = stat_load_coalesced = 0;
  g_opt_native = 0;
  if ((v = getenv("PIM_OPT_NATIVE")) && atoi(v) && (atoi(v) & OPT_EMIT)) {
    g_opt_native = atoi(v);
    if (g_opt_native & ~(OPT_EMIT | NE_BITS)) {
      fprintf(stderr, "[pim-runtime] ERROR: PIM_OPT_NATIVE=%d mixes the native emitter (%d) "
                      "with the replay, a conventions mask or an unknown bit\n",
              g_opt_native, OPT_EMIT);
      exit(1);
    }
    if (__pim_opt_residency_count <= 0) {
      fprintf(stderr, "[pim-runtime] ERROR: the native emitter charges by the compiler's "
                      "residency stamps (im_optipim_roles); this kernel states none\n");
      exit(1);
    }
    if (__pim_opt_hold_count > 0) {
      const int32_t *h = __pim_opt_hold;
      int ok = h[1] > 0 && h[1] < 8 && h[5] > 0;
      /* Bound 1 only. A higher bound delivers each tile more than once, which their memo
       * never does. */
      for (int a = 0; a < 3; a++)
        ok &= (h[1] >> a) & 1 ? h[2 + a] == 1 : h[2 + a] == -1;
      if (!ok) {
        fprintf(stderr, "[pim-runtime] ERROR: the held input (im_optipim_hold_input) states "
                        "held-id mask %d, bounds %d,%d,%d and %d tiles, which is no hold\n", h[1],
                h[2], h[3], h[4], h[5]);
        exit(1);
      }
    }
    if ((g_opt_native & NE_R0) && !(g_opt_native & NE_LEADER_CH)) {
      fprintf(stderr, "[pim-runtime] ERROR: R0 (%d) reads out the channel leader's partials, "
                      "which LEADER_CH (%d) makes the leader\n", NE_R0, NE_LEADER_CH);
      exit(1);
    }
    if ((g_opt_native & NE_SA0) && !(g_opt_native & NE_MAC)) {
      fprintf(stderr, "[pim-runtime] ERROR: SA0 (%d) wraps the arranged weight row, which only "
                      "NE_MAC (%d) places\n", NE_SA0, NE_MAC);
      exit(1);
    }
    if ((g_opt_native & (NE_FAN | NE_CAP16 | NE_LEADER_CH | NE_R0 | NE_BCAST)) &&
        (__pim_opt_lane_ext_count <= 0 || __pim_opt_lane_rep_count <= 0)) {
      fprintf(stderr, "[pim-runtime] ERROR: the native bug bits and BCAST take slices from "
                      "im_optipim_conv's lane statements, which this kernel lacks\n");
      exit(1);
    }
    if ((g_opt_native & NE_MAC) &&
        (__pim_opt_macs_count <= 0 || __pim_opt_mac_geom[0] != cfg_dq_bits / cfg_data_width_bits)) {
      fprintf(stderr, "[pim-runtime] ERROR: NE_MAC (%d) issues the transfers of im-optipim-mac's "
                      "calls; this kernel states %d lanes of MACs of %d values, the word holds "
                      "%d\n", NE_MAC, (int)__pim_opt_macs_count, (int)__pim_opt_mac_geom[0],
              cfg_dq_bits / cfg_data_width_bits);
      exit(1);
    }
  } else if (v && atoi(v)) {
    g_opt_native = atoi(v);
    int m = g_opt_native;
    if ((m & OPT_REPLAY) ? m != OPT_REPLAY
                         : (!(m & OPT_CONVENTIONS) || (m & ~(CV_ALL | OPT_CONVENTIONS)) ||
                            ((m & CV_POS) && !(m & CV_TILE_BR)) ||
                            ((m & CV_S1DROP) && !(m & (CV_TILE_BR | CV_TILE_WR))) ||
                            ((m & CV_R0) && !(m & CV_READOUT)))) {
      fprintf(stderr, "[pim-runtime] ERROR: PIM_OPT_NATIVE=%d is not 1 or a conventions mask "
                      "(2 plus bits, POS needs TILE_BR, S1DROP a TILE bit, R0 READOUT)\n", m);
      exit(1);
    }
    if (__pim_opt_decided != 1 || __pim_opt_cells <= 0) {
      fprintf(stderr, "[pim-runtime] ERROR: PIM_OPT_NATIVE needs the compiler's OptiPIM "
                      "convention (im_optipim_conv); this kernel states none\n");
      exit(1);
    }
  }
  ne_reset(g_opt_native & OPT_EMIT ? g_opt_native : 0);


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
  g_lane_row_top = -1;
  stat_lane_placed = 0;

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
    if (!g_warned_dispatch_overflow && __pim_persistent &&
        tile >= (1ULL << PIM_TILE_BITS)) {
      g_warned_dispatch_overflow = 1;
      fprintf(stderr,
              "[pim-runtime] WARN: tile %llu does not fit the dispatch id's %d tile "
              "bits (epoch %llu). Ids alias and distinct dispatches collapse.\n",
              (unsigned long long)tile, PIM_TILE_BITS, (unsigned long long)epoch);
    }
    /* Only a persistent kernel spends bits on the tile index. A gridded one has
     * none, and shifting its epoch left by 10 overflowed the field at 128
     * instances. The artifact says which shape this is. */
    cur_dispatch = packed;
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
  if (__pim_lanes > 0 &&
      (int)__pim_lanes != cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks)
    fprintf(stderr, "[pim-runtime] WARN kernel compiled for %d lanes, machine has %d "
                    "banks; lane placement puts lane b in bank b and assumes they agree.\n",
            (int)__pim_lanes, cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks);
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
  if (n <= 0) {
    /* The SIMDRAM twin of this silent return was a 64x under-charge (review 2026-09-12);
     * here the tensor merely stays on the interleaved map, but a placement decision made
     * by default rather than by the compiler must still be audible. */
    static int warned = 0;
    if (!warned++)
      fprintf(stderr, "[pim-runtime] WARN kernel carries no layout table; every tensor "
                      "stays on the interleaved map and the host roles stand.\n");
    return;
  }
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
      int all_banks = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
      int vpr = _t->values_per_row > 0 ? _t->values_per_row : 512;
      int share2 = 2 * ((_t->num_elements + all_banks - 1) / all_banks);
      int per_lane = _t->bank_replicated >= 1 ? _t->num_elements
                     : share2 > 65536 ? share2
                     : (_t->num_elements < 65536 ? _t->num_elements : 65536);
      int rows = (per_lane + vpr - 1) / vpr;
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
     * an input to check, and a disagreement fails invariant 6 instead of costing a day. */
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
    stat_layout_from_compiler++;
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
  if (__pim_opt_decided == 1)
    for (int i = 0; i < (int)__pim_opt_role_count && i < PIM_MAX_OPT_ROLES; i++)
      if (__pim_opt_role[2 * i] == tid)
        g_opt_role[tid] = (int)__pim_opt_role[2 * i + 1];
  tensor_info_t *t = &tensors[tid];
  for (int i = 0; i < (int)__pim_opt_lane_rep_count && i < PIM_MAX_OPT_ROLES; i++)
    if (__pim_opt_lane_rep[2 * i] == tid)
      t->opt_lane_rep = (int)__pim_opt_lane_rep[2 * i + 1];
  for (int i = 0; i < (int)__pim_opt_lane_ext_count && i < PIM_MAX_OPT_LANE_AXES; i++) {
    t->opt_lane_dim[i] = (int)__pim_opt_lane_ext[2 * i];
    t->opt_lane_ext[i] = (int)__pim_opt_lane_ext[2 * i + 1];
    t->opt_lane_n = i + 1;
  }
  for (int i = 0; i < (int)__pim_opt_residency_count && i < PIM_MAX_OPT_ROLES; i++)
    if (__pim_opt_residency[2 * i] == tid)
      t->opt_res = (int)__pim_opt_residency[2 * i + 1];
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
  t->opt_lane_rep = -1;
  t->opt_lane_n = 0;
  t->opt_res = -1;
  t->emitted_br = t->emitted_bw = t->emitted_r = t->emitted_w = 0;
  t->dedup_skips = t->range_calls = 0;
  if (t->lane_slot) { slot_map_destroy(t->lane_slot); t->lane_slot = NULL; }

  int total = 1;
  for (int i = 0; i < ndims && i < 4; i++) {
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
    if (cur_phase == PIM_PHASE_COMPUTE) {
      cv_flush();
      ne_flush();
      if (!g_ne_mask) /* ne_finish counts the modelled spill, which no access issued */
        emit_acc_spill();
    }
  }
  cur_phase = phase;
}

/* Arm the modelled spill. RESTORED 2026-09-22: the definition was lost in 36a995c,
 * which deleted three neighbouring levers. emit_acc_spill() and its globals survived,
 * so for two weeks IM_CHARGE_ACC_SPILL=1 charged nothing and an over-capacity tile was
 * free again, which is the exact thing this mechanism exists to prevent. The host has
 * called it the whole time and swallowed the AttributeError. */
void pim_set_acc_spill(int tensor_id, int overflow_per_pe, int k_steps) {
  if (tensor_id < 0 || tensor_id >= num_tensors || overflow_per_pe <= 0 ||
      k_steps <= 0) {
    g_acc_spill_tensor = -1;
    return;
  }
  g_acc_spill_tensor = tensor_id;
  g_acc_spill_overflow = overflow_per_pe;
  g_acc_spill_ksteps = k_steps;
  fprintf(stderr,
          "[pim-runtime] accumulator spill armed: tensor %d, %d values/PE x %d K steps\n",
          tensor_id, overflow_per_pe, k_steps);
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
      g_ctr_outside[3]++;
      g_ctr_outside[1]++;
    }
  }
  fprintf(stderr, "[pim-runtime]   Accumulator spill (MODELLED): %" PRIu64 " records\n",
          stat_acc_spill_emitted);
}

void pim_finalize(void) {
  if (finalized)
    return;

  finalized = 1;

  if (trace_fp && g_opt_native == OPT_REPLAY)
    opt_emit();
  if (g_opt_native & OPT_CONVENTIONS) {
    cv_flush();
    opt_stats(NULL);
  }
  if (g_ne_mask)
    ne_finish();
  if (trace_fp) {
    fclose(trace_fp);
    trace_fp = NULL;
  }
  if (g_opt_native && !g_ne_mask)
    fprintf(stderr, "[pim-runtime] OptiPIM codegen_new: %zu steps, %" PRIu64 " novel, %d "
                    "active banks, %" PRIu64 " commands per tensor per step (C=%d)\n",
            g_opt_n, stat_opt_novel, stat_opt_lanes, stat_opt_cmds, (int)__pim_opt_cells);

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
  if (stat_operand_repl_skips)
    fprintf(stderr,
            "[pim-runtime]   Operand replication skips       : %" PRIu64
            " (compiler said fewer banks receive it)\n",
            stat_operand_repl_skips);
  fprintf(stderr, "[pim-runtime]   Lane-placed accesses            : %" PRIu64 "\n",
          stat_lane_placed);
  /* Slab occupancy per lane: how many values each lane placed in its own bank. Uneven
   * rows are masked or straddling lanes, not a bug; a zero row for an active lane is. */
  for (int i = 0; i < num_tensors; i++) {
    tensor_info_t *t = &tensors[i];
    if (t->bank_replicated < 0 || t->lane_row_base < 0)
      continue;
    int all_banks = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
    fprintf(stderr, "[pim-runtime]   tensor %d slab slots per lane:", i);
    for (int l = 0; l < all_banks && l < MAX_BANKS; l++)
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
          stat_load_coalesced);
  if (stat_coalesce_overflow)
    fprintf(stderr,
            "[pim-runtime]   WARN coalesce buffer overflow    : %" PRIu64
            " (PIM_COALESCE_MAX is a compile-time #define, not an env var)\n",
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
  cov_report();

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
  ctr_add(t, is_write, CC_ISSUED);

  /* Operand register-residency reuse skip retired 2026-09-04: reuse is now the
   * kernel tile, and a broadcast operand collapses to one WB below. ACCUMULATOR
   * (psum) residency is realized in codegen (loop-carried SSA), never a runtime
   * skip. Stores go through the same lockstep collapse as loads; the separate
   * per-program-id store write-once model was removed 2026-09-09. */

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
  /* No operand exemption any more: the evidence above retired it, so every role
     takes the collapse. The `&& !is_operand_load` term that used to sit here was a
     hardcoded 0 and could not fire. */
  if (g_lockstep_enabled && g_lockstep_collapse &&
      cur_phase == PIM_PHASE_COMPUTE) {
    /* Scope the collapse to ONE pseudochannel. An all-bank PIM command reaches the
     * banks of a single (ch,pch); replicas in a different pch or channel ride a
     * separate command bus and need their own event. Omitting them collapsed
     * across independent buses and under-charged (2026-09-05 audit). Bank/bg stay
     * OUT of the key: collapsing those IS the all-bank SIMD model. */
    uint64_t tensor_key = lockstep_ns(t, loc.ch, loc.pch, is_write);
    /* The whole dispatch id is in the key, so two dispatches touching the same physical
     * tuple stay distinct while the 32 bank replays of ONE dispatch still collapse.
     * Replaces resetting the table per program instance. */
    uint64_t key_row = (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    uint64_t h = lockstep_key(tensor_key, cur_dispatch, key_row, (uint64_t)loc.col);
    if (!addr_dedup_check_and_mark(g_lockstep_collapse, h, h >> 16, h >> 48)) {
      stat_lockstep_skips++;
      t->dedup_skips++;
      ctr_add(t, is_write, CC_FOLDED);
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
static void pim_trace_access_emit(uint64_t base_addr, uint64_t size,
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
  {
    uint64_t es = (uint64_t)(t->elem_size > 0 ? t->elem_size : 1);
    cov_mark(tidx, t, base_addr, size / es ? size / es : 1, is_write);
  }

  /* Track program-id boundaries so the dedup table is reset per tile. */
  if (cur_phase == PIM_PHASE_COMPUTE) {
    advance_program_epoch_if_needed();
    if (g_ctr_on)
      ctr_dispatch();
  }
  if (g_opt_native == OPT_REPLAY) {
    if (cur_phase == PIM_PHASE_COMPUTE)
      opt_note_range(t, base_addr, size, is_write);
    return;
  }
  /* The conventions hold a dispatch until the next one starts, so its novelty and its storing
   * banks are complete when it is written. */
  if ((g_opt_native & OPT_CONVENTIONS) && cur_phase == PIM_PHASE_COMPUTE) {
    if (g_opt_n && g_opt_steps[g_opt_n - 1].epoch != __pim_program_epoch)
      cv_flush();
    opt_note_range(t, base_addr, size, is_write);
    int lane = __pim_get_bank_id();
    if (is_write && lane >= 0 && lane < MAX_BANKS)
      g_cv_store[lane] = 1;
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
      /* Key on the PLACED tuple. Keying on the interleaved home dropped every
       * word-mate before it reached place_in_lane_slab, so a partitioned tensor's
       * slab held one slot per word instead of one per element and came out 2-8x
       * too few columns (matmul B read 16 BR where its 16 words per lane owe 32). */
      pim_phys_loc_t pl;
      pl.elem_idx = elem_idx;
      map_element(t, elem_idx, &pl.ch, &pl.pch, &pl.bg, &pl.bank, &pl.sa, &pl.row,
                  &pl.col);
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

static void pim_trace_access_range(uint64_t base_addr, uint64_t size,
                                   int is_write, uint64_t compiler_lanes) {
  if (g_ne_mask) {
    /* The counters number dispatches where the as-emitted path would have. */
    if (g_ctr_on && cur_phase == PIM_PHASE_COMPUTE && find_tensor(base_addr) >= 0)
      ctr_dispatch();
    ne_access(base_addr, size, is_write);
  } else {
    pim_trace_access_emit(base_addr, size, is_write, compiler_lanes);
  }
  if (g_ctr_on && cur_phase == PIM_PHASE_COMPUTE)
    ctr_call(base_addr, size, is_write);
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

