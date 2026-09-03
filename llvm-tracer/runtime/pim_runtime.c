#include "pim_runtime.h"
#include "im_addr_dedup.h"
#include "im_runtime.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
static int cfg_num_sa = 16;
static int cfg_num_rows = 512;
static int cfg_num_cols = 64;
static int cfg_dq_bits = 128;

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
  int layout_kind;        /* pim_layout_kind_t. Only UNSET vs not is read. */
  int reduction_col_axis; /* contraction axis. Recorded, not yet read. */
  int resident_capacity;  /* per-bank register budget. 0 = no cache. */
  /* Reduction-to-column layout (compiler-honored, per-tensor). When both are
   * powers of two > 1, this tensor's contraction axis is transposed onto the
   * column-low address bits so the K-reduction sweeps columns within one open
   * DRAM row instead of activating a new row per K step. Set by
   * pim_set_tensor_redcol() from the compiler's im.residency
   * reduction_to_column decision + the shape-derived (stride=N, extent=K). 0 =
   * disabled (default). */
  int redcol_stride;         /* stride between consecutive contraction steps (=N) */
  int redcol_extent;         /* contraction extent (=K) */

  /* Per-tensor, per-bank register-residency cache honoring the COMPILER's
   * resident_capacity partition of the physical register file (used when
   * g_honor_layout). This REPLACES the single global per-bank LRU that was
   * shared across all tensors: the compiler decides each operand's register
   * budget (resident_capacity) so a low-reuse operand can no longer evict a
   * high-reuse one (the capacity-contention flaw of the shared LRU). NULL until
   * pim_set_tensor_layout allocates it (only for resident_capacity > 0). */
  struct lru_cache *resident_cache[MAX_BANKS];

  /* Per-tensor diagnostic counters (Stage 0). Emitted = trace lines actually
   * written; dedup_skips = times check_and_mark returned 0 for this tensor;
   * range_calls = number of __mem_trace_load/store invocations into this
   * tensor (vs. expanded element accesses).
   *
   * persistent_calls / persistent_skips track loads routed via the persistent
   * scope (program-id-invariant pointer chain — see __pim_load_persistent). */
  uint64_t emitted_br;
  uint64_t emitted_bw;
  uint64_t emitted_r;
  uint64_t emitted_w;
  uint64_t dedup_skips;
  uint64_t range_calls;
  uint64_t persistent_calls;
  uint64_t persistent_skips;
} tensor_info_t;

static tensor_info_t tensors[MAX_TENSORS];
static int num_tensors = 0;

/* ================================================================
 *  Runtime state
 * ================================================================ */
static FILE *trace_fp = NULL;

/* Reduction-col-axis lever. When IM_REDUCTION_COL=1, map_element_interleaved
 * transposes the OPERAND tensor's flat [extent,stride] view to [stride,extent]
 * so the contraction (K) axis lands on the column-low bits — turning the K
 * row-buffer thrash (one ACT per access) into one open row per physical bank.
 * stride/extent are supplied for the proof via IM_REDCOL_STRIDE/EXTENT (=N,K);
 * the production lever will carry these per-tensor from the im.residency
 * classifier. Default OFF => dead branch => trace byte-identical. */
static int g_reduction_col = 0;
static int g_redcol_stride = 0;
static int g_redcol_extent = 0;

/* Bankgroup-interleave lever (PIM_BG_INTERLEAVE=1, default OFF). Places the
 * bankgroup bits BELOW the column bits in the interleaved element->physical map
 * so consecutive elements round-robin the bankgroups. Consecutive column
 * commands then hit DIFFERENT bankgroups (nCCDS spacing) instead of the same
 * one (nCCDL) -> up to ~2x column throughput on the streamed reduction operand.
 * Pure address-bit reorder over power-of-two hardware dims (bijective ->
 * correctness-invariant; only ramulator cycles change). The row bits are the
 * same high bits either way, so row-buffer locality is preserved. */
static int g_bg_interleave = 0;

/* Per-role operand residency gates (ABLATION KNOBS; DEFAULT = hold resident, i.e.
 * let the im.residency classifier's per-tensor decision govern via attr-gating).
 *
 * IMPORTANT (per-lever ablation MEASURED 2026-06-26, leave-one-out cycle ratios
 * toggled/full-config-baseline; register-role map: STREAMED=matmul A / conv INPUT,
 * OPERAND=matmul B / conv WEIGHT). The load-bearing reuse lives almost ENTIRELY on
 * the OPERAND role in BOTH workloads -- a global re-stream flag is the wrong knob:
 *   - matmul: OPERAND(B)-off = 32.7x (== all-reuse-off 32.8x, i.e. ~all of it);
 *             STREAMED(A)-off = 1.06x (A barely reused).
 *   - conv:   OPERAND(WEIGHT)-off = 60x -- the weight is stationary across the
 *             BLOCK_HW output positions of a tile, so re-streaming it costs ~BLOCK_HW
 *             extra reads/value. STREAMED(INPUT)-off = only 1.59x on 16x32x28x28,
 *             and 0.82x (FASTER) on deep-CI 64x64x28x28: holding the low-reuse input
 *             resident EVICTS the high-reuse weight from the 136-int32 GRF (capacity
 *             contention -> motivates per-tensor capacity-aware residency).
 * (Supersedes the earlier audit note that called conv's input/STREAMED reuse "load-
 * bearing/catastrophic" -- measurement shows the WEIGHT/OPERAND dominates by ~40x.)
 * Holding the classifier-marked reused operands resident is the OptiPIM-symmetric,
 * defensible default, so BOTH gates DEFAULT 1; residency fires ONLY for tensors the
 * classifier marked (g_attr_gated_residency). The gates stay as ablation knobs:
 * IM_RESIDENT_STREAMED=0 / IM_RESIDENT_OPERAND=0 re-stream that role to attribute
 * per-operand reuse (NB: =0 on STREAMED regresses small/mid conv in aggregate but
 * HELPS deep-CI conv -- shape-dependent, hence not a global default). A true
 * per-tensor "re-stream weight / hold input" split needs the classifier to label
 * input-vs-weight; deferred. */
static int g_resident_streamed = 1;
static int g_resident_operand = 1;
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
static uint64_t stat_store_coalesced = 0; /* per-call BWs collapsed when a
                                             vector store had multiple lanes
                                             land on the same (bank,row,col) */

/* g_store_write_once_enabled now gates ONLY the accumulator per-pid store-dedup model
 * (g_store_write_once). The blank physical-address per-pid dedup state
 * (formerly g_dedup) was removed 2026-06-22. */
static int g_store_write_once_enabled = 1;

/* Matched-reuse fairness mode. Default = 0 (off). EXPERIMENTAL.
 *
 * Set IM_MATCHED_REUSE=1 to model operand/weight reuse the SAME way
 * OptiPIM does: emit each operand/weight re-read and let the shared
 * Ramulator row-buffer model charge it (cheap on a row-hit, full ACT on
 * a miss) instead of zeroing it at trace level. This makes the
 * per-operator comparison apples-to-apples on the reuse cost model
 * (OptiPIM is already row-buffer-charged).
 *
 * When ON:
 *   - the per-pid (g_dedup) and the four persistent axis-scope dedups are
 *     DISABLED for STREAMED/OPERAND loads (temporal/cross-pid re-reads are
 *     emitted, not collapsed), and the bcast_scalar->persistent unbounded
 *     promotion is suppressed;
 *   - lockstep bank-collapse stays as in default mode: it collapses the 32
 *     STREAMED bank-events at one (sa,row,col) OFFSET into ONE SIMD dispatch.
 *     NOTE (corrected 2026-06-25): this is NOT a data broadcast — the 32 banks
 *     hold DISTINCT bank-spread elements (verified bank0=A[0..3],
 *     bank1=A[256..259]); they share the (sa,row,col) offset and are accessed
 *     in PARALLEL, so collapsing to one dispatch is the free-bank-parallelism
 *     convention, symmetric with OptiPIM single_bank_opt (emit-one-bank+break).
 *     OPERAND loads keep the lockstep BYPASS so per-bank operand work
 *     stays SPREAD across the 32 banks — matching OptiPIM codegen_bank_new,
 *     which fans each input write across all spatial banks (input
 *     replication) rather than serializing on one bank. (An earlier version
 *     collapsed operands onto bank 0, which serialized OptiPIM-parallel work
 *     and inflated cycles — reverted.)
 *   - ACCUMULATOR residency (psum in PE register; in-capacity, fair) and
 *     per-BG broadcast fanout are PRESERVED.
 * Default OFF so no other measurement regresses. */
static int g_matched_mode = 0;

/* (g_bcast_scalar_enabled / broadcast-scalar enable flag removed 2026-06-22.) */
/* Reuse-as-layout honoring (DEFAULT ON as of 2026-07-03). The compiler-decided
 * layout pushed in via pim_set_tensor_layout() drives residency: each tensor is
 * held resident in its own per-bank register partition sized to the compiler's
 * resident_capacity, and the generic shared per-bank LRU is DROPPED (used only as
 * the IM_HONOR_LAYOUT=0 ablation fallback). This makes operand reuse a COMPILER
 * decision the runtime merely enforces, per the layout-via-compiler directive.
 * Set IM_HONOR_LAYOUT=0 to revert to the legacy shared-LRU residency for ablation. */
static int g_honor_layout = 1;

/* Per-pid residency reset (DEFAULT ON as of 2026-07-07). Real hardware has NO
 * cross-program-id operand reuse: GPU registers/shared-memory are strictly
 * intra-thread-block (grid-wide register reuse does not exist; cross-block reuse
 * is only opportunistic L2, which neither Aquabolt-XL nor SIMDRAM has), and the
 * PIM register file is reloaded per SIMD dispatch. Cross-tile operand reuse would
 * require a persistent kernel (one pid iterating many tiles = intra-pid reuse),
 * not automatic cross-pid residency. OptiPIM does not model cross-pid reuse
 * either, so persisting our register residency across program-ids is an
 * unphysical asymmetry in our favor. Therefore reset every tensor's register
 * residency at each program-id boundary, making residency INTRA-pid only.
 * Set IM_CROSS_PID_RESIDENCY=1 to restore the legacy persistent-across-pid cache
 * (ablation / cross-pid-contribution delta measurement only). */
static int g_perpid_residency_reset = 1;

/* (g_acc_resident_enabled + stat_acc_resident_skips removed 2026-06-26: the
 * accumulator-read reuse dedup was proven byte-identical inert — the psum is
 * loop-carried SSA in codegen and never round-trips to DRAM, so there are no
 * ACCUMULATOR loads to collapse. Accumulator residency is realized in compiler
 * codegen, not a runtime reuse skip.) */

/* (g_per_bg_bcast_enabled + g_duplicate_bcast_enabled removed 2026-06-22 with
 * the broadcast / row-duplicate blank-dedup machinery.) */

/* Axis-wise persistent dedup states (g_dedup_persistent + g_dedup_invariant_*)
 * removed 2026-06-22: the 4 __pim_load_persistent* entry points now route
 * pid-invariant operand reuse through the faithful per-bank LRU register cache.
 * stat_persistent_skips is retained — it counts LRU-resident skips on that
 * path. */
static uint64_t stat_persistent_skips = 0;

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
 *   2. OPERAND GRF RESIDENCY  (g_lru_resident / lru_touch)  — the ONE reuse
 *      mechanism. Models the physical PE register file (136 int32/bank); a hit
 *      is register-resident. Compiler-gated (im.residency -> attr-gating). It is
 *      the only thing that removes *reuse* accesses, and it is named lru/resident
 *      (NOT "dedup"). Irreducible without grid-serializing codegen (slower).
 *   3. STORE WRITE-ONCE  (g_store_write_once)  — CORRECTNESS, not reuse. One BW
 *      per output tuple per program-id (psum written back once). Keep.
 *
 * `addr_dedup_*` is a generic hash-set HELPER (im_addr_dedup.c) backing #1 and
 * #3 — that is why the token "dedup" appears throughout the file; it is the data
 * structure's name, not N separate reuse dedups.
 * ============================================================================ */
static addr_dedup_state_t *g_lockstep_collapse = NULL;
static int g_lockstep_enabled = 1;
static uint64_t stat_lockstep_skips = 0;

/* ============================================================
 *  Per-bank LRU residency cache (faithful PE-register model)
 *
 *  Models each PE's register file as a fixed-capacity, fully-
 *  associative LRU cache: an operand re-read that is still resident
 *  is served from the register (NO DRAM event); a re-read of an
 *  evicted operand re-fetches. This replaces the first-N addr_dedup
 *  residency, which skipped the *first N tuples seen* rather than the
 *  N *currently resident* — only an LRU faithfully models "still in a
 *  register" (so a skip provably means register-resident). Fully
 *  associative + LRU is the best-case capacity model: if reuse does
 *  not fit this, it cannot fit a real (set-associative) GRF either.
 *  Enable with IM_LRU_RESIDENCY=1; capacity = g_resident_per_bank.
 * ============================================================ */
#define LRU_EMPTY ((uint64_t)-1)
typedef struct lru_cache {
  int cap;
  uint64_t *keys;
  uint64_t *ts;
  uint64_t clock;
} lru_cache_t;
static lru_cache_t *g_lru_resident[MAX_BANKS];
static int g_lru_residency = 0;
static int g_resident_per_bank = 136;

/* Phase A: attr-gated residency (IM_ATTR_GATED_RESIDENCY=1). When set, a tensor
 * gets the register-residency reuse skip ONLY if the compiler marked it
 * reuse-bearing (t->layout_kind != UNSET, set via pim_set_tensor_layout from the
 * im.residency attrs). Absent a compiler decision (UNSET, e.g. no bridge call),
 * NO reuse skip — every access emits and is faithfully row-buffer-costed. This
 * makes reuse a COMPILER decision the runtime merely applies (and that an
 * ablation lever can switch off), the step toward removing runtime reuse
 * entirely. DEFAULT ON (validated: all 34 matmul/matvec/conv shapes hold under
 * attr-gating — the compiler marks every reuse-bearing operand via the bridge);
 * IM_ATTR_GATED_RESIDENCY=0 reverts to legacy role-based residency. Reuse now
 * REQUIRES the compiler decision: a runtime path that does not call
 * pim_set_tensor_layout (e.g. a hand-written driver) gets no reuse. */
static int g_attr_gated_residency = 0;

static lru_cache_t *lru_create(int cap) {
  if (cap < 1) cap = 1;
  lru_cache_t *c = (lru_cache_t *)calloc(1, sizeof(lru_cache_t));
  c->cap = cap;
  c->keys = (uint64_t *)malloc((size_t)cap * sizeof(uint64_t));
  c->ts = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
  for (int i = 0; i < cap; i++)
    c->keys[i] = LRU_EMPTY;
  c->clock = 0;
  return c;
}
/* Returns 1 if `key` is resident (HIT, caller skips DRAM), 0 if not
 * (MISS: inserted, LRU victim evicted if full; caller emits). */
static int lru_touch(lru_cache_t *c, uint64_t key) {
  c->clock++;
  int empty = -1, victim = 0;
  uint64_t victim_ts = UINT64_MAX;
  for (int i = 0; i < c->cap; i++) {
    if (c->keys[i] == key) {
      c->ts[i] = c->clock;
      return 1; /* HIT */
    }
    if (c->keys[i] == LRU_EMPTY) {
      if (empty < 0)
        empty = i;
    } else if (c->ts[i] < victim_ts) {
      victim_ts = c->ts[i];
      victim = i;
    }
  }
  int slot = (empty >= 0) ? empty : victim;
  c->keys[slot] = key;
  c->ts[slot] = c->clock;
  return 0; /* MISS */
}
static void lru_reset(lru_cache_t *c) {
  if (!c)
    return;
  for (int i = 0; i < c->cap; i++)
    c->keys[i] = LRU_EMPTY;
  c->clock = 0;
}
static void lru_destroy(lru_cache_t *c) {
  if (!c)
    return;
  free(c->keys);
  free(c->ts);
  free(c);
}
static inline uint64_t lru_key(int tensor_id, uint64_t linear_row, int col) {
  return ((uint64_t)(tensor_id & 0x3F) << 58) |
         ((linear_row & 0x3FFFFFFFFFFULL) << 16) | ((uint64_t)col & 0xFFFF);
}

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
static addr_dedup_state_t *g_store_write_once = NULL;
static uint64_t stat_store_write_once_skips = 0;

static int last_program_id = -1;
static int last_program_id_y = -1;
static int last_program_id_z = -1;

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
  if (g_store_write_once) {
    addr_dedup_destroy(g_store_write_once);
    g_store_write_once = NULL;
  }
  if (g_lockstep_collapse) {
    addr_dedup_destroy(g_lockstep_collapse);
    g_lockstep_collapse = NULL;
  }
}

static void advance_program_epoch_if_needed(void) {
  int pid = __pim_get_program_id();
  int pid_y = __pim_get_program_id_y();
  int pid_z = __pim_get_program_id_z();

  int x_changed = (pid != last_program_id);
  int y_changed = (pid_y != last_program_id_y);
  int z_changed = (pid_z != last_program_id_z);

  if (!(x_changed || y_changed || z_changed)) {
    return;
  }

  last_program_id = pid;
  last_program_id_y = pid_y;
  last_program_id_z = pid_z;

  /* g_store_write_once is per-program-id (any axis change resets).
   * g_lockstep_collapse is also per-pid: lockstep replicas of a single SIMD
   * dispatch are all within one pid, so the scope matches. (The blank per-pid
   * g_dedup + the axis-wise persistent states were removed 2026-06-22.) */
  if (g_store_write_once) {
    addr_dedup_reset(g_store_write_once);
  }
  if (g_lockstep_collapse) {
    addr_dedup_reset(g_lockstep_collapse);
  }
  /* Register residency is intra-pid only (see g_perpid_residency_reset): reset
   * every tensor's per-bank register partition (and the legacy shared LRU) at
   * each program-id boundary so no operand carries resident into the next tile.
   * Mirrors the phase-boundary reset in pim_set_phase(). */
  if (g_perpid_residency_reset) {
    int n_banks = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
    if (n_banks > MAX_BANKS)
      n_banks = MAX_BANKS;
    if (g_lru_residency) {
      for (int b = 0; b < n_banks; b++)
        lru_reset(g_lru_resident[b]);
    }
    for (int ti = 0; ti < num_tensors; ti++)
      for (int b = 0; b < n_banks; b++)
        lru_reset(tensors[ti].resident_cache[b]);
  }
  (void)x_changed; (void)y_changed; (void)z_changed;
}

static int compute_global_bank(int ch, int pch, int bg, int bank) {
  int banks_per_pch = cfg_num_bg * cfg_num_banks;
  int banks_per_ch = cfg_num_pch * banks_per_pch;
  return ch * banks_per_ch + pch * banks_per_pch + bg * cfg_num_banks + bank;
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
  /* Reduction-col-axis lever: transpose the contraction axis onto the
   * column-low bits. Applies only to the OPERAND operand (the stride-N
   * reduction matrix) with pow2 stride and extent. This is a pure trace-address
   * remap — it never touches computed data, so correctness is invariant; only
   * ramulator cycles change. Two sources: (a) the compiler-honored PER-TENSOR
   * decision set by pim_set_tensor_redcol() (redcol_stride/extent), active under
   * g_honor_layout; (b) the legacy GLOBAL env proof hook (IM_REDUCTION_COL,
   * default OFF). Per-tensor takes precedence. */
  int rc_stride = 0, rc_extent = 0;
  if (t->role == PIM_ROLE_OPERAND) {
    if (g_honor_layout && t->redcol_stride > 1 && t->redcol_extent > 1) {
      rc_stride = t->redcol_stride;
      rc_extent = t->redcol_extent;
    } else if (g_reduction_col && g_redcol_stride > 1 && g_redcol_extent > 1) {
      rc_stride = g_redcol_stride;
      rc_extent = g_redcol_extent;
    }
  }
  if (rc_stride > 1 && rc_extent > 1 &&
      (rc_stride & (rc_stride - 1)) == 0 &&
      (rc_extent & (rc_extent - 1)) == 0) {
    int S = rc_stride, Kd = rc_extent;
    int k = elem_idx / S;
    int n = elem_idx % S;
    elem_idx = n * Kd + k;
  }
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
      case PIM_ROLE_OPERAND:
        emit_trace("W", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa,
                   loc->row, loc->col);
        stat_writes++;
        t->emitted_w++;
        break;
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

  /* Reduction-col-axis lever (default OFF). */
  if ((v = getenv("IM_REDUCTION_COL")))
    g_reduction_col = atoi(v);
  if ((v = getenv("IM_REDCOL_STRIDE")))
    g_redcol_stride = atoi(v);
  if ((v = getenv("IM_REDCOL_EXTENT")))
    g_redcol_extent = atoi(v);

  /* Bankgroup-interleave lever (default OFF). */
  if ((v = getenv("PIM_BG_INTERLEAVE")))
    g_bg_interleave = atoi(v);

  /* Per-role residency gates (default ON). */
  if ((v = getenv("IM_RESIDENT_STREAMED")))
    g_resident_streamed = atoi(v);
  if ((v = getenv("IM_RESIDENT_OPERAND")))
    g_resident_operand = atoi(v);

  /* Placement scheme: PIM_LAYOUT=interleaved (default) | striped
   * Interleaved is bit-interleaved scheme8-like placement; sequential
   * elements walk (within-DQ → col → bank-within-BG → BG → pch → row),
   * giving small/medium tensors natural bank parallelism and avoiding
   * the single-bank hot-spot of the legacy compact path. Striped is the
   * legacy scheme (compact for small tensors, row-stripe for large)
   * retained for ablation and for non-power-of-2 configs. */
  cfg_layout_scheme = PIM_LAYOUT_INTERLEAVED;
  const char *layout_env = getenv("PIM_LAYOUT");
  if (layout_env && strcmp(layout_env, "striped") == 0) {
    cfg_layout_scheme = PIM_LAYOUT_STRIPED;
  }
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
  last_program_id = -1;
  last_program_id_y = -1;
  last_program_id_z = -1;

  /* Store write-once gate. IM_DEDUP=0 disables it for ablation (env name kept
   * for back-compat; it now gates ONLY the accumulator store write-once model,
   * NOT any reuse dedup — see the access-collapse taxonomy near the globals). */
  const char *dedup_env = getenv("IM_DEDUP");
  g_store_write_once_enabled = (dedup_env && dedup_env[0] == '0') ? 0 : 1;

  /* (PIM_BCAST_SCALAR / broadcast-scalar modeling removed 2026-06-22.) */

  /* Reuse-as-layout honoring knob (default OFF). When set, the compiler's
   * layout decisions (pim_set_tensor_layout) drive physical placement. See
   * g_honor_layout. */
  const char *honor_env = getenv("IM_HONOR_LAYOUT");
  /* DEFAULT ON: compiler layout drives residency. IM_HONOR_LAYOUT=0 reverts to
   * the legacy shared-LRU residency (ablation). */
  g_honor_layout = (honor_env && honor_env[0] == '0') ? 0 : 1;
  fprintf(stderr,
          "[pim-runtime] honor_layout=%d (compiler residency; set "
          "IM_HONOR_LAYOUT=0 for legacy shared-LRU ablation)\n",
          g_honor_layout);

  /* Per-pid residency reset knob. DEFAULT ON: no cross-program-id operand reuse
   * (matches real HW + OptiPIM symmetry, see g_perpid_residency_reset).
   * IM_CROSS_PID_RESIDENCY=1 restores the legacy persistent-across-pid cache. */
  const char *xpid_env = getenv("IM_CROSS_PID_RESIDENCY");
  g_perpid_residency_reset = (xpid_env && xpid_env[0] == '1') ? 0 : 1;
  fprintf(stderr,
          "[pim-runtime] perpid_residency_reset=%d (intra-pid residency only; "
          "set IM_CROSS_PID_RESIDENCY=1 for legacy cross-pid cache)\n",
          g_perpid_residency_reset);

  /* (PIM_PER_BG_BCAST + PIM_DUPLICATE_BCAST / row-duplicate modeling removed
   * 2026-06-22 with the broadcast blank-dedup machinery.) */

  /* (Accumulator-residency runtime knob removed 2026-06-26 — accumulator reuse
   * is realized in codegen via the loop-carried-SSA psum; the runtime dedup was
   * proven inert. PIM_ACC_RESIDENT is no longer read.) */

  /* Lockstep collapse knob (default ON). See g_lockstep_collapse declaration
   * for model. Set PIM_LOCKSTEP_COLLAPSE=0 to disable for ablation. */
  const char *lockstep_env = getenv("PIM_LOCKSTEP_COLLAPSE");
  g_lockstep_enabled = (lockstep_env && lockstep_env[0] == '0') ? 0 : 1;
  stat_lockstep_skips = 0;
  fprintf(stderr,
          "[pim-runtime] lockstep_collapse=%d (set PIM_LOCKSTEP_COLLAPSE=0 to disable)\n",
          g_lockstep_enabled);

  /* Matched-reuse fairness mode (default OFF). See g_matched_mode
   * declaration. Set IM_MATCHED_REUSE=1 to charge operand/weight
   * re-reads via the row-buffer model instead of zeroing them. */
  const char *matched_env = getenv("IM_MATCHED_REUSE");
  g_matched_mode = (matched_env && matched_env[0] == '1') ? 1 : 0;
  fprintf(stderr,
          "[pim-runtime] matched_reuse=%d (set IM_MATCHED_REUSE=1 to enable)\n",
          g_matched_mode);

  /* Dedup-table capacities. Defaults are tuned for ~tens-of-thousands of
   * unique (bank,row,col) tuples per scope, which covers single-channel
   * HBM3-PIM problem sizes. Override via env vars when running larger
   * configurations:
   *   IM_DEDUP_CAP            — capacity for per-program-id load and store
   *                             scopes (default 16K).
   *   IM_PERSISTENT_CAP       — capacity for persistent / axis-wise load
   *                             scopes (default 256K).
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
     * cache. The dedup table keys on global_bank, so an aggregate cap of
     * active_banks × per_bank approximates per-bank residency for the regular
     * (evenly bank-spread) GEMM/conv layouts. Override the per-bank buffer with
     * IM_RESIDENT_PER_BANK, or set the aggregate directly with IM_DEDUP_CAP /
     * IM_PERSISTENT_CAP. See docs/ablation-levers-plan.md. */
    int active_banks =
        cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
    if (active_banks < 1) active_banks = 1;
    /* Per-bank resident operand budget, in elements. A residency hit is served
     * from the PE register file with no DRAM event, so the physical bound is
     * the PE register capacity: HBM-PIM GRF 16x256b = 128 int32 + SRF 8x32b = 8,
     * ~136 int32/bank. The measured per-bank reuse working sets (matvec 8,
     * matmul 24, conv 136) all fit this — i.e. operand reuse here is
     * register-resident, not an unphysical large cache. Override with
     * IM_RESIDENT_PER_BANK. */
    int resident_per_bank = 136; /* PE register file: GRF(128) + SRF(8) int32 */
    if ((v = getenv("IM_RESIDENT_PER_BANK"))) resident_per_bank = atoi(v);
    if (resident_per_bank < 1) resident_per_bank = 1;
    int physical_cap = active_banks * resident_per_bank;

    int perpid_cap = physical_cap;
    int persistent_cap = physical_cap;
    if ((v = getenv("IM_DEDUP_CAP"))      ) perpid_cap = atoi(v);
    if ((v = getenv("IM_PERSISTENT_CAP")) ) persistent_cap = atoi(v);
    if (perpid_cap < 16) perpid_cap = 16;
    if (persistent_cap < 16) persistent_cap = 16;
    fprintf(stderr,
            "[pim-runtime] residency cap: %d/bank x %d banks = %d "
            "(physical row-buffer model; was 256K blank)\n",
            resident_per_bank, active_banks, physical_cap);

    /* Faithful per-bank LRU residency (IM_LRU_RESIDENCY=1): model each PE's
     * register file as a fully-associative LRU cache of g_resident_per_bank
     * entries, instead of the first-N addr_dedup. Persists across program-ids
     * within a COMPUTE phase (reset in pim_set_phase) — same scope as the
     * persistent dedup, so this is an apples-to-apples LRU-vs-first-N swap. */
    g_resident_per_bank = resident_per_bank;
    /* DEFAULT ON: the faithful per-bank LRU register cache is the default
     * residency model — a skip provably means register-resident. Set
     * IM_LRU_RESIDENCY=0 to fall back to the (verified-equivalent, faster)
     * legacy first-N dedup. This is an interim faithful model; the goal is for
     * reuse to be decided by the compiler (im.residency attrs) and ultimately
     * realized in codegen so the runtime needs no reuse skip at all. */
    const char *lru_env = getenv("IM_LRU_RESIDENCY");
    g_lru_residency = (lru_env && lru_env[0] == '0') ? 0 : 1;
    if (g_lru_residency) {
      for (int b = 0; b < active_banks && b < MAX_BANKS; b++)
        g_lru_resident[b] = lru_create(resident_per_bank);
      fprintf(stderr,
              "[pim-runtime] LRU residency ON (default): per-bank register "
              "cache, %d entries/bank (faithful; IM_LRU_RESIDENCY=0 for legacy "
              "first-N)\n",
              resident_per_bank);
    }
    const char *attrgate_env = getenv("IM_ATTR_GATED_RESIDENCY");
    g_attr_gated_residency = (attrgate_env && attrgate_env[0] == '0') ? 0 : 1;
    fprintf(stderr,
            "[pim-runtime] attr-gated residency %s (DEFAULT ON): reuse skip fires "
            "only for tensors the compiler marked resident "
            "(IM_ATTR_GATED_RESIDENCY=0 to disable). Validated: all 34 "
            "matmul/matvec/conv shapes hold.\n",
            g_attr_gated_residency ? "ON" : "OFF");

    // Lockstep collapse is FAITHFUL host-emulation-artifact correction: real
    // HBM-PIM issues one all-bank command, but launcher.py replays the kernel
    // per (pid,bank). It is allocated independently of g_store_write_once_enabled so the
    // blank operand dedups (per-pid physical + persistent scope) can be
    // disabled (IM_DEDUP=0) while lockstep stays on. (Previously g_lockstep_collapse
    // lived inside the g_store_write_once_enabled block, so IM_DEDUP=0 silently killed
    // lockstep too — conflating an artifact correction with the blank dedups.)
    // See docs/ablation-levers-plan.md.
    g_lockstep_collapse = g_lockstep_enabled ? addr_dedup_create(perpid_cap) : NULL;

    /* Only the accumulator per-pid store-dedup state remains (the blank per-pid
     * physical g_dedup + the 4 axis-wise persistent states were removed
     * 2026-06-22). g_store_write_once_enabled now gates only this store model. */
    g_store_write_once = g_store_write_once_enabled ? addr_dedup_create(perpid_cap) : NULL;
  }
  stat_persistent_skips = 0;
  stat_store_write_once_skips = 0;

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
 * Record: [operand_arg, layout_kind, reduction_col_axis, resident_capacity].
 * operand_arg == tensor id, because pointer args are registered first and in
 * signature order.
 *
 * resident_capacity is always 0 from the compiler. The pass cannot see a
 * runtime-valued reduction extent, so the host sends it via
 * pim_set_tensor_capacity(). */
#define PIM_LAYOUT_REC_WORDS 4

/* Weak DEFINITIONS, not weak references. A weak reference does not link on
 * Mach-O when nothing defines the symbol, which is the normal case whenever
 * im-operand-residency-layout is skipped. So count == 0, not a null pointer,
 * is the "no compiler decision" signal.
 * The split decl/def keeps external linkage under C++ and dodges
 * -Wextern-initializer. */
__attribute__((
    weak)) extern const int32_t __pim_layout_table[PIM_LAYOUT_REC_WORDS];
__attribute__((weak)) extern const int32_t __pim_layout_count;

const int32_t __pim_layout_table[PIM_LAYOUT_REC_WORDS] = {0};
const int32_t __pim_layout_count = 0;

static int stat_layout_from_compiler = 0;

/* Honor the compiler's descriptor for one tensor. No-op if the kernel has none.
 */
static void apply_compiler_layout(int tensor_id) {
  int n = (int)__pim_layout_count;
  if (n <= 0)
    return; /* weak fallback in force: no table in this kernel */
  const int32_t *table = __pim_layout_table;
  for (int i = 0; i < n; i++) {
    const int32_t *rec = table + (size_t)i * PIM_LAYOUT_REC_WORDS;
    if (rec[0] != tensor_id)
      continue;
    if (stat_layout_from_compiler == 0)
      fprintf(
          stderr,
          "[pim-runtime] compiler-emitted layout table found in the kernel "
          "artifact (%d entries); host ctypes descriptor push is not used\n",
          n);
    pim_set_tensor_layout(tensor_id, rec[1], rec[2], rec[3]);
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

  /* Compute how many elements fit per column and per row */
  t->values_per_col = cfg_dq_bits / (elem_size * 8);
  // if (t->values_per_col < 1) t->values_per_col = 1;
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

    /* Align only to values_per_col (DQ packing) so tensors don't share
     * a column slot with the previous tensor. Different banks/cols within
     * a shared row are fine. */
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

/* Record the compiler-decided reuse-as-layout descriptor for a tensor.
 * Mirrors pim_set_tensor_broadcast_scalar: bounds-check, store the fields,
 * log. ABI-only at this stage — the fields are recorded but not yet read by
 * map_element or the dedup accounting, so calling this leaves every emitted
 * trace byte-identical. A later increment teaches the mapping/accounting to
 * honor layout_kind / reduction_col_axis / resident_capacity. */
void pim_set_tensor_layout(int tensor_id, int layout_kind,
                           int reduction_col_axis, int resident_capacity) {
  if (tensor_id < 0 || tensor_id >= num_tensors) {
    fprintf(stderr,
            "[pim-runtime] WARN: pim_set_tensor_layout tensor_id=%d "
            "out of range [0,%d); ignored.\n",
            tensor_id, num_tensors);
    return;
  }
  tensor_info_t *t = &tensors[tensor_id];
  t->layout_kind = layout_kind;
  t->reduction_col_axis = reduction_col_axis;
  t->resident_capacity = resident_capacity;

  static const char *kind_names[] = {"UNSET", "RESIDENT", "BANK_SPREAD",
                                     "ROW_DUP", "LEADER"};
  const char *kname =
      (layout_kind >= 0 && layout_kind <= PIM_LAYOUT_KIND_LEADER)
          ? kind_names[layout_kind]
          : "?";
  fprintf(stderr,
          "[pim-runtime] tensor %d: layout_kind=%s reduction_col_axis=%d "
          "resident_capacity=%d (honor=%d)\n",
          tensor_id, kname, reduction_col_axis, resident_capacity,
          g_honor_layout);

  if (!g_honor_layout)
    return; /* ABI-only: record the descriptor, change no emitted trace */

  /* Honor the layout decision by driving the physical-placement machinery.
   * ROW_DUP (the pass's BroadcastReplicate class) maps onto the per-BG resident
   * replication map_element implements (== OptiPIM alloc_method=row_duplicate). */
  switch (layout_kind) {
  case PIM_LAYOUT_KIND_ROW_DUP:
    pim_set_tensor_broadcast_scalar(tensor_id, 1);
    break;
  default:
    break;
  }

  /* COMPILER-PARTITIONED REGISTER RESIDENCY (drops the global shared LRU).
   * The compiler decided this tensor's register budget = resident_capacity
   * (tuples/bank). Allocate this tensor its OWN per-bank residency cache of
   * exactly that capacity; the operand-residency skip sites consult it instead
   * of the global g_lru_resident[] cache when g_honor_layout is set. A tensor
   * the compiler did NOT mark resident (capacity 0) gets NO cache -> every
   * access re-streams (faithful: the compiler chose not to hold it resident).
   * Because each operand has its own compiler-sized partition, a low-reuse
   * operand can no longer evict a high-reuse one — the capacity-contention flaw
   * of the single shared LRU. */
  int n_banks = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
  if (n_banks > MAX_BANKS)
    n_banks = MAX_BANKS;
  for (int b = 0; b < n_banks; b++) {
    if (t->resident_cache[b]) { /* re-registration: rebuild at new capacity */
      lru_destroy(t->resident_cache[b]);
      t->resident_cache[b] = NULL;
    }
    if (resident_capacity > 0)
      t->resident_cache[b] = lru_create(resident_capacity);
  }
}

/* Set only the per-bank register budget, leaving the compiler's structural
 * layout alone. Split from pim_set_tensor_layout so the host can size a
 * residency but not invent one. Budget comes from partition_register_file
 * (run_optipim_tritonim_comparison.py). */
void pim_set_tensor_capacity(int tensor_id, int resident_capacity) {
  if (tensor_id < 0 || tensor_id >= num_tensors) {
    fprintf(stderr,
            "[pim-runtime] WARN: pim_set_tensor_capacity tensor_id=%d out of "
            "range [0,%d); ignored.\n",
            tensor_id, num_tensors);
    return;
  }
  tensor_info_t *t = &tensors[tensor_id];

  /* No compiler layout means no residency, whatever the host says. Without this
   * the host budget silently re-enables reuse when the pass is skipped, which
   * makes that ablation measure nothing. */
  if (t->layout_kind == PIM_LAYOUT_KIND_UNSET) {
    static int warned = 0;
    if (!warned) {
      warned = 1;
      fprintf(stderr,
              "[pim-runtime] host offered resident_capacity=%d for tensor %d "
              "but the compiler marked no layout for it (layout_kind=UNSET); "
              "ignoring -- no compiler decision means no residency\n",
              resident_capacity, tensor_id);
    }
    return;
  }

  if (t->resident_capacity == resident_capacity)
    return; /* nothing to rebuild */
  t->resident_capacity = resident_capacity;
  fprintf(stderr,
          "[pim-runtime] tensor %d: resident_capacity=%d (host-supplied "
          "register budget; layout_kind=%d from the kernel artifact)\n",
          tensor_id, resident_capacity, t->layout_kind);
  if (!g_honor_layout)
    return;
  int n_banks = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
  if (n_banks > MAX_BANKS)
    n_banks = MAX_BANKS;
  for (int b = 0; b < n_banks; b++) {
    if (t->resident_cache[b]) {
      lru_destroy(t->resident_cache[b]);
      t->resident_cache[b] = NULL;
    }
    if (resident_capacity > 0)
      t->resident_cache[b] = lru_create(resident_capacity);
  }
}

/* Compiler-honored reduction-to-column layout for one tensor. Sets the stride
 * (=N, between consecutive contraction steps) and extent (=K, contraction
 * length) that map_element_interleaved uses to transpose the reduction axis
 * onto the column-low bits. Both must be powers of two > 1 to take effect (the
 * remap is guarded); 0 leaves the tensor unremapped. ABI-additive — separate
 * from pim_set_tensor_layout so older callers are unaffected. */
void pim_set_tensor_redcol(int tensor_id, int redcol_stride, int redcol_extent) {
  if (tensor_id < 0 || tensor_id >= num_tensors) {
    fprintf(stderr,
            "[pim-runtime] WARN: pim_set_tensor_redcol tensor_id=%d "
            "out of range [0,%d); ignored.\n",
            tensor_id, num_tensors);
    return;
  }
  tensors[tensor_id].redcol_stride = redcol_stride;
  tensors[tensor_id].redcol_extent = redcol_extent;
  fprintf(stderr,
          "[pim-runtime] tensor %d: reduction-col stride=%d extent=%d "
          "(honor=%d)\n",
          tensor_id, redcol_stride, redcol_extent, g_honor_layout);
}

void pim_set_phase(pim_phase_t phase) {
  const char *names[] = {"IDLE", "COMPUTE", "HOST"};
  if (phase <= PIM_PHASE_HOST) {
    fprintf(stderr, "[pim-runtime] Phase -> %s\n", names[phase]);
  }
  /* Phase boundaries are the natural reset points for every persistent dedup
   * scope: each phase is a distinct logical workload and any cached row
   * activations from one phase must not carry into the next. */
  if (phase != cur_phase) {
    if (g_store_write_once)
      addr_dedup_reset(g_store_write_once);
    /* The per-bank register cache persists across program-ids but is cleared at
     * a phase boundary (a distinct logical workload). Reset BOTH the legacy
     * global shared LRU and every tensor's compiler-partitioned residency. */
    int n_banks = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
    if (n_banks > MAX_BANKS)
      n_banks = MAX_BANKS;
    if (g_lru_residency) {
      for (int b = 0; b < n_banks; b++)
        lru_reset(g_lru_resident[b]);
    }
    for (int ti = 0; ti < num_tensors; ti++)
      for (int b = 0; b < n_banks; b++)
        lru_reset(tensors[ti].resident_cache[b]);
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
          "[pim-runtime]   Persistent (LRU-resident) skips : %" PRIu64 "\n",
          stat_persistent_skips);
  fprintf(stderr,
          "[pim-runtime]   Store dedup skips (per-pid)     : %" PRIu64 "\n",
          stat_store_write_once_skips);
  fprintf(stderr,
          "[pim-runtime]   Lockstep collapse skips         : %" PRIu64 " (lockstep_collapse=%d)\n",
          stat_lockstep_skips, g_lockstep_enabled);
  fprintf(stderr,
          "[pim-runtime]   Store coalesce (intra-vector)   : %" PRIu64 "\n",
          stat_store_coalesced);

  /* Per-tensor breakdown — Stage-0 diagnostics. Useful for figuring out
   * which tensor's accesses dominate the trace and where dedup is biting.
   * Persistent columns track loads routed via __pim_load_persistent
   * (program-id-invariant pointer). */
  fprintf(stderr,
          "\n[pim-runtime] === Per-tensor breakdown ===\n"
          "[pim-runtime]   tid  role         range_calls   BR        BW        "
          "R         W         dedup_skips  pers_calls  pers_skips\n");
  for (int i = 0; i < num_tensors; i++) {
    tensor_info_t *t = &tensors[i];
    const char *role_str = (t->role == PIM_ROLE_STREAMED)  ? "STREAMED"
                           : (t->role == PIM_ROLE_OPERAND) ? "OPERAND "
                                                           : "ACCUMUL.";
    fprintf(stderr,
            "[pim-runtime]   %3d  %-9s    %10" PRIu64 "  %8" PRIu64
            "  %8" PRIu64 "  %8" PRIu64 "  %8" PRIu64 "  %12" PRIu64
            "  %10" PRIu64 "  %10" PRIu64 "\n",
            i, role_str, t->range_calls, t->emitted_br, t->emitted_bw,
            t->emitted_r, t->emitted_w, t->dedup_skips, t->persistent_calls,
            t->persistent_skips);
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

  /* Physical-address dedup: collapse spatially-redundant reads that all
   * resolve to the same row buffer into one trace line per unique
   * (global_bank, sa*rows+row, col) within the current per-program-id scope.
   *
   * Scope:
   *   - Only in COMPUTE phase, and only the OPERAND tensors (STREAMED/OPERAND
   *     roles, gated by the per-role residency knobs + the im.residency
   *     classifier via attr-gating). This is the physical PE-register (GRF)
   *     residency model for reused operands.
   *   - ACCUMULATOR (psum) residency is NOT a runtime skip: it is realized in
   *     compiler codegen (loop-carried SSA — the psum never round-trips to DRAM),
   *     so the old PIM_ACC_RESIDENT dedup was removed 2026-06-26 (proven inert).
   *   - Stores are handled separately (per-program-id store write-once, below). */
  if (cur_phase == PIM_PHASE_COMPUTE && !is_write && g_honor_layout) {
    /* COMPILER-PARTITIONED register residency (the global shared LRU is dropped
     * under g_honor_layout). Consult THIS tensor's own per-bank cache, sized to
     * the compiler's resident_capacity. resident_capacity==0 -> no cache -> the
     * access re-streams (the compiler chose not to hold this operand resident).
     * No role gate and no global LRU: the compiler's per-tensor register budget
     * IS the residency decision, so a low-reuse operand cannot evict a
     * high-reuse one. A hit is register-resident (no DRAM event). */
    int global_bank = compute_global_bank(loc.ch, loc.pch, loc.bg, loc.bank);
    uint64_t linear_row =
        (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    lru_cache_t *cache = (global_bank >= 0 && global_bank < MAX_BANKS)
                             ? t->resident_cache[global_bank]
                             : NULL;
    if (cache && lru_touch(cache, lru_key((int)(t - tensors), linear_row,
                                          (int)loc.col))) {
      t->dedup_skips++;
      return; /* resident in this tensor's compiler-sized register partition */
    }
  } else if (g_lru_residency && cur_phase == PIM_PHASE_COMPUTE && !is_write &&
             (!g_attr_gated_residency ||
              t->layout_kind != PIM_LAYOUT_KIND_UNSET) &&
             ((t->role == PIM_ROLE_STREAMED && g_resident_streamed) ||
              (t->role == PIM_ROLE_OPERAND && g_resident_operand))) {
    /* LEGACY global-shared-LRU residency (ablation fallback, g_honor_layout=0):
     * a single per-bank cache of g_resident_per_bank shared across all tensors.
     * Superseded by the compiler partition above; kept for ablation. */
    int global_bank = compute_global_bank(loc.ch, loc.pch, loc.bg, loc.bank);
    uint64_t linear_row =
        (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    lru_cache_t *cache = (global_bank >= 0 && global_bank < MAX_BANKS)
                             ? g_lru_resident[global_bank]
                             : NULL;
    if (cache && lru_touch(cache, lru_key((int)(t - tensors), linear_row,
                                          (int)loc.col))) {
      t->dedup_skips++;
      return; /* resident in PE register cache; no DRAM event */
    }
  }
  /* (blank reuse dedup removed 2026-06-22: the physical-per-pid g_dedup +
   * bcast_scalar->persistent promotion branch is gone. Faithful operand reuse
   * is the LRU register-residency branch above; cross-bank collapse is the
   * lockstep dedup below; both are the fair mechanisms. The removed branch only
   * fired when LRU was off, so this is behavior-preserving for the faithful
   * config — see docs/phase-b-lowering-plan.md.) */

  /* Per-program-id store dedup (scalar-store path).
   *
   * This is the cross-store layer of the ACCUMULATOR write-once model
   * described above: one logical output tuple should produce at most one BW
   * per program-id. The vector-store path in pim_trace_store_tile_coalesced
   * runs both:
   *   (1) an intra-vector dedup for duplicate lanes within one LLVM store, and
   *   (2) this cross-vector/per-program-id dedup for revisits across store
   *       instructions in the same program-id.
   *
   * Scalar stores have n=1, so there is no intra-vector coalescing step; they
   * consult only this state.
   */
  if (g_store_write_once_enabled && g_store_write_once &&
      cur_phase == PIM_PHASE_COMPUTE && is_write &&
      (!g_attr_gated_residency || t->layout_kind != PIM_LAYOUT_KIND_UNSET) &&
      t->role == PIM_ROLE_ACCUMULATOR) {
    int global_bank = compute_global_bank(loc.ch, loc.pch, loc.bg, loc.bank);
    uint64_t linear_row =
        (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    if (!addr_dedup_check_and_mark(g_store_write_once, (uint64_t)global_bank,
                                   linear_row, (uint64_t)loc.col)) {
      stat_store_write_once_skips++;
      t->dedup_skips++;
      return;
    }
  }

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
  int is_operand_load = (!is_write && t->role == PIM_ROLE_OPERAND);
  if (g_lockstep_enabled && g_lockstep_collapse &&
      cur_phase == PIM_PHASE_COMPUTE && !is_operand_load) {
    uint64_t tensor_key = (uint64_t)(t - tensors);
    uint64_t linear_row =
        (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    if (!addr_dedup_check_and_mark(g_lockstep_collapse, tensor_key,
                                   linear_row, (uint64_t)loc.col)) {
      stat_lockstep_skips++;
      t->dedup_skips++;
      return;
    }
  }

  emit_access_by_role_phase(t, &loc, is_write);

  /* (per-BG broadcast fanout removed 2026-06-22 with the bcast_scalar/
   * row-duplicate machinery — faithful configs never set bcast_scalar.) */
}

/* Persistent-scope counterpart of pim_trace_access_one. Used for loads whose
 * pointer chain is INVARIANT in some subset of program-id axes (the compiler-
 * side analysis in MemTracePass classifies and routes by axis-mask).
 *
 * Parameterized by:
 *   `dedup_state`  - which axis-wise dedup scope to use (the four globals
 *                    g_dedup_persistent / _invariant_yz / _xz / _xy).
 *   `state_skips`  - optional pointer to a global skip counter (NULL to
 *                    skip global accounting). Per-tensor counters are
 *                    always updated.
 */
static void pim_trace_access_one_persistent(tensor_info_t *t, uint64_t addr,
                                            addr_dedup_state_t *dedup_state,
                                            uint64_t *state_skips) {
  pim_phys_loc_t loc;
  if (!resolve_access_location(t, addr, &loc)) {
    return;
  }

  if (cur_phase == PIM_PHASE_COMPUTE && g_honor_layout) {
    /* COMPILER-PARTITIONED residency for pid-invariant operand reuse (e.g.
     * matmul B reused across M-tiles). Same physical register file as the
     * per-pid path (pim_trace_access_one), so route through THIS tensor's own
     * compiler-sized per-bank cache. resident_capacity==0 -> re-stream. */
    int global_bank = compute_global_bank(loc.ch, loc.pch, loc.bg, loc.bank);
    uint64_t linear_row =
        (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    lru_cache_t *cache = (global_bank >= 0 && global_bank < MAX_BANKS)
                             ? t->resident_cache[global_bank]
                             : NULL;
    if (cache && lru_touch(cache, lru_key((int)(t - tensors), linear_row,
                                          (int)loc.col))) {
      t->persistent_skips++;
      stat_persistent_skips++;
      return;
    }
  } else if (cur_phase == PIM_PHASE_COMPUTE && g_lru_residency &&
             (!g_attr_gated_residency ||
              t->layout_kind != PIM_LAYOUT_KIND_UNSET) &&
             ((t->role == PIM_ROLE_STREAMED && g_resident_streamed) ||
              (t->role == PIM_ROLE_OPERAND && g_resident_operand))) {
    /* LEGACY global-shared-LRU path (ablation fallback, g_honor_layout=0): route
     * pid-invariant operand reuse through the SAME shared per-bank cache as the
     * per-pid path — one physical PE register file, first-come reuse. */
    int global_bank = compute_global_bank(loc.ch, loc.pch, loc.bg, loc.bank);
    uint64_t linear_row =
        (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    lru_cache_t *cache = (global_bank >= 0 && global_bank < MAX_BANKS)
                             ? g_lru_resident[global_bank]
                             : NULL;
    if (cache && lru_touch(cache, lru_key((int)(t - tensors), linear_row,
                                          (int)loc.col))) {
      t->persistent_skips++;
      stat_persistent_skips++;
      return;
    }
  }

  /* Lockstep dedup (mirrors pim_trace_access_one). Bypass for OPERAND
   * loads — see access_one for rationale (preserve per-PE register-load
   * cost matching OptiPIM's PimCodeGen abstraction). Persistent-scope
   * loads are always reads here. */
  int is_operand_load_p = (t->role == PIM_ROLE_OPERAND);
  if (g_lockstep_enabled && g_lockstep_collapse &&
      cur_phase == PIM_PHASE_COMPUTE && !is_operand_load_p) {
    uint64_t tensor_key = (uint64_t)(t - tensors);
    uint64_t linear_row =
        (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    if (!addr_dedup_check_and_mark(g_lockstep_collapse, tensor_key,
                                   linear_row, (uint64_t)loc.col)) {
      stat_lockstep_skips++;
      t->dedup_skips++;
      return;
    }
  }

  emit_access_by_role_phase(t, &loc, 0);
  /* (per-BG broadcast fanout removed 2026-06-22 with the bcast machinery.) */
}

/* Vector-store coalescer for ACCUMULATOR drains. Two layers of dedup:
 *
 *   Layer 1 (intra-vector, this function): a single LLVM `store <N x T>`
 *     is one bus-level column-write. With values_per_col > 1, adjacent
 *     lanes of the vector land at the same physical (bank,sa,row,col) and
 *     are written by ONE column-write — emitting one BW per lane would
 *     over-count actual DRAM activity. A small stack-allocated linear-probe
 *     buffer catches these intra-vector duplicates without touching the
 *     global hashmap.
 *
 *   Layer 2 (cross-vector, g_store_write_once): under the Triton
 *     execution model, each program-id is one unit of work and materializes
 *     its outputs once per program-id. Multiple vector stores within the
 *     same program-id that resolve to the same physical tuple correspond
 *     to one logical write to that tuple, so the second emission is
 *     redundant. The per-program-id store dedup state catches these.
 *
 * Example:
 *   Suppose values_per_col=2 and one 4-lane LLVM vector store maps to
 *     lane0 -> (B=5,R=12,C=3)
 *     lane1 -> (B=5,R=12,C=3)
 *     lane2 -> (B=5,R=12,C=4)
 *     lane3 -> (B=5,R=12,C=4)
 *   Layer 1 emits only the first C=3 and first C=4 tuple. If a later vector
 *   store in the same program-id touches (5,12,3) again, Layer 2 suppresses
 *   that revisit so the tuple is still charged once for the whole program-id.
 *
 * The two layers compose: a lane is emitted only if it survives both. The
 * intra-vector layer is a fast path that avoids the global hashmap when
 * the duplicate is local; the per-program-id layer catches the rest.
 *
 * SIMDRAM extension: same shape applies; replace BW emission with the
 * bit-serial expansion (pe_bits W ops per unique col). One helper per
 * runtime, zero compiler-side change. */
#define PIM_STORE_COALESCE_MAX 64

static void pim_trace_store_tile_coalesced(tensor_info_t *t, uint64_t base_addr,
                                           uint64_t n_elements) {
  /* Each entry packs (global_bank, sa, row, col) — same scheme as the
   * global dedup primitive. Stored as uint64 for fast equality. */
  uint64_t seen[PIM_STORE_COALESCE_MAX];
  int n_seen = 0;

  int elem_size = t->elem_size > 0 ? t->elem_size : 1;
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
    int elem_idx = (int)((elem_addr - (uint64_t)t->base_addr) / t->elem_size);
    if (elem_idx < 0 || elem_idx >= t->num_elements) {
      stat_ignored++;
      continue;
    }

    int ch, pch, bg, bank, sa, row, col;
    map_element(t, elem_idx, &ch, &pch, &bg, &bank, &sa, &row, &col);

    int global_bank = compute_global_bank(ch, pch, bg, bank);
    uint64_t key = ((uint64_t)global_bank & 0xFFFFULL) |
                   (((uint64_t)sa & 0xFFULL) << 16) |
                   (((uint64_t)row & 0xFFFFFFULL) << 24) |
                   (((uint64_t)col & 0xFFFFULL) << 48);

    int duplicate = 0;
    for (int j = 0; j < n_seen; j++) {
      if (seen[j] == key) {
        duplicate = 1;
        break;
      }
    }
    if (duplicate) {
      stat_store_coalesced++;
      t->dedup_skips++;
      continue;
    }
    if (n_seen < PIM_STORE_COALESCE_MAX) {
      seen[n_seen++] = key;
    }
    /* Otherwise the per-call buffer is full — fall through to layer 2. */

    /* Layer 2: cross-vector dedup within the current program-id. */
    if (g_store_write_once_enabled && g_store_write_once &&
        (!g_attr_gated_residency || t->layout_kind != PIM_LAYOUT_KIND_UNSET)) {
      uint64_t linear_row =
          (uint64_t)sa * (uint64_t)cfg_num_rows + (uint64_t)row;
      if (!addr_dedup_check_and_mark(g_store_write_once,
                                     (uint64_t)global_bank, linear_row,
                                     (uint64_t)col)) {
        stat_store_write_once_skips++;
        t->dedup_skips++;
        continue;
      }
    }

    /* Layer 3: lockstep collapse. Per (tensor_id, sa, row, col) within a
     * pid, only the first bank's BW emits — the remaining 31 banks are
     * lockstep replicas of the same SIMD store. */
    if (g_lockstep_enabled && g_lockstep_collapse) {
      uint64_t tensor_key = (uint64_t)(t - tensors);
      uint64_t linear_row =
          (uint64_t)sa * (uint64_t)cfg_num_rows + (uint64_t)row;
      if (!addr_dedup_check_and_mark(g_lockstep_collapse, tensor_key,
                                     linear_row, (uint64_t)col)) {
        stat_lockstep_skips++;
        t->dedup_skips++;
        continue;
      }
    }

    emit_trace("BW", ch, pch, bg, bank, sa, row, col);
    stat_bank_writes++;
    t->emitted_bw++;
  }
}

/* Process a vector or scalar access spanning [base_addr, base_addr+size).
 * MemTracePass calls this once per IR-level load/store; we expand to one
 * pim_trace_access_one per logical element so multi-bank/multi-row vector
 * loads are correctly accounted for, and the dedup primitive then collapses
 * intra-tile spatial redundancy. */
static void pim_trace_access_range(uint64_t base_addr, uint64_t size,
                                   int is_write) {
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

  /* Coalesced C-drain: a single vector store to an ACCUMULATOR tensor in
   * COMPUTE phase emits at most one BW per unique (bank, sa, row, col) it
   * touches. See pim_trace_store_tile_coalesced for the full rationale. */
  if (is_write && cur_phase == PIM_PHASE_COMPUTE &&
      t->role == PIM_ROLE_ACCUMULATOR && n_elements > 1) {
    pim_trace_store_tile_coalesced(t, base_addr, n_elements);
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

void __mem_trace_load(void *addr, uint64_t size) {
  if (!trace_fp)
    return;
  pim_trace_access_range((uint64_t)addr, size, 0);
}

void __mem_trace_store(void *addr, uint64_t size) {
  if (!trace_fp)
    return;
  pim_trace_access_range((uint64_t)addr, size, 1);
}

/* Process a load whose pointer is program-id-invariant. The compiler-side
 * analysis in MemTracePass routes such loads here. We do NOT advance the
 * program-id epoch — the persistent dedup scope is meant to span every
 * program-id in the COMPUTE phase. */
static void pim_trace_persistent_range(uint64_t base_addr, uint64_t size,
                                       addr_dedup_state_t *dedup_state,
                                       uint64_t *state_skips) {
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
  t->persistent_calls++;

  int elem_size = t->elem_size > 0 ? t->elem_size : 1;
  uint64_t n_elements = size / (uint64_t)elem_size;
  if (n_elements == 0)
    n_elements = 1;

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
    pim_trace_access_one_persistent(t, elem_addr, dedup_state, state_skips);
  }
}

/* Single persistent load entry point emitted by MemTracePass for any
 * pid-invariant load. The per-axis variants (_yz/_xz/_xy) were consolidated
 * away 2026-06-22 — the axis-wise reuse dedups are gone, so all pid-invariant
 * loads route through the SAME faithful per-bank LRU register cache. */
void __pim_load_persistent(void *addr, uint64_t size) {
  if (!trace_fp)
    return;
  pim_trace_persistent_range((uint64_t)addr, size, NULL, &stat_persistent_skips);
}

