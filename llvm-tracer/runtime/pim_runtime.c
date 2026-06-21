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

  /* Broadcast-scalar hint. When set (=1), every access to this tensor
   * is dedup'd against the xyz-invariant persistent state, modeling a
   * PE-register cache that retains broadcast values for the entire
   * COMPUTE phase. See pim_set_tensor_broadcast_scalar() in the
   * header for the correctness invariant. Default 0. */
  int bcast_scalar;

  /* Per-BG duplication state. When duplicated=1, this tensor has a
   * complete copy in bank=0 of every (channel, pch, bg) triple,
   * starting at row dup_row_base[bg_global_id]. Set at registration
   * time when (bcast_scalar=1 AND g_per_bg_bcast_enabled=1 AND
   * g_duplicate_bcast_enabled=1). map_element routes reads of a
   * duplicated tensor to the host PE's own BG copy.
   *
   * bg_global_id = ch * (num_pch * num_bg) + pch * num_bg + bg.
   * Sized at MAX_BANKS as a generous upper bound — only the first
   * cfg_num_channels * cfg_num_pch * cfg_num_bg entries are used. */
  int duplicated;
  int dup_row_base[MAX_BANKS];

  /* Reuse-as-layout descriptor, set by pim_set_tensor_layout() from the
   * compiler's im-operand-residency-layout decisions. layout_kind=UNSET
   * (=0, the zero-initialized default for the static tensors[] array)
   * preserves every existing mapping and accounting path unchanged. The
   * other fields are recorded now and consumed by a later increment. */
  int layout_kind;           /* pim_layout_kind_t */
  int reduction_col_axis;    /* contraction axis -> column-low bits; -1 = none */
  uint32_t bank_spread_mask; /* bitmask of tensor axes spread across banks */
  int resident_capacity;     /* resident budget in tuples; 0 = unbounded */

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
static pim_phase_t cur_phase = PIM_PHASE_IDLE;
static int initialized = 0;
static int finalized = 0;

/* Per-bank linear row allocation, indexed by flat bank id across all
 * ch/pch/bg/bank */
static int next_free_row[MAX_BANKS];

/* Unified DUP row allocator. All BGs share the same row index for a
 * given bcast_scalar tensor — replicas in different BGs live at the
 * SAME (sa, row, col), differing only in (ch, pch, bg, bank). This is
 * Lever 2: lockstep dedup at (tensor_id, sa, row, col) can then collapse
 * the per-BG fanout for bcast_scalar broadcasts to a single event
 * (one controller dispatch in the hardware model).
 *
 * Allocated downward from the top of bank=0's row space; -1 means
 * uninitialized (first allocation seeds it). */
static int g_next_free_dup_row = -1;

/* Statistics */
static uint64_t stat_bank_reads = 0;
static uint64_t stat_bank_writes = 0;
static uint64_t stat_reads = 0;
static uint64_t stat_writes = 0;
static uint64_t stat_ignored = 0;
/* Number of per-BG broadcast fan-outs emitted with PIM_PER_BG_BCAST=1.
 * Each fan-out adds (cfg_num_channels * cfg_num_pch * cfg_num_bg) - 1
 * extra W events beyond the source-bank emission. Diagnostic only. */
static uint64_t stat_per_bg_bcast_fanouts = 0;
static uint64_t stat_store_coalesced = 0; /* per-call BWs collapsed when a
                                             vector store had multiple lanes
                                             land on the same (bank,row,col) */

/* Physical-address dedup: collapses spatially-redundant reads (multiple PEs
 * in a bank-group hitting the same row buffer) into one trace line per
 * unique (global_bank, sa+row, col). Resets per program-id. See
 * im_addr_dedup.h. */
static addr_dedup_state_t *g_dedup = NULL;
static int g_dedup_enabled = 1;

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
 *   - lockstep bank-collapse stays as in default mode: it collapses STREAMED
 *     bank-replicas (genuine broadcast: all banks read the same (sa,row,col)),
 *     but OPERAND loads keep the lockstep BYPASS so per-bank operand work
 *     stays SPREAD across the 32 banks — matching OptiPIM codegen_bank_new,
 *     which fans each input write across all spatial banks (input
 *     replication) rather than serializing on one bank. (An earlier version
 *     collapsed operands onto bank 0, which serialized OptiPIM-parallel work
 *     and inflated cycles — reverted.)
 *   - ACCUMULATOR residency (psum in PE register; in-capacity, fair) and
 *     per-BG broadcast fanout are PRESERVED.
 * Default OFF so no other measurement regresses. */
static int g_matched_mode = 0;

/* bcast_scalar global enable flag. Default = 1 (on).
 *
 * Gates the per-tensor bcast_scalar bit set by
 * pim_set_tensor_broadcast_scalar(). When ON, tensors flagged from the
 * harness (via the kernel config's `broadcast_scalar_tensors` field)
 * become eligible for:
 *   1. xyz-invariant persistent dedup (the `bcast_scalar && g_dedup_persistent`
 *      branch in pim_trace_access_one) — collapses cross-pid repeats of the
 *      same scalar load to a single emission.
 *   2. Per-BG row-duplicate layout (PIM_DUPLICATE_BCAST=1 also required) —
 *      the tensor is replicated to bank=0 of every BG and each PE reads
 *      its host BG's local copy via BR.
 *
 * When OFF: pim_set_tensor_broadcast_scalar() is a no-op (per-tensor
 * bcast_scalar bit stays 0), neither persistent dedup nor DUP fires, and
 * scalar-broadcast operands go through the standard interleaved layout —
 * 32 PE banks emit 32 distinct BR events per logical broadcast, no
 * cross-pid dedup. Useful for ablation against an OptiPIM PimCodeGen run
 * that does NOT use alloc_method=row_duplicate. */
static int g_bcast_scalar_enabled = 1;
/* Reuse-as-layout honoring (default OFF). When IM_HONOR_LAYOUT=1, the
 * compiler-decided layout pushed in via pim_set_tensor_layout() drives
 * physical placement (currently ROW_DUP -> per-BG resident replication, the
 * same audited path as a manual broadcast_scalar, matching OptiPIM's
 * alloc_method=row_duplicate). Default 0 keeps pim_set_tensor_layout()
 * ABI-only (record + log, no behavior change) so existing traces are
 * byte-identical and A/B comparison is a single env flag. */
static int g_honor_layout = 0;

/* Accumulator residency. Default = 1 (on).
 *
 * Models the OptiPIM "accumulator stays in PE register across the
 * K-loop" pattern. Without this, an inner-product kernel emits one R
 * (load partial sum) + one BW (store partial sum) per K iteration —
 * the BW side is already deduped per-pid by g_dedup_store_perpid (one
 * BW per output tuple per program-id), but the R side is NOT deduped,
 * so a K=64 matmul still emits 64 R events per output tuple.
 *
 * When ON (set PIM_ACC_RESIDENT=0 to disable), ACCUMULATOR loads are
 * deduped at per-pid scope (same scope as the store side, keyed on
 * (global_bank, linear_row, col)). Within one program-id, the first
 * R to a tuple emits; subsequent R's at the same tuple are dropped —
 * modelling "the accumulator value lives in the PE register, no need
 * to re-fetch from the bank." Combined with the existing per-pid BW
 * dedup, this produces 1 R + 1 BW per accumulator tuple per program-
 * id, matching OptiPIM's first_time_in_col + single_bank_opt
 * semantics for accumulator residency. */
static int g_acc_resident_enabled = 1;
static uint64_t stat_acc_resident_skips = 0;

/* Per-bank-group broadcast modeling. Default = 1 (on).
 *
 * Real HBM-PIM SCALAR_LD broadcasts a value on ONE bank-group's I/O
 * bus per command — to reach PEs in N bank-groups, the controller
 * must issue N SCALAR_LD-equivalent commands (one per BG).
 *
 * When ON (default; disable with PIM_PER_BG_BCAST=0), each unique
 * bcast_scalar logical broadcast emits one W event per (channel, pch,
 * bg) triple in the system, modelling the per-BG bus cycle required
 * for that BG's PEs to receive the value. The total event count grows
 * from 1 to cfg_num_channels * cfg_num_pch * cfg_num_bg per unique
 * broadcast.
 *
 * Default ON because the legacy "single W per broadcast" model
 * systematically under-counted cross-BG cycles in multi-BG configs.
 * Set PIM_PER_BG_BCAST=0 to fall back to that legacy single-event
 * mode (useful for ablation against older numbers). Note that when
 * PIM_DUPLICATE_BCAST=1 is also on, the fanout is bypassed for any
 * tensor that successfully duplicated — the per-BG bus cost is then
 * obviated by the per-BG row_duplicate replicas. */
static int g_per_bg_bcast_enabled = 1;

/* Per-BG duplication of bcast_scalar tensors. Default = 1 (on).
 *
 * When ON (default; disable with PIM_DUPLICATE_BCAST=0) AND
 * PIM_PER_BG_BCAST is also on, ANY tensor flagged bcast_scalar at
 * registration time (regardless of role) is REPLICATED into a per-BG
 * local copy at the top of bank=0's row space for each (channel, pch,
 * bg) triple. All BGs share the SAME row index (alloc_dup_rows_shared),
 * so per-BG copies live at identical (sa, row, col) — enabling
 * downstream lockstep dedup to collapse the per-BG fanout to one event.
 * No setup-cost W events are emitted (parity with OptiPIM's
 * alloc_method=row_duplicate, which leaves replication implicit in the
 * layout rather than the trace).
 *
 * At runtime, each PE reads from its host BG's bank=0 copy via BR
 * (bank-local within the BG, no cross-BG fanout), sidestepping the
 * per-BG broadcast cost paid by PIM_PER_BG_BCAST alone.
 *
 * Eligibility: any bcast_scalar tensor, and only with PIM_PER_BG_BCAST
 * also on (without per-BG cost modelling there's nothing to amortise
 * away). If the dup pool is exhausted at registration, t->duplicated
 * stays 0 and the runtime falls back to per-BG W fanout for that tensor.
 *
 * Matches OptiPIM's alloc_method=row_duplicate, which physically
 * replicates Weight/operand tensors across banks. */
static int g_duplicate_bcast_enabled = 1;

/* Axis-wise persistent dedup states.
 *
 * Each state corresponds to a class of loads whose pointer is INVARIANT in
 * a particular subset of program-id axes (and therefore depends only on the
 * complementary axes — or nothing at all):
 *
 *   g_dedup_persistent     — invariant in {x,y,z}: reset only on phase change
 *   g_dedup_invariant_yz   — invariant in {y,z}, depends on x: reset on pid_x
 * change g_dedup_invariant_xz   — invariant in {x,z}, depends on y: reset on
 * pid_y change g_dedup_invariant_xy   — invariant in {x,y}, depends on z: reset
 * on pid_z change
 *
 * MemTracePass classifies each load's pointer chain and routes it to the
 * matching __pim_load_persistent[_axes] entry point. Each entry point
 * pushes its access through the corresponding state.
 *
 * Reset cadence is enforced in advance_program_epoch_if_needed by tracking
 * which axis changed between successive runtime calls. */
static addr_dedup_state_t *g_dedup_persistent = NULL;   /* invariant in xyz */
static addr_dedup_state_t *g_dedup_invariant_yz = NULL; /* depends on x only */
static addr_dedup_state_t *g_dedup_invariant_xz = NULL; /* depends on y only */
static addr_dedup_state_t *g_dedup_invariant_xy = NULL; /* depends on z only */
static uint64_t stat_persistent_skips = 0;
static uint64_t stat_invariant_yz_skips = 0;
static uint64_t stat_invariant_xz_skips = 0;
static uint64_t stat_invariant_xy_skips = 0;

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
static addr_dedup_state_t *g_dedup_lockstep = NULL;
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
typedef struct {
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
static addr_dedup_state_t *g_dedup_store_perpid = NULL;
static uint64_t stat_store_perpid_skips = 0;

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
  if (g_dedup) {
    addr_dedup_destroy(g_dedup);
    g_dedup = NULL;
  }
  if (g_dedup_persistent) {
    addr_dedup_destroy(g_dedup_persistent);
    g_dedup_persistent = NULL;
  }
  if (g_dedup_invariant_yz) {
    addr_dedup_destroy(g_dedup_invariant_yz);
    g_dedup_invariant_yz = NULL;
  }
  if (g_dedup_invariant_xz) {
    addr_dedup_destroy(g_dedup_invariant_xz);
    g_dedup_invariant_xz = NULL;
  }
  if (g_dedup_invariant_xy) {
    addr_dedup_destroy(g_dedup_invariant_xy);
    g_dedup_invariant_xy = NULL;
  }
  if (g_dedup_store_perpid) {
    addr_dedup_destroy(g_dedup_store_perpid);
    g_dedup_store_perpid = NULL;
  }
  if (g_dedup_lockstep) {
    addr_dedup_destroy(g_dedup_lockstep);
    g_dedup_lockstep = NULL;
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

  /* g_dedup and g_dedup_store_perpid are per-program-id (any axis change
   * resets). g_dedup_lockstep is also per-pid: lockstep replicas of a
   * single SIMD dispatch are all within one pid, so the scope matches. */
  if (g_dedup) {
    addr_dedup_reset(g_dedup);
  }
  if (g_dedup_store_perpid) {
    addr_dedup_reset(g_dedup_store_perpid);
  }
  if (g_dedup_lockstep) {
    addr_dedup_reset(g_dedup_lockstep);
  }

  /* Axis-wise states: reset only when an axis the load depends on changes.
   *
   *   g_dedup_invariant_yz holds (bank,row,col) for loads that depend on
   *   pid_x only. They're invariant across pid_y/pid_z, so we keep the state
   *   alive when only y or z changes; we reset when x changes.
   *
   *   g_dedup_invariant_xz / xy follow the symmetric rule.
   */
  if (x_changed && g_dedup_invariant_yz) {
    addr_dedup_reset(g_dedup_invariant_yz);
  }
  if (y_changed && g_dedup_invariant_xz) {
    addr_dedup_reset(g_dedup_invariant_xz);
  }
  if (z_changed && g_dedup_invariant_xy) {
    addr_dedup_reset(g_dedup_invariant_xy);
  }
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

/*
 * Map a tensor element index to physical HBM coordinates.
 * Dispatches between the legacy striped scheme and the bit-interleaved
 * scheme based on the per-tensor layout_scheme field.
 */
static void map_element(const tensor_info_t *t, int elem_idx, int *ch, int *pch,
                        int *bg, int *bank, int *sa, int *row, int *col) {
  /* Per-BG duplicated tensor (PIM_DUPLICATE_BCAST=1): route to the host
   * PE's own BG copy at bank=0. Each BG has a complete replica from the
   * top of bank=0's row space (allocated by duplicate_bcast_tensor_per_bg).
   * The read is bank-local within the host PE's bank-group — no cross-BG
   * bus broadcast needed at runtime. The replication is treated as
   * pre-existing layout (zero setup W events emitted), matching OptiPIM's
   * alloc_method=row_duplicate semantics. */
  if (t->duplicated) {
    int host_bank_global = (int)__pim_get_bank_id();
    int total_flat = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
    if (host_bank_global < 0 || host_bank_global >= total_flat) {
      host_bank_global = 0;
    }
    int host_bg_global = host_bank_global / cfg_num_banks;
    int bg_per_pch = cfg_num_bg;
    int bg_per_ch  = cfg_num_pch * cfg_num_bg;
    *ch   = host_bg_global / bg_per_ch;
    *pch  = (host_bg_global / bg_per_pch) % cfg_num_pch;
    *bg   = host_bg_global % bg_per_pch;
    *bank = 0;
    int linear_row = t->dup_row_base[host_bg_global]
                   + (elem_idx / t->values_per_row);
    *sa  = linear_row / cfg_num_rows;
    *row = linear_row % cfg_num_rows;
    *col = (elem_idx % t->values_per_row) / t->values_per_col;
    return;
  }

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
      /* Duplicated bcast_scalar override: a per-BG replica lives in the
       * host PE's own bank=0, so the access is bank-local regardless of
       * the compiler-classified role. Emit BR; the cross-BG bus cost was
       * skipped at setup (parity with OptiPIM row_duplicate). */
      if (t->duplicated && t->bcast_scalar) {
        emit_trace("BR", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa,
                   loc->row, loc->col);
        stat_bank_reads++;
        t->emitted_br++;
        return;
      }
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

/* Per-BG broadcast fanout. Called once per UNIQUE bcast_scalar logical
 * broadcast (after the persistent-dedup check has already counted the
 * source emission via the caller's normal emit path). Emits one
 * additional W event per remaining (channel, pch, bg) triple, modelling
 * the per-BG I/O-bus cycle required for each receiving bank-group's
 * PEs to latch the broadcast value into their registers.
 *
 * The source BG is identified by (loc->ch, loc->pch, loc->bg) and
 * skipped — the caller already emitted an event (BR or W per the
 * tensor role) there. The fanout always emits W, regardless of the
 * source-tensor role, because cross-BG transfer is bus-mediated (a
 * priority-read at the source, write-into-register at each receiving
 * PE) — modelled as one W per receiving BG. The "representative bank"
 * is bank 0 (deterministic, distributes load across BG controllers in
 * Ramulator2's scheduler). row/col stay at the source's row/col. */
static void emit_per_bg_bcast_fanout(tensor_info_t *t,
                                     const pim_phys_loc_t *src) {
  if (!g_per_bg_bcast_enabled)
    return;
  if (cur_phase != PIM_PHASE_COMPUTE)
    return;
  for (int chan = 0; chan < cfg_num_channels; chan++) {
    for (int pch = 0; pch < cfg_num_pch; pch++) {
      for (int bg = 0; bg < cfg_num_bg; bg++) {
        /* Skip source BG (already emitted by the caller). */
        if (chan == src->ch && pch == src->pch && bg == src->bg)
          continue;
        emit_trace("W", chan, pch, bg, /*bank=*/0, src->sa, src->row, src->col);
        stat_writes++;
        t->emitted_w++;
        stat_per_bg_bcast_fanouts++;
      }
    }
  }
}

/* Allocate `n_rows` contiguous rows from the top of bank=0's row space.
 * Single shared pool — same row index applies to ALL bank-groups so the
 * per-BG replicas live at identical (sa, row, col), enabling lockstep
 * collapse downstream. Grows downward (newest allocation at the highest
 * row indices first). Returns the row index of the FIRST row, -1 on
 * failure (pool exhausted). */
static int alloc_dup_rows_shared(int n_rows) {
  if (g_next_free_dup_row < 0) {
    g_next_free_dup_row = cfg_num_sa * cfg_num_rows - 1;
  }
  int top = g_next_free_dup_row;
  int first = top - n_rows + 1;
  if (first < 0) {
    fprintf(stderr,
            "[pim-runtime] ERROR: ran out of rows for shared dup allocation "
            "(need %d rows; top=%d)\n",
            n_rows, top);
    return -1;
  }
  g_next_free_dup_row = first - 1;
  return first;
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
  stat_per_bg_bcast_fanouts = 0;
  stat_bank_reads = stat_bank_writes = stat_reads = stat_writes = stat_ignored =
      0;
  last_program_id = -1;
  last_program_id_y = -1;
  last_program_id_z = -1;

  /* Allow disabling the physical-address dedup for ablation: IM_DEDUP=0. */
  const char *dedup_env = getenv("IM_DEDUP");
  g_dedup_enabled = (dedup_env && dedup_env[0] == '0') ? 0 : 1;

  /* bcast_scalar ablation knob: PIM_BCAST_SCALAR=0 disables the
   * SCALAR_LD-broadcast modeling, forcing per-bank emission for
   * what would otherwise be xyz-invariant scalar-broadcast loads.
   * Apples-to-apples vs OptiPIM's PimCodeGen which doesn't model
   * SCALAR_LD. Default = 1 (enabled). */
  const char *bcast_env = getenv("PIM_BCAST_SCALAR");
  g_bcast_scalar_enabled = (bcast_env && bcast_env[0] == '0') ? 0 : 1;
  fprintf(stderr, "[pim-runtime] bcast_scalar=%d (set PIM_BCAST_SCALAR=0 to disable)\n",
          g_bcast_scalar_enabled);

  /* Reuse-as-layout honoring knob (default OFF). When set, the compiler's
   * layout decisions (pim_set_tensor_layout) drive physical placement. See
   * g_honor_layout. */
  const char *honor_env = getenv("IM_HONOR_LAYOUT");
  g_honor_layout = (honor_env && honor_env[0] == '1') ? 1 : 0;
  fprintf(stderr,
          "[pim-runtime] honor_layout=%d (set IM_HONOR_LAYOUT=1 to let the "
          "compiler layout drive placement)\n",
          g_honor_layout);

  /* Per-BG broadcast modeling knob (default ON). Emits
   * cfg_num_channels * cfg_num_pch * cfg_num_bg W events per unique
   * scalar broadcast (one per bank-group's I/O bus cycle). Set
   * PIM_PER_BG_BCAST=0 to revert to the legacy single-event-per-
   * broadcast mode. See g_per_bg_bcast_enabled declaration. */
  const char *per_bg_env = getenv("PIM_PER_BG_BCAST");
  g_per_bg_bcast_enabled = (per_bg_env && per_bg_env[0] == '0') ? 0 : 1;
  fprintf(stderr,
          "[pim-runtime] per_bg_bcast=%d (set PIM_PER_BG_BCAST=0 to disable cross-BG broadcast cost modeling)\n",
          g_per_bg_bcast_enabled);

  /* Per-BG bcast_scalar duplication knob (default ON). Only effective
   * when PIM_PER_BG_BCAST is also on — without the per-BG cost model
   * there's nothing to amortise away. Set PIM_DUPLICATE_BCAST=0 to
   * disable. */
  const char *dup_env = getenv("PIM_DUPLICATE_BCAST");
  g_duplicate_bcast_enabled = (dup_env && dup_env[0] == '0') ? 0 : 1;
  g_next_free_dup_row = -1;
  fprintf(stderr,
          "[pim-runtime] duplicate_bcast=%d (set PIM_DUPLICATE_BCAST=0 to disable; requires per_bg_bcast=1 to take effect)\n",
          g_duplicate_bcast_enabled);

  /* Accumulator residency knob (default ON). Dedups ACCUMULATOR loads
   * at per-pid scope so K-loop partial-sum reads collapse to 1 R per
   * output tuple per program-id. Set PIM_ACC_RESIDENT=0 to disable. */
  const char *acc_env = getenv("PIM_ACC_RESIDENT");
  g_acc_resident_enabled = (acc_env && acc_env[0] == '0') ? 0 : 1;
  stat_acc_resident_skips = 0;
  fprintf(stderr,
          "[pim-runtime] acc_resident=%d (set PIM_ACC_RESIDENT=0 to disable)\n",
          g_acc_resident_enabled);

  /* Lockstep collapse knob (default ON). See g_dedup_lockstep declaration
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
    // per (pid,bank). It is allocated independently of g_dedup_enabled so the
    // blank operand dedups (per-pid physical + persistent scope) can be
    // disabled (IM_DEDUP=0) while lockstep stays on. (Previously g_dedup_lockstep
    // lived inside the g_dedup_enabled block, so IM_DEDUP=0 silently killed
    // lockstep too — conflating an artifact correction with the blank dedups.)
    // See docs/ablation-levers-plan.md.
    g_dedup_lockstep = g_lockstep_enabled ? addr_dedup_create(perpid_cap) : NULL;

    if (g_dedup_enabled) {
      g_dedup               = addr_dedup_create(perpid_cap);
      g_dedup_persistent    = addr_dedup_create(persistent_cap);
      g_dedup_invariant_yz  = addr_dedup_create(persistent_cap);
      g_dedup_invariant_xz  = addr_dedup_create(persistent_cap);
      g_dedup_invariant_xy  = addr_dedup_create(persistent_cap);
      g_dedup_store_perpid  = addr_dedup_create(perpid_cap);
    } else {
      g_dedup = NULL;
      g_dedup_persistent = NULL;
      g_dedup_invariant_yz = NULL;
      g_dedup_invariant_xz = NULL;
      g_dedup_invariant_xy = NULL;
      g_dedup_store_perpid = NULL;
    }
  }
  stat_persistent_skips = 0;
  stat_invariant_yz_skips = 0;
  stat_invariant_xz_skips = 0;
  stat_invariant_xy_skips = 0;
  stat_store_perpid_skips = 0;

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
  t->bcast_scalar = 0;
  t->layout_scheme = cfg_layout_scheme;
  t->layout_linear_base = 0;
  t->duplicated = 0;
  for (int i = 0; i < MAX_BANKS; i++) t->dup_row_base[i] = 0;

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
    return num_tensors++;
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
    return num_tensors++;
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

  return num_tensors++;
}

/* Per-BG bcast_scalar duplication: replicate `t` into one copy in
 * bank=0 of every (ch, pch, bg). All BGs share the same row index
 * (alloc_dup_rows_shared), so the per-BG copies live at identical
 * (sa, row, col) — this lets the lockstep dedup collapse the per-BG
 * fan-out to one event downstream. The replication itself is treated
 * as a pre-kernel layout property (host arranges the data before
 * trace start) — NO setup W events are emitted. Matches OptiPIM's
 * `alloc_method=row_duplicate` semantics.
 *
 * Returns 0 on success, -1 if the shared dup pool is exhausted (caller
 * leaves t->duplicated=0 and the runtime falls back to per-BG W fanout). */
static int duplicate_bcast_tensor_per_bg(tensor_info_t *t) {
  int rows_needed = (t->num_elements + t->values_per_row - 1)
                    / t->values_per_row;
  /* Single shared allocation — all BGs use the SAME row base. This places
   * each BG's replica at identical (sa, row, col), enabling the lockstep
   * dedup downstream to collapse 8 per-BG events to one. */
  int shared_base = alloc_dup_rows_shared(rows_needed);
  if (shared_base < 0) return -1;
  int n_bg_total = cfg_num_channels * cfg_num_pch * cfg_num_bg;
  for (int bg_global = 0; bg_global < n_bg_total; bg_global++) {
    t->dup_row_base[bg_global] = shared_base;
  }
  t->duplicated = 1;
  fprintf(stderr,
          "[pim-runtime] per-BG-dup layout: %d BGs reserved %d rows each "
          "(pre-stored, no setup W emitted)\n",
          n_bg_total, rows_needed);
  return 0;
}

void pim_set_tensor_broadcast_scalar(int tensor_id, int on) {
  if (tensor_id < 0 || tensor_id >= num_tensors) {
    fprintf(stderr,
            "[pim-runtime] WARN: pim_set_tensor_broadcast_scalar tensor_id=%d "
            "out of range [0,%d); ignored.\n",
            tensor_id, num_tensors);
    return;
  }
  /* Honor the global PIM_BCAST_SCALAR ablation flag: when disabled,
   * this setter is a no-op (bcast_scalar stays 0) regardless of `on`. */
  tensors[tensor_id].bcast_scalar = (on && g_bcast_scalar_enabled) ? 1 : 0;
  fprintf(stderr,
          "[pim-runtime] tensor %d: bcast_scalar=%d (requested=%d, global_enabled=%d)\n",
          tensor_id, tensors[tensor_id].bcast_scalar, on, g_bcast_scalar_enabled);
  /* Per-BG duplication for bcast_scalar tensors. Eligibility:
   *   - Both flags on (g_per_bg_bcast_enabled + g_duplicate_bcast_enabled).
   *   - Tensor is bcast_scalar (set above).
   *
   * Applies regardless of role (no STREAMED/OPERAND gate). Once
   * duplicated, every access is routed to the host PE's own BG bank=0
   * by map_element, and emit_access_by_role_phase emits BR (bank-local
   * read). The cross-BG bus cost that would otherwise justify W for
   * OPERAND is gone because the data is resident in the receiving BG. */
  if (tensors[tensor_id].bcast_scalar &&
      g_per_bg_bcast_enabled && g_duplicate_bcast_enabled &&
      !tensors[tensor_id].duplicated) {
    duplicate_bcast_tensor_per_bg(&tensors[tensor_id]);
  }
}

/* Record the compiler-decided reuse-as-layout descriptor for a tensor.
 * Mirrors pim_set_tensor_broadcast_scalar: bounds-check, store the fields,
 * log. ABI-only at this stage — the fields are recorded but not yet read by
 * map_element or the dedup accounting, so calling this leaves every emitted
 * trace byte-identical. A later increment teaches the mapping/accounting to
 * honor layout_kind / reduction_col_axis / bank_spread_mask /
 * resident_capacity. */
void pim_set_tensor_layout(int tensor_id, int layout_kind,
                           int reduction_col_axis, uint32_t bank_spread_mask,
                           int resident_capacity) {
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
  t->bank_spread_mask = bank_spread_mask;
  t->resident_capacity = resident_capacity;

  static const char *kind_names[] = {"UNSET", "RESIDENT", "BANK_SPREAD",
                                     "ROW_DUP", "LEADER"};
  const char *kname =
      (layout_kind >= 0 && layout_kind <= PIM_LAYOUT_KIND_LEADER)
          ? kind_names[layout_kind]
          : "?";
  fprintf(stderr,
          "[pim-runtime] tensor %d: layout_kind=%s reduction_col_axis=%d "
          "bank_spread_mask=0x%x resident_capacity=%d (honor=%d)\n",
          tensor_id, kname, reduction_col_axis, bank_spread_mask,
          resident_capacity, g_honor_layout);

  if (!g_honor_layout)
    return; /* ABI-only: record the descriptor, change no emitted trace */

  /* Honor the layout decision by driving the existing physical-placement
   * machinery. ROW_DUP (the pass's BroadcastReplicate class) maps onto the
   * per-BG resident replication map_element already implements, which matches
   * OptiPIM's alloc_method=row_duplicate — the FAIR layout for a scalar-
   * broadcast operand and the compiler-decided replacement for a manual
   * broadcast_scalar annotation. BANK_SPREAD is the default placement (no
   * change); RESIDENT/LEADER and reduction_col_axis/resident_capacity
   * honoring land in a later increment. */
  switch (layout_kind) {
  case PIM_LAYOUT_KIND_ROW_DUP:
    pim_set_tensor_broadcast_scalar(tensor_id, 1);
    break;
  default:
    break;
  }
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
    if (g_dedup_persistent)
      addr_dedup_reset(g_dedup_persistent);
    if (g_dedup_invariant_yz)
      addr_dedup_reset(g_dedup_invariant_yz);
    if (g_dedup_invariant_xz)
      addr_dedup_reset(g_dedup_invariant_xz);
    if (g_dedup_invariant_xy)
      addr_dedup_reset(g_dedup_invariant_xy);
    if (g_dedup_store_perpid)
      addr_dedup_reset(g_dedup_store_perpid);
    /* The per-bank register cache persists across program-ids but is cleared at
     * a phase boundary (a distinct logical workload). */
    if (g_lru_residency) {
      int n_banks = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
      for (int b = 0; b < n_banks && b < MAX_BANKS; b++)
        lru_reset(g_lru_resident[b]);
    }
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
  fprintf(stderr, "[pim-runtime]   Dedup hits     : %" PRIu64 " (enabled=%d)\n",
          addr_dedup_hits(g_dedup), g_dedup_enabled);
  fprintf(stderr,
          "[pim-runtime]   Persistent skips (xyz invariant): %" PRIu64 "\n",
          stat_persistent_skips);
  fprintf(stderr,
          "[pim-runtime]   Persistent skips (yz invariant) : %" PRIu64 "\n",
          stat_invariant_yz_skips);
  fprintf(stderr,
          "[pim-runtime]   Persistent skips (xz invariant) : %" PRIu64 "\n",
          stat_invariant_xz_skips);
  fprintf(stderr,
          "[pim-runtime]   Persistent skips (xy invariant) : %" PRIu64 "\n",
          stat_invariant_xy_skips);
  fprintf(stderr,
          "[pim-runtime]   Store dedup skips (per-pid)     : %" PRIu64 "\n",
          stat_store_perpid_skips);
  fprintf(stderr,
          "[pim-runtime]   Acc residency skips (R per-pid) : %" PRIu64 " (acc_resident=%d)\n",
          stat_acc_resident_skips, g_acc_resident_enabled);
  fprintf(stderr,
          "[pim-runtime]   Lockstep collapse skips         : %" PRIu64 " (lockstep_collapse=%d)\n",
          stat_lockstep_skips, g_lockstep_enabled);
  fprintf(stderr,
          "[pim-runtime]   Store coalesce (intra-vector)   : %" PRIu64 "\n",
          stat_store_coalesced);
  fprintf(stderr,
          "[pim-runtime]   Per-BG bcast fanouts (extra W)  : %" PRIu64 " (per_bg_bcast=%d)\n",
          stat_per_bg_bcast_fanouts, g_per_bg_bcast_enabled);

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
   *   - Only in COMPUTE phase.
   *   - STREAMED/OPERAND loads: always.
   *   - ACCUMULATOR loads: only when PIM_ACC_RESIDENT is on. Models
   *     the partial sum staying in the PE register across K iterations,
   *     so the first R to (bank, row, col) per program-id emits and
   *     subsequent reads at the same tuple are dropped. This pairs
   *     with the existing per-pid store dedup to produce 1 R + 1 BW
   *     per accumulator tuple per pid (matching OptiPIM).
   *   - Stores are handled separately (per-program-id store dedup, below). */
  if (g_lru_residency && cur_phase == PIM_PHASE_COMPUTE && !is_write &&
      (!g_attr_gated_residency || t->layout_kind != PIM_LAYOUT_KIND_UNSET) &&
      (t->role == PIM_ROLE_STREAMED || t->role == PIM_ROLE_OPERAND ||
       (t->role == PIM_ROLE_ACCUMULATOR && g_acc_resident_enabled))) {
    /* Faithful PE-register residency: a hit is served from the register
     * (no DRAM event); a miss re-fetches (LRU eviction at GRF capacity).
     * Unifies the per-pid / persistent / bcast scopes into one physical
     * per-bank register cache. */
    int global_bank = compute_global_bank(loc.ch, loc.pch, loc.bg, loc.bank);
    uint64_t linear_row =
        (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    lru_cache_t *cache = (global_bank >= 0 && global_bank < MAX_BANKS)
                             ? g_lru_resident[global_bank]
                             : NULL;
    if (cache && lru_touch(cache, lru_key((int)(t - tensors), linear_row,
                                          (int)loc.col))) {
      t->dedup_skips++;
      if (t->role == PIM_ROLE_ACCUMULATOR)
        stat_acc_resident_skips++;
      return; /* resident in PE register cache; no DRAM event */
    }
  } else if (g_dedup_enabled && g_dedup && cur_phase == PIM_PHASE_COMPUTE &&
      !is_write &&
      (!g_attr_gated_residency || t->layout_kind != PIM_LAYOUT_KIND_UNSET) &&
      ((!g_matched_mode &&
        (t->role == PIM_ROLE_STREAMED || t->role == PIM_ROLE_OPERAND)) ||
       (t->role == PIM_ROLE_ACCUMULATOR && g_acc_resident_enabled))) {
    /* For tensors flagged as broadcast-scalar (kernel reads them with
     * the same physical address from every bank), promote dedup to the
     * xyz-invariant persistent scope which never resets on program-id
     * boundaries. This models a PE-register cache that retains the
     * broadcast value for the entire COMPUTE phase. Each unique
     * (bank, row, col) tuple still emits exactly once — the promotion
     * catches *additional* repeats across program-id resets that the
     * per-pid scope would otherwise re-emit. Distinct physical tuples
     * are still emitted distinctly. ACCUMULATOR tensors are never
     * bcast_scalar, so this branch only fires for STREAMED/OPERAND. */
    addr_dedup_state_t *bcast_state =
        (t->bcast_scalar && g_dedup_persistent) ? g_dedup_persistent : g_dedup;
    int global_bank = compute_global_bank(loc.ch, loc.pch, loc.bg, loc.bank);
    uint64_t linear_row =
        (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    if (!addr_dedup_check_and_mark(bcast_state, (uint64_t)global_bank,
                                   linear_row, (uint64_t)loc.col)) {
      t->dedup_skips++;
      if (t->role == PIM_ROLE_ACCUMULATOR)
        stat_acc_resident_skips++;
      return; /* already emitted this scope; skip */
    }
  }

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
  if (g_dedup_enabled && g_dedup_store_perpid &&
      cur_phase == PIM_PHASE_COMPUTE && is_write &&
      (!g_attr_gated_residency || t->layout_kind != PIM_LAYOUT_KIND_UNSET) &&
      t->role == PIM_ROLE_ACCUMULATOR) {
    int global_bank = compute_global_bank(loc.ch, loc.pch, loc.bg, loc.bank);
    uint64_t linear_row =
        (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    if (!addr_dedup_check_and_mark(g_dedup_store_perpid, (uint64_t)global_bank,
                                   linear_row, (uint64_t)loc.col)) {
      stat_store_perpid_skips++;
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
  if (g_lockstep_enabled && g_dedup_lockstep &&
      cur_phase == PIM_PHASE_COMPUTE && !is_operand_load) {
    uint64_t tensor_key = (uint64_t)(t - tensors);
    uint64_t linear_row =
        (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    if (!addr_dedup_check_and_mark(g_dedup_lockstep, tensor_key,
                                   linear_row, (uint64_t)loc.col)) {
      stat_lockstep_skips++;
      t->dedup_skips++;
      return;
    }
  }

  emit_access_by_role_phase(t, &loc, is_write);

  /* Per-BG broadcast fanout (PIM_PER_BG_BCAST=1). On the first emission
   * for a bcast_scalar load (i.e. the dedup miss that just fell through
   * to the emit above), also emit one W per other (ch, pch, bg) triple
   * to model the cross-BG bus cycles required for that scalar to reach
   * PEs in other bank-groups. Fires for any LOAD role tagged as
   * bcast_scalar — STREAMED (e.g. row-tiled matmul A) and OPERAND
   * (e.g. matvec x, conv2d Weight) both qualify.
   *
   * Skipped when t->duplicated: the per-BG copies are already in place
   * (laid out by duplicate_bcast_tensor_per_bg as a pre-kernel layout
   * property, with zero setup events), so this PE's access was already
   * routed via map_element to its local BG copy and no cross-BG bus
   * transfer is needed. */
  if (!is_write && t->bcast_scalar &&
      t->role != PIM_ROLE_ACCUMULATOR &&
      cur_phase == PIM_PHASE_COMPUTE && !t->duplicated) {
    emit_per_bg_bcast_fanout(t, &loc);
  }
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

  if (cur_phase == PIM_PHASE_COMPUTE &&
      (!g_attr_gated_residency || t->layout_kind != PIM_LAYOUT_KIND_UNSET) &&
      (t->role == PIM_ROLE_STREAMED || t->role == PIM_ROLE_OPERAND)) {
    int global_bank = compute_global_bank(loc.ch, loc.pch, loc.bg, loc.bank);
    uint64_t linear_row =
        (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    if (g_lru_residency) {
      /* Faithful path: route pid-invariant operand reuse (e.g. matmul B
       * reused across M-tiles) through the SAME per-bank register cache as
       * the per-pid path — one physical PE register file. A skip provably
       * means register-resident at GRF capacity, not a first-N artifact. */
      lru_cache_t *cache = (global_bank >= 0 && global_bank < MAX_BANKS)
                               ? g_lru_resident[global_bank]
                               : NULL;
      if (cache && lru_touch(cache, lru_key((int)(t - tensors), linear_row,
                                            (int)loc.col))) {
        t->persistent_skips++;
        stat_persistent_skips++;
        return;
      }
    } else if (!g_matched_mode && g_dedup_enabled && dedup_state) {
      /* Legacy first-N path. Broadcast-scalar promotion: when flagged,
       * override the axis-scope persistent state the LLVM classifier picked
       * with the xyz-invariant g_dedup_persistent (catches repeats across
       * program-id boundaries that axis-scoped resets would miss). */
      addr_dedup_state_t *use_state = dedup_state;
      uint64_t *use_skips = state_skips;
      if (t->bcast_scalar && g_dedup_persistent) {
        use_state = g_dedup_persistent;
        use_skips = &stat_persistent_skips;
      }
      if (!addr_dedup_check_and_mark(use_state, (uint64_t)global_bank,
                                     linear_row, (uint64_t)loc.col)) {
        t->persistent_skips++;
        if (use_skips)
          (*use_skips)++;
        return;
      }
    }
  }

  /* Lockstep dedup (mirrors pim_trace_access_one). Bypass for OPERAND
   * loads — see access_one for rationale (preserve per-PE register-load
   * cost matching OptiPIM's PimCodeGen abstraction). Persistent-scope
   * loads are always reads here. */
  int is_operand_load_p = (t->role == PIM_ROLE_OPERAND);
  if (g_lockstep_enabled && g_dedup_lockstep &&
      cur_phase == PIM_PHASE_COMPUTE && !is_operand_load_p) {
    uint64_t tensor_key = (uint64_t)(t - tensors);
    uint64_t linear_row =
        (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    if (!addr_dedup_check_and_mark(g_dedup_lockstep, tensor_key,
                                   linear_row, (uint64_t)loc.col)) {
      stat_lockstep_skips++;
      t->dedup_skips++;
      return;
    }
  }

  emit_access_by_role_phase(t, &loc, 0);

  /* Per-BG broadcast fanout (PIM_PER_BG_BCAST=1). See pim_trace_access_one
   * for the rationale; identical injection point for the persistent
   * path. Fires for any LOAD role tagged as bcast_scalar. Skipped when
   * duplicated — the per-BG replicas make the broadcast bank-local. */
  if (t->bcast_scalar &&
      t->role != PIM_ROLE_ACCUMULATOR &&
      cur_phase == PIM_PHASE_COMPUTE && !t->duplicated) {
    emit_per_bg_bcast_fanout(t, &loc);
  }
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
 *   Layer 2 (cross-vector, g_dedup_store_perpid): under the Triton
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
    if (g_dedup_enabled && g_dedup_store_perpid &&
        (!g_attr_gated_residency || t->layout_kind != PIM_LAYOUT_KIND_UNSET)) {
      uint64_t linear_row =
          (uint64_t)sa * (uint64_t)cfg_num_rows + (uint64_t)row;
      if (!addr_dedup_check_and_mark(g_dedup_store_perpid,
                                     (uint64_t)global_bank, linear_row,
                                     (uint64_t)col)) {
        stat_store_perpid_skips++;
        t->dedup_skips++;
        continue;
      }
    }

    /* Layer 3: lockstep collapse. Per (tensor_id, sa, row, col) within a
     * pid, only the first bank's BW emits — the remaining 31 banks are
     * lockstep replicas of the same SIMD store. */
    if (g_lockstep_enabled && g_dedup_lockstep) {
      uint64_t tensor_key = (uint64_t)(t - tensors);
      uint64_t linear_row =
          (uint64_t)sa * (uint64_t)cfg_num_rows + (uint64_t)row;
      if (!addr_dedup_check_and_mark(g_dedup_lockstep, tensor_key,
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

/* Public entry points emitted by MemTracePass. Each one selects an axis-wise
 * dedup state (see comment block on the g_dedup_invariant_* globals). */
void __pim_load_persistent(void *addr, uint64_t size) {
  if (!trace_fp)
    return;
  pim_trace_persistent_range((uint64_t)addr, size, g_dedup_persistent,
                             &stat_persistent_skips);
}

void __pim_load_persistent_yz(void *addr, uint64_t size) {
  if (!trace_fp)
    return;
  pim_trace_persistent_range((uint64_t)addr, size, g_dedup_invariant_yz,
                             &stat_invariant_yz_skips);
}

void __pim_load_persistent_xz(void *addr, uint64_t size) {
  if (!trace_fp)
    return;
  pim_trace_persistent_range((uint64_t)addr, size, g_dedup_invariant_xz,
                             &stat_invariant_xz_skips);
}

void __pim_load_persistent_xy(void *addr, uint64_t size) {
  if (!trace_fp)
    return;
  pim_trace_persistent_range((uint64_t)addr, size, g_dedup_invariant_xy,
                             &stat_invariant_xy_skips);
}

