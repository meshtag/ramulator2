#include "simdram_runtime.h"
#include "im_addr_dedup.h"
#include "im_runtime.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * SIMDRAM Runtime Library
 *
 * Intercepts LLVM-instrumented loads/stores and arithmetic operations,
 * translating them into SIMDRAM trace operations consumable by
 * Ramulator2's PimTrace frontend with the HBM3_PIM DRAM model.
 *
 * Trace format:  <OP> <ch>,<pch>,<bg>,<bank>,<sa>,<row>,<col>
 *   OP: R=read, W=write, BR=bank-read (MAJ gate op), BW=bank-write
 *
 * SIMDRAM data layout: bit-serial — each value of pe_bits bits occupies
 * pe_bits consecutive rows (one bit per row) within a subarray.
 * Each column (dq_bits wide) stores dq_bits values' corresponding bit.
 *
 * SIMDRAM computation: each LLVM arithmetic instruction emits its
 * SIMDRAM paper Table 1 cost as a stream of bank-read (BR) ops
 * cycling through B-group rows (see maj_cost() for the per-opcode
 * polynomials in pe_bits). Every instruction pays its full bit-serial
 * cost — no K-collapse, no per-tensor amortization.
 *
 * Operand R loads (accumulator reads in particular) DO go through the
 * standard load dedup states; this approximates DRAM row-buffer
 * locality, which Ramulator2 also models at the controller level for
 * both stacks. This is a defensible idempotent re-read collapse, not
 * a compute amortization.
 *
 * Note: this differs from OptiPIM's SimDRAMCodeGen, which emits a
 * single fixed-length block of 7*pe_bits^2+1 BR ops per output cell
 * per temporal step regardless of opcode (see
 * third_party/OptiPIM/.../simdram.cpp:55-67, inside the
 * `for (tid = 0; tid < temporal_steps; ...)` loop at line 121 —
 * full K compute cost is paid). Our per-op model emits the per-opcode
 * Table 1 cost per arithmetic instruction; both pay K-scaled compute.
 */

/* ================================================================
 *  HBM3_PIM organization constants — match simdram_hbm3.yaml
 *  Defaults match OptiPIM's simdram_config.yaml exactly.
 * ================================================================ */
static int cfg_num_channels = 16;
static int cfg_num_pch = 2; /* rank: 2 */
static int cfg_num_bg = 4;
static int cfg_num_banks = 4; /* per bank group */
static int cfg_num_sa = 16;
static int cfg_num_rows = 512; /* total physical rows per subarray  */
static int cfg_num_cols = 64;
static int cfg_dq_bits = 128;
static int cfg_pe_bits = 16;

/* Placement scheme. Selectable via PIM_SIMDRAM_LAYOUT env var:
 *
 *   "pinned"      — legacy. Each tensor lives entirely at one
 *                  (ch, pch, bg, bank), packed into row_groups inside
 *                  that one bank's subarrays. Compute fires at the
 *                  accumulator's pinned bank → only ~3 banks (one per
 *                  tensor) ever see commands → no bank-level
 *                  parallelism for SIMDRAM compute, even though we pay
 *                  the cost of replicating inputs to 16 banks via the
 *                  always-on input-replication fanout.
 *
 *   "interleaved" (default) — bit-interleaved placement within ONE
 *                  (ch, pch=0) pair. All tensors share the 16-bank
 *                  window of pch 0. Sequential elements walk
 *                  DQ → bank-in-BG → BG → col → row_group, so adjacent
 *                  element ranges land at different banks. The
 *                  accumulator's elements distribute across many banks,
 *                  so per-cell MAJ-3 sequences fire on different banks
 *                  in parallel. Mirrors OptiPIM's m_spatial_banks
 *                  distribution of output cells, with input replication
 *                  already handling per-consumer-bank operand presence.
 *
 *                  Constraint: cfg_num_banks, cfg_num_bg, cfg_num_cols,
 *                  cfg_dq_bits MUST all be powers of two. simdram_init
 *                  enforces this when interleaved mode is selected.
 *
 *                  Atomic placement unit: one element's pe_bits-row
 *                  bit-serial layout. Multiple values (= dq_bits per
 *                  col_slot) share a col_slot in one bank — those
 *                  cannot be split, which is why DQ is the LSB of the
 *                  bit-walk. Tensors smaller than dq_bits elements
 *                  (e.g. y vector for matvec_64x64 with M=64) fit in
 *                  one col_slot of one bank and don't benefit from
 *                  spreading; tensors >= dq_bits × num_banks_per_pch
 *                  (= 2048 in default config) distribute across all
 *                  16 banks. */
typedef enum {
  SIMDRAM_LAYOUT_PINNED = 0,
  SIMDRAM_LAYOUT_INTERLEAVED = 1,
} simdram_layout_scheme_t;

static simdram_layout_scheme_t cfg_layout_scheme = SIMDRAM_LAYOUT_INTERLEAVED;

/* In interleaved mode, the next free linear-element position. Tensors
 * are appended consecutively in the global linear-element address space
 * within (ch=0, pch=0) and aligned to a dq_bits boundary so a new
 * tensor doesn't share a packed col_slot with the previous one. */
static uint64_t g_interleaved_next_linear = 0;

/* Compute log2(n) where n is a power of two. Returns -1 for n<=0 or
 * non-power-of-2. Used by map_element_interleaved to derive bit field
 * widths from the runtime cfg_* values. */
static int ilog2_pow2(int n) {
  if (n <= 0)
    return -1;
  /* Reject non-powers-of-2: a power of 2 has exactly one set bit. */
  if ((n & (n - 1)) != 0)
    return -1;
  int k = 0;
  while ((1 << k) < n)
    k++;
  return k;
}

/*
 * SIMDRAM subarray row groups (paper §3.1, Fig. 2).
 *
 * Each subarray is partitioned into three groups:
 *
 *   D-group  — regular data rows (program/system data).
 *   C-group  — 2 constant rows: C0 (all-0) and C1 (all-1).
 *              Used as initial inputs (e.g., carry-in for addition)
 *              and for AND/OR reductions.
 *   B-group  — 16 compute rows: T0–T3 (4 regular rows),
 *              DCC0/DCC1 (2 dual-contact-cell rows with separate
 *              d-wordlines and n-wordlines), plus 10 additional
 *              bitwise scratch rows.  Connected to a special row
 *              decoder that can activate three rows simultaneously
 *              (TRA — triple-row activation) for MAJ-3 computation.
 *
 * With 1024 rows/SA (paper): 1006 D + 2 C + 16 B = 1024.
 * With  512 rows/SA (HBM3) :  494 D + 2 C + 16 B = 512.
 *
 * The D-group rows are the only rows available for tensor data.
 * C-group and B-group rows are reserved for SIMDRAM control and
 * computation; they are never allocated for user data.
 */
#define SIMDRAM_C_GROUP_ROWS 2  /* C0 (all-0) + C1 (all-1) */
#define SIMDRAM_B_GROUP_ROWS 16 /* T0–T3 + DCC0/DCC1 + scratch rows */
#define SIMDRAM_RESERVED_ROWS (SIMDRAM_C_GROUP_ROWS + SIMDRAM_B_GROUP_ROWS)

/* Usable data rows per subarray = cfg_num_rows − reserved rows.
 * Computed in simdram_init() once cfg_num_rows is finalized (after
 * any SIMDRAM_NUM_ROWS env override). */
static int cfg_data_rows_per_sa;

/* ================================================================
 *  Tensor registry
 * ================================================================ */
#define MAX_TENSORS 16

typedef struct {
  void *base_addr;
  size_t total_bytes;
  int elem_size;
  int dims[4];
  int num_elements;
  simdram_role_t role;

  /* Physical placement in HBM */
  int assigned_ch;
  int assigned_pch;
  int assigned_bg;
  int assigned_bank;
  int base_row;            /* linear row across all subarrays */
  int elems_per_col_slot;  /* complete values sharing one column
                              slot across a row group (= dq_bits) */
  int elems_per_row_group; /* complete values in one row group
                              (pe_bits rows × num_cols cols)
                              = num_cols × dq_bits              */
  int acc_tracker_idx;     /* index into acc_trackers[] or -1   */

  /* Placement scheme this tensor was registered under. For PINNED, the
   * tensor's data lives in (assigned_ch, assigned_pch, assigned_bg,
   * assigned_bank) starting at base_row. For INTERLEAVED, the
   * layout_linear_base offsets the element index into the global
   * bit-interleaved address space within (ch=0, pch=0); the per-tensor
   * assigned_* fields are unused for placement (kept = 0). */
  simdram_layout_scheme_t layout_scheme;
  uint64_t layout_linear_base;

  /* Per-tensor DRAM-opcode counters. Each emission in the trace file is
   * attributed to exactly one tensor by the call site (compute_trace BRs
   * resolve via dest_addr → acc tracker → tensor_idx; load/store ops
   * resolve via find_tensor(addr) at the access site). */
  uint64_t emitted_R;      /* regular DRAM read  (R)  */
  uint64_t emitted_W;      /* regular DRAM write (W)  */
  uint64_t emitted_BR;     /* bank-read MAJ-3 op (BR) */
  uint64_t emitted_BW;     /* bank-write         (BW) — reserved, currently unused on SIMDRAM */
  uint64_t emitted_WB;     /* broadcast write    (WB) — input replication */
  uint64_t dedup_skips;    /* (bank, sa·rows+row+bit, col) coalesced */
  uint64_t persistent_skips;
} tensor_info_t;

static tensor_info_t tensors[MAX_TENSORS];
static int num_tensors = 0;

/* ================================================================
 *  Runtime state
 * ================================================================ */
static FILE *trace_fp = NULL;
static simdram_phase_t cur_phase = SIMDRAM_PHASE_IDLE;
static int initialized = 0;
static int finalized = 0;

/* Per-bank linear row allocation, indexed by flat bank id */
#define MAX_BANKS 4096
static int next_free_row[MAX_BANKS];

/* Per-accumulator-tensor location tracking.
 * Each registered ACCUMULATOR tensor gets its own tracker entry so that
 * interleaved accesses to different accumulators are handled correctly.
 * current_acc is set on every accumulator load/store and used by
 * __compute_trace to determine where to emit BR ops.
 */
#define MAX_ACC_TRACKERS 16
typedef struct {
  int tensor_idx;
  int ch, pch, bg, bank, sa, row;
} acc_tracker_t;

static acc_tracker_t acc_trackers[MAX_ACC_TRACKERS];
static int num_acc_trackers = 0;
static int current_acc = -1; /* index into acc_trackers */

/* Statistics */
static uint64_t stat_bank_reads = 0;
static uint64_t stat_bank_writes = 0;
static uint64_t stat_reads = 0;
static uint64_t stat_writes = 0;
static uint64_t stat_compute_ops = 0;
static uint64_t stat_ignored = 0;

/* ================================================================
 *  Dedup state — operand-load coalescing
 *
 * Mirrors pim_runtime.c's dedup design but applies only to
 * operand-load trace ops (STREAMED W ops, ACCUMULATOR R ops, HOST
 * R/W ops). The bit-serial gate ops emitted by __compute_trace are
 * NOT routed through this — each MAJ-3 gate is a distinct
 * computation that the simulator must count.
 *
 * For the bit-serial bit-row reads of an operand: the runtime emits
 * `pe_bits` ops at coords (..., base_row + bit, col) for bit in
 * [0, pe_bits). Two distinct kernel loads of the *same* operand
 * element resolve to the same `(bank, sa·rows+base_row+bit, col)`
 * tuple at every bit, and dedup collapses them on the second load.
 * Different operand elements resolve to different physical tuples
 * (different base_row, or different col, or different bank) and
 * are not collapsed.
 *
 * Scope keys:
 *   key1 = global_bank
 *   key2 = sa * cfg_num_rows + base_row + bit
 *   key3 = col
 *
 * Reset cadence (mirrors HBM-PIM):
 *   g_dedup                — reset on any pid axis change
 *   g_dedup_persistent     — reset on phase change
 *   g_dedup_invariant_yz   — reset on pid_x change
 *   g_dedup_invariant_xz   — reset on pid_y change
 *   g_dedup_invariant_xy   — reset on pid_z change
 */
static addr_dedup_state_t *g_dedup = NULL;
static addr_dedup_state_t *g_dedup_persistent = NULL;
static addr_dedup_state_t *g_dedup_invariant_yz = NULL;
static addr_dedup_state_t *g_dedup_invariant_xz = NULL;
static addr_dedup_state_t *g_dedup_invariant_xy = NULL;
static int g_dedup_enabled = 1;

/* Compute-row residency (corrected Lever 2).
 *
 * Models SIMDRAM's column-parallel TRA: one row activation triggers
 * MAJ-3 simultaneously across all DQ-bits columns in that row. Cost
 * is paid ONCE per unique (bank, data-row) per COMPUTE phase. K-iter
 * accumulations into a cell in the same row collapse on the row's
 * single MAJ-3 sequence; sibling cells in the same row's other
 * columns ride that activation for free.
 *
 * This matches OptiPIM's SimDRAMCodeGen emission rule (simdram.cpp:55-67):
 * emit `7n²+1` BRs per unique (row, col) entry in row_col_accesses,
 * but where row_col_accesses keys on row only (cols accumulate into
 * the same row activation). Their per-row cost is what we model here.
 *
 * Reset cadence: phase change only. Persists across pid axes — analogous
 * to g_dedup_persistent. Cross-pid persistence is what makes the model
 * match OptiPIM's `temporal_steps × 7n²+1` cost rather than
 * `pids × temporal_steps × 7n²+1`.
 *
 * PIM_ROW_COMPUTE_DEDUP=0 disables (every compute_trace fires its full
 * maj_cost — useful as the honest-per-arithmetic-op ablation baseline). */
static addr_dedup_state_t *g_compute_row_dedup = NULL;
static int g_compute_row_dedup_enabled = 1;
static uint64_t stat_compute_row_skips = 0;

/* K-axis batch amortization (corrected A2).
 *
 * The earlier row-dedup design (Option A) emitted ONE MAJ-3 cost per
 * unique (bank, row) per phase. That collapsed K-iter MAC chains
 * into one sequence — physically wrong, since each k yields a
 * distinct multiply with different operand rows. OptiPIM amortizes
 * differently: it emits `temporal_steps × 7n²+1` BRs, where
 * temporal_steps reflects the bit-serial-K decomposition chosen by
 * its MILP.
 *
 * Honest approximation: count compute_traces (after the leader-bank
 * gate) and emit one MAJ-3 cost block per BATCH compute_traces.
 * BATCH = cfg_num_cols * cfg_dq_bits / cfg_pe_bits = column-parallel
 * TRA width × bit-serial value depth = the number of arithmetic ops
 * one row activation can amortize over (cells-per-row × pe_bits).
 *
 * Empirically, on matmul_128×128×64 leader-bank this lands at 4-5
 * emissions, matching OptiPIM's `temporal_steps=4`. The emission
 * triggers at the current (bank, row) of the triggering compute_trace,
 * so emissions spread naturally across (bank, row) tuples when present.
 *
 * Override via PIM_K_AMORT_BATCH; set to 1 for the no-amortization
 * baseline (every compute_trace emits its full maj_cost). */
static int g_k_amort_batch = 0;
static uint64_t g_compute_trace_count = 0;
static uint64_t stat_k_amort_emissions = 0;

static uint64_t stat_persistent_skips = 0;
static uint64_t stat_invariant_yz_skips = 0;
static uint64_t stat_invariant_xz_skips = 0;
static uint64_t stat_invariant_xy_skips = 0;

/* SIMDRAM broadcast-write input distribution.
 *
 * SIMDRAM compute (TRA / MAJ-3) requires operand bits to be physically
 * resident in the same subarray as the accumulator. For an input value
 * needed by N consuming banks, its bit-rows have to be written into
 * each bank's subarray.
 *
 * Modeling choice: emit ONE broadcast-write (WB / BCAST_W) per unique
 * STREAMED bit-row instead of N per-bank Ws. The BCAST_W command is a
 * SIMDRAM extension to HBM3-PIM (see third_party/ramulator2/src/dram/
 * impl/HBM3_PIM.cpp) that occupies the pseudochannel bus for nBL
 * cycles and delivers the value to all banks of the (ch, pch) in one
 * channel dispatch. Per-bank execution overlaps in parallel.
 *
 * Rationale: the SIMDRAM paper (Hajinazar et al., ASPLOS '21, §3)
 * claims cross-bank parallelism for SIMDRAM operations, which by
 * implication includes input setup via Ambit-style broadcast. Modeling
 * this as a single channel dispatch with bank-parallel execution is
 * consistent with the paper's hardware claim and with how the model
 * already extends HBM3 (PIMOp, SARD, SAWR, BKRD, BKWR are all
 * SIMDRAM-specific extensions in the HBM3_PIM model).
 *
 * Comparison fairness vs OptiPIM: their codegen emits per-consuming-
 * bank standard W events (input_load_banks fanout). Their cycles
 * reflect that cost. We emit broadcast-writes — their cycle counts
 * remain unchanged. The simulator processes each stack's chosen
 * command stream honestly. */
static uint64_t stat_input_replication_writes = 0;

static int last_program_id = -1;
static int last_program_id_y = -1;
static int last_program_id_z = -1;

/* Leader-bank-only emission gate (mirrors OptiPIM `single_bank_opt`).
 *
 * SIMDRAM compute fires across all banks in a channel in lockstep —
 * a channel-wide row activation triggers MAJ-3 simultaneously on
 * every bank's row buffer. When the workload is uniform per-bank
 * (matmul/matvec/conv2d patterns: each bank does the same loop body
 * on different operand cells), one bank's emitted op stream is a
 * valid proxy for the channel's wallclock — the simulator's
 * bank-parallel scheduling would produce the same cycle count if all
 * 32 banks emitted in parallel.
 *
 * Concretely: the host harness iterates (pid × bank_id) calling the
 * kernel. With the gate on, only the bank_id == 0 iteration emits
 * COMPUTE-phase ops; bank_id 1..N skip. HOST-phase setup ops are
 * unaffected (per-bank input placement is not uniform across banks).
 *
 * Matches OptiPIM SimDRAMCodeGen at simdram.cpp:136-142 (emits first
 * bank's row_col_accesses then breaks the spatial loop). Soundness
 * is the same uniformity argument both compilers rely on.
 *
 * PIM_LEADER_BANK_ONLY=0 disables the gate (every bank emits its own
 * trace; useful for ablation against the leader-bank model). */
static int g_leader_bank_only_enabled = 1;
static uint64_t stat_leader_bank_skips = 0;

/* PIM_BCAST_W: select the input-replication emission mode.
 *
 * Default 0 (per-bank fanout): emit cfg_num_bg * cfg_num_banks regular
 *   WR events per STREAMED bit-row. Mirrors OptiPIM SimDRAMCodeGen
 *   (simdram.cpp:46-49) which pays the per-bank WR cost. Hardware
 *   target: plain SIMDRAM, no BCAST_W extension.
 *
 * Set to 1: emit a single BCAST_W ("WB") per STREAMED bit-row.
 *   Models the SIMDRAM paper's BCAST_W extension (MICRO 2021); one
 *   pseudochannel-bus dispatch hits all banks in the (ch, pch) with
 *   per-bank execution latency overlapped in parallel. Hardware
 *   target: SIMDRAM + BCAST_W extension.
 *
 * Default flipped from 1 to 0 on 2026-05-27 for fairness symmetry
 * with OptiPIM's input-replication cost model. */
static int g_bcast_w_enabled = 0;

/* Persistent-load classification, set by __pim_load_persistent[_*]
 * before the access falls into simdram_trace_access. The classification
 * tells simdram_trace_access which dedup state to push the access
 * through. Reset to PIM_LOAD_NORMAL after each classified call. */
typedef enum {
  PIM_LOAD_NORMAL = 0,
  PIM_LOAD_PERSISTENT_XYZ,
  PIM_LOAD_INVARIANT_YZ,
  PIM_LOAD_INVARIANT_XZ,
  PIM_LOAD_INVARIANT_XY,
} pim_load_class_t;

static pim_load_class_t cur_load_class = PIM_LOAD_NORMAL;

/* ================================================================
 *  Helpers
 * ================================================================ */

static void emit_trace(const char *op, int ch, int pch, int bg, int bank,
                       int sa, int row, int col) {
  fprintf(trace_fp, "%s %d,%d,%d,%d,%d,%d,%d\n", op, ch, pch, bg, bank, sa, row,
          col);
}

static int compute_global_bank(int ch, int pch, int bg, int bank) {
  int banks_per_pch = cfg_num_bg * cfg_num_banks;
  int banks_per_ch = cfg_num_pch * banks_per_pch;
  return ch * banks_per_ch + pch * banks_per_pch + bg * cfg_num_banks + bank;
}

/* STREAMED bit-row input broadcast. Two modes, selected by
 * PIM_BCAST_W:
 *
 * Default (PIM_BCAST_W=0): per-bank fanout. Emits one regular WR
 *   (opcode "W") per receiving bank in the (src_ch, src_pch) — i.e.
 *   cfg_num_bg * cfg_num_banks separate events per bit-row. Matches
 *   OptiPIM SimDRAMCodeGen's per-bank input replication
 *   (simdram.cpp:46-49), which pays a full WR cycle per receiving
 *   bank. Hardware target: plain SIMDRAM, no BCAST_W extension.
 *
 * PIM_BCAST_W=1: single broadcast-write event ("WB" → BCAST_W
 *   opcode in HBM3_PIM.cpp:43). One pseudochannel-bus dispatch
 *   covers all banks in the (ch, pch); per-bank execution latency
 *   overlaps in parallel. Hardware target: SIMDRAM + BCAST_W
 *   extension (Hajinazar et al., MICRO 2021).
 *
 * Default flipped to per-bank fanout on 2026-05-27 for symmetry
 * with OptiPIM's input-replication cost model. */
static void emit_input_broadcast(tensor_info_t *t,
                                 int src_ch, int src_pch,
                                 int src_bg, int src_bank,
                                 int sa, int row, int col) {
  (void)src_bg;
  (void)src_bank;
  if (g_bcast_w_enabled) {
    emit_trace("WB", src_ch, src_pch, /*bg=*/0, /*bank=*/0, sa, row, col);
    stat_writes++;
    t->emitted_WB++;
    stat_input_replication_writes++;
    return;
  }
  for (int bg = 0; bg < cfg_num_bg; bg++) {
    for (int bk = 0; bk < cfg_num_banks; bk++) {
      emit_trace("W", src_ch, src_pch, bg, bk, sa, row, col);
      stat_writes++;
      t->emitted_W++;
      stat_input_replication_writes++;
    }
  }
}

static void destroy_dedup_state(void) {
  if (g_dedup)               { addr_dedup_destroy(g_dedup);              g_dedup = NULL; }
  if (g_dedup_persistent)    { addr_dedup_destroy(g_dedup_persistent);   g_dedup_persistent = NULL; }
  if (g_dedup_invariant_yz)  { addr_dedup_destroy(g_dedup_invariant_yz); g_dedup_invariant_yz = NULL; }
  if (g_dedup_invariant_xz)  { addr_dedup_destroy(g_dedup_invariant_xz); g_dedup_invariant_xz = NULL; }
  if (g_dedup_invariant_xy)  { addr_dedup_destroy(g_dedup_invariant_xy); g_dedup_invariant_xy = NULL; }
  if (g_compute_row_dedup)   { addr_dedup_destroy(g_compute_row_dedup);  g_compute_row_dedup = NULL; }
}

/* Track program-id transitions on each access entry. Resets the dedup
 * states whose scope just closed:
 *   - g_dedup                reset on ANY pid axis change
 *   - g_dedup_invariant_yz   reset on pid_x change
 *   - g_dedup_invariant_xz   reset on pid_y change
 *   - g_dedup_invariant_xy   reset on pid_z change
 *
 * g_dedup_persistent is xyz-invariant — it never resets on pid changes
 * (only on phase change in simdram_set_phase). */
static void advance_program_epoch_if_needed(void) {
  int pid = __pim_get_program_id();
  int pid_y = __pim_get_program_id_y();
  int pid_z = __pim_get_program_id_z();

  int x_changed = (pid != last_program_id);
  int y_changed = (pid_y != last_program_id_y);
  int z_changed = (pid_z != last_program_id_z);

  if (!(x_changed || y_changed || z_changed))
    return;

  last_program_id = pid;
  last_program_id_y = pid_y;
  last_program_id_z = pid_z;

  if (g_dedup)              addr_dedup_reset(g_dedup);
  if (x_changed && g_dedup_invariant_yz) addr_dedup_reset(g_dedup_invariant_yz);
  if (y_changed && g_dedup_invariant_xz) addr_dedup_reset(g_dedup_invariant_xz);
  if (z_changed && g_dedup_invariant_xy) addr_dedup_reset(g_dedup_invariant_xy);
}

/* Resolve the dedup state to consult based on the current load
 * classification (set by __pim_load_persistent[_*]). Returns NULL when
 * no dedup applies (dedup disabled, or ACCUMULATOR-side ops we
 * deliberately don't dedup across the per-pid scope). */
static addr_dedup_state_t *pick_load_dedup_state(const tensor_info_t *t) {
  (void)t;
  if (!g_dedup_enabled)
    return NULL;
  switch (cur_load_class) {
    case PIM_LOAD_PERSISTENT_XYZ: return g_dedup_persistent;
    case PIM_LOAD_INVARIANT_YZ:   return g_dedup_invariant_yz;
    case PIM_LOAD_INVARIANT_XZ:   return g_dedup_invariant_xz;
    case PIM_LOAD_INVARIANT_XY:   return g_dedup_invariant_xy;
    case PIM_LOAD_NORMAL:         return g_dedup;
  }
  return g_dedup;
}

static void account_load_dedup_skip(tensor_info_t *t,
                                    const addr_dedup_state_t *which) {
  t->dedup_skips++;
  if (which == g_dedup_persistent)         { stat_persistent_skips++;    t->persistent_skips++; }
  else if (which == g_dedup_invariant_yz)  { stat_invariant_yz_skips++;  t->persistent_skips++; }
  else if (which == g_dedup_invariant_xz)  { stat_invariant_xz_skips++;  t->persistent_skips++; }
  else if (which == g_dedup_invariant_xy)  { stat_invariant_xy_skips++;  t->persistent_skips++; }
}

/*
 * Find which registered tensor contains 'addr'.
 * Returns tensor index or -1 if not found.
 */
static int find_tensor(uint64_t addr) {
  for (int i = 0; i < num_tensors; i++) {
    uint64_t base = (uint64_t)tensors[i].base_addr;
    if (addr >= base && addr < base + tensors[i].total_bytes)
      return i;
  }
  return -1;
}

/*
 * Bit-interleaved element-to-physical mapping for SIMDRAM.
 *
 * Address bit layout (LSB → MSB):
 *   [ DQ | bank-within-BG | BG | col | row_group_idx ]
 *
 * Sequential element indices first fill the DQ slot (dq_bits values
 * share one col_slot bit-serially), then advance through banks within a
 * BG, then BGs, then cols, then row_groups. All tensors live in
 * (ch=0, pch=0) so the 16-bank within-pch fanout of input replication
 * naturally covers every consumer bank.
 *
 * Atomic unit: one element's pe_bits-row bit-serial layout — the bits
 * of one value cannot be split across banks. DQ is therefore the LSB
 * (dq_bits values pack into one bank's col_slot at the same bit
 * position) and bank increments only after a col_slot is full.
 *
 * Field widths are derived from log2 of cfg_dq_bits, cfg_num_banks,
 * cfg_num_bg, cfg_num_cols at runtime — all four must be powers of
 * two, which simdram_init enforces when interleaved mode is selected.
 *
 * Worked example for the default 4-banks/BG × 4 BG config:
 *   bits 0..6   DQ (128 values per col_slot)
 *   bits 7..8   bank within BG (4)
 *   bits 9..10  BG (4)
 *   bits 11..16 col (64 cols)
 *   bits 17+    row_group_idx (each row_group = pe_bits rows in the SA)
 * Under that config: elements 0..127 share col_slot 0 of bank 0 of
 * BG 0; 128..255 advance to bank 1 of BG 0; after 4 banks (512 elements)
 * advance to BG 1; after 4 BGs (2048 elements) advance to col 1; etc. */
static void map_element_interleaved(const tensor_info_t *t, int elem_idx,
                                    int *ch, int *pch, int *bg, int *bank,
                                    int *sa, int *base_row, int *col) {
  uint64_t linear = t->layout_linear_base + (uint64_t)elem_idx;

  int log2_dq    = ilog2_pow2(cfg_dq_bits);
  int log2_banks = ilog2_pow2(cfg_num_banks);
  int log2_bg    = ilog2_pow2(cfg_num_bg);
  int log2_cols  = ilog2_pow2(cfg_num_cols);

  /* Strip the DQ bits — dq_bits values pack into one col_slot. */
  linear >>= log2_dq;

  *bank = (int)(linear & ((1ULL << log2_banks) - 1));
  linear >>= log2_banks;

  *bg = (int)(linear & ((1ULL << log2_bg) - 1));
  linear >>= log2_bg;

  *col = (int)(linear & ((1ULL << log2_cols) - 1));
  linear >>= log2_cols;

  /* Remaining bits = row_group index within the bank's SAs. Each
   * row_group occupies pe_bits consecutive rows; advance the linear
   * row counter accordingly. */
  uint64_t linear_row = linear * (uint64_t)cfg_pe_bits;
  *sa = (int)(linear_row / (uint64_t)cfg_data_rows_per_sa);
  *base_row = (int)(linear_row % (uint64_t)cfg_data_rows_per_sa);

  /* Interleaved layout confines tensor data to one (ch, pch) so that
   * input replication's per-pch fanout makes operand bits available
   * at every consuming bank. */
  *ch = 0;
  *pch = 0;
}

/*
 * Map a tensor element to physical HBM coordinates for SIMDRAM.
 *
 * Dispatches between the legacy pinned layout (one tensor → one bank,
 * compute serialized through that bank) and the bit-interleaved
 * layout (elements distributed across all banks in one (ch, pch)).
 *
 * Bit-serial layout: each element occupies pe_bits consecutive rows.
 * Within a row_group (pe_bits rows), each column stores dq_bits values.
 * Returns the base row of the pe_bits-row group containing this element.
 */
static void map_element(const tensor_info_t *t, int elem_idx, int *ch, int *pch,
                        int *bg, int *bank, int *sa, int *base_row, int *col) {
  if (t->layout_scheme == SIMDRAM_LAYOUT_INTERLEAVED) {
    map_element_interleaved(t, elem_idx, ch, pch, bg, bank, sa, base_row, col);
    return;
  }

  *ch = t->assigned_ch;
  *pch = t->assigned_pch;
  *bg = t->assigned_bg;
  *bank = t->assigned_bank;

  int row_group_idx = elem_idx / t->elems_per_row_group;
  *col = (elem_idx % t->elems_per_row_group) / t->elems_per_col_slot;

  int linear_base = t->base_row + row_group_idx * cfg_pe_bits;
  *sa = linear_base / cfg_data_rows_per_sa;
  *base_row = linear_base % cfg_data_rows_per_sa;
}

/* PIM_MUL_OPTIPIM (default ON) selects the multiplication MAJ-3 cost.
 *
 * Default ON: use OptiPIM's emulated 7n² + 1 formula (see
 * third_party/OptiPIM/simulator/src/pim_codegen/impl/simdram.cpp:58).
 * This matches OptiPIM's cost model so the comparison is MUL-formula-
 * symmetric.
 *
 * Set PIM_MUL_OPTIPIM=0 to use the SIMDRAM paper Table 1 polynomial
 * (11n² − 5n − 1) — useful for ablation / cross-checking against the
 * canonical silicon model. */
static int g_optipim_mul_formula = 1;

/*
 * Compute SIMDRAM AAP/AP cost for a given integer operation.
 * Costs from the SIMDRAM paper (Table 1), in units of AAPs/APs,
 * where n = pe_bits (number of bits per operand, stored bit-serially).
 *
 *   Operation       Cost (AAPs/APs)     Class
 *   ─────────────   ─────────────────   ─────────
 *   addition        8n + 1              Linear
 *   subtraction     8n + 1              Linear
 *   multiplication  7n² + 1             Quadratic  (default, OptiPIM-
 *                                                   symmetric; set
 *                                                   PIM_MUL_OPTIPIM=0
 *                                                   for paper 11n²−5n−1)
 *   division        8n² + 12n           Quadratic
 *   bitwise AND     n                   Linear   (1 MAJ per bit)
 *   bitwise OR      n                   Linear   (1 MAJ per bit)
 *   bitwise XOR     5n                  Linear   (~5 MAJ per bit)
 *   shift           n                   Linear
 *
 * Floating-point operations are NOT supported by SIMDRAM (which operates
 * on raw bits via MAJ-3 gates).  Any float opcode receives cost 0 and
 * produces no trace output.
 */
static int maj_cost(int opcode) {
  int n = cfg_pe_bits;
  switch (opcode) {
  case SIMDRAM_OP_MUL:
    return g_optipim_mul_formula ? (7 * n * n + 1)
                                 : (11 * n * n - 5 * n - 1);
  case SIMDRAM_OP_ADD:
  case SIMDRAM_OP_SUB:
    return 8 * n + 1;
  case SIMDRAM_OP_AND:
  case SIMDRAM_OP_OR:
    return n;
  case SIMDRAM_OP_XOR:
    return 5 * n;
  case SIMDRAM_OP_SHL:
  case SIMDRAM_OP_LSHR:
  case SIMDRAM_OP_ASHR:
    return n;
  case SIMDRAM_OP_DIV:
  case SIMDRAM_OP_REM:
    return 8 * n * n + 12 * n;
  default:
    /* Float ops and anything else: not supported, skip silently */
    return 0;
  }
}

/* ================================================================
 *  Public API
 * ================================================================ */

void simdram_init(const char *trace_file) {
  if (initialized && trace_fp) {
    fclose(trace_fp);
    trace_fp = NULL;
  }

  if (!trace_file)
    trace_file = "simdram_trace.txt";

  trace_fp = fopen(trace_file, "w");
  if (!trace_fp) {
    fprintf(stderr, "[simdram] ERROR: cannot open %s\n", trace_file);
    exit(1);
  }

  /* Read optional overrides from environment */
  const char *v;
  if ((v = getenv("SIMDRAM_NUM_CHANNELS")))
    cfg_num_channels = atoi(v);
  if ((v = getenv("SIMDRAM_NUM_PCH")))
    cfg_num_pch = atoi(v);
  if ((v = getenv("SIMDRAM_NUM_BG")))
    cfg_num_bg = atoi(v);
  if ((v = getenv("SIMDRAM_NUM_BANKS")))
    cfg_num_banks = atoi(v);
  if ((v = getenv("SIMDRAM_NUM_SA")))
    cfg_num_sa = atoi(v);
  if ((v = getenv("SIMDRAM_NUM_ROWS")))
    cfg_num_rows = atoi(v);
  if ((v = getenv("SIMDRAM_NUM_COLS")))
    cfg_num_cols = atoi(v);
  if ((v = getenv("SIMDRAM_DQ_BITS")))
    cfg_dq_bits = atoi(v);
  if ((v = getenv("SIMDRAM_PE_BITS")))
    cfg_pe_bits = atoi(v);

  /* Recompute usable data rows after any cfg_num_rows override */
  cfg_data_rows_per_sa = cfg_num_rows - SIMDRAM_RESERVED_ROWS;
  if (cfg_data_rows_per_sa <= 0) {
    fprintf(stderr,
            "[simdram] ERROR: cfg_num_rows=%d too small for "
            "reserved rows (%d C-group + %d B-group = %d)\n",
            cfg_num_rows, SIMDRAM_C_GROUP_ROWS, SIMDRAM_B_GROUP_ROWS,
            SIMDRAM_RESERVED_ROWS);
    exit(1);
  }

  num_tensors = 0;
  cur_phase = SIMDRAM_PHASE_IDLE;
  memset(next_free_row, 0, sizeof(next_free_row));
  stat_bank_reads = stat_bank_writes = stat_reads = stat_writes = 0;
  stat_compute_ops = stat_ignored = 0;
  num_acc_trackers = 0;
  current_acc = -1;
  initialized = 1;
  finalized = 0;

  /* Placement scheme. PIM_SIMDRAM_LAYOUT=pinned reverts to legacy
   * one-tensor-per-bank placement; default is "interleaved" (bit-walk
   * distribution within one (ch=0, pch=0)). */
  const char *layout_env = getenv("PIM_SIMDRAM_LAYOUT");
  if (layout_env && layout_env[0] == 'p') {
    cfg_layout_scheme = SIMDRAM_LAYOUT_PINNED;
  } else {
    cfg_layout_scheme = SIMDRAM_LAYOUT_INTERLEAVED;
  }
  g_interleaved_next_linear = 0;
  fprintf(stderr,
          "[simdram] layout=%s (set PIM_SIMDRAM_LAYOUT=pinned to revert)\n",
          cfg_layout_scheme == SIMDRAM_LAYOUT_INTERLEAVED ? "interleaved"
                                                          : "pinned");

  /* Power-of-2 guard for interleaved bit walk. */
  if (cfg_layout_scheme == SIMDRAM_LAYOUT_INTERLEAVED) {
    if (ilog2_pow2(cfg_num_banks) < 0 || ilog2_pow2(cfg_num_bg) < 0 ||
        ilog2_pow2(cfg_num_cols) < 0 || ilog2_pow2(cfg_dq_bits) < 0) {
      fprintf(stderr,
              "[simdram] ERROR: interleaved layout requires power-of-2 "
              "cfg_num_banks (%d), cfg_num_bg (%d), cfg_num_cols (%d), "
              "cfg_dq_bits (%d). Set PIM_SIMDRAM_LAYOUT=pinned for "
              "non-power-of-2 configs.\n",
              cfg_num_banks, cfg_num_bg, cfg_num_cols, cfg_dq_bits);
      exit(1);
    }
  }

  /* Dedup state setup. PIM_DEDUP=0 disables (for ablation). */
  destroy_dedup_state();
  const char *dedup_env = getenv("PIM_DEDUP");
  g_dedup_enabled = (dedup_env && dedup_env[0] == '0') ? 0 : 1;

  stat_persistent_skips = stat_invariant_yz_skips = 0;
  stat_invariant_xz_skips = stat_invariant_xy_skips = 0;

  /* Per-consuming-bank input replication is always on; disabling it
   * would model a non-existent infinite cross-bank broadcast cache. */
  stat_input_replication_writes = 0;

  /* MUL-formula override. Defaults OFF (paper Table 1: 11n²−5n−1).
   * Default ON (uses OptiPIM-symmetric 7n²+1); set PIM_MUL_OPTIPIM=0
   * to switch to the paper's 11n²−5n−1 silicon formula. */
  const char *mul_env = getenv("PIM_MUL_OPTIPIM");
  g_optipim_mul_formula = (mul_env && mul_env[0] == '0') ? 0 : 1;
  fprintf(stderr,
          "[simdram] mul_formula=%s (set PIM_MUL_OPTIPIM=0 for paper Table 1 11n²−5n−1)\n",
          g_optipim_mul_formula ? "7n²+1 (OptiPIM)" : "11n²−5n−1 (Table 1)");

  /* Leader-bank-only emission. Defaults ON (matches OptiPIM
   * single_bank_opt); PIM_LEADER_BANK_ONLY=0 disables for ablation. */
  const char *leader_env = getenv("PIM_LEADER_BANK_ONLY");
  g_leader_bank_only_enabled = (leader_env && leader_env[0] == '0') ? 0 : 1;
  stat_leader_bank_skips = 0;
  fprintf(stderr,
          "[simdram] leader_bank_only=%d (set PIM_LEADER_BANK_ONLY=0 to disable)\n",
          g_leader_bank_only_enabled);

  /* BCAST_W input-replication shortcut. Defaults OFF (per-bank WR
   * fanout — symmetric with OptiPIM). Set PIM_BCAST_W=1 to enable the
   * SIMDRAM-paper BCAST_W extension (single broadcast event). */
  const char *bcast_env = getenv("PIM_BCAST_W");
  g_bcast_w_enabled = (bcast_env && bcast_env[0] == '1') ? 1 : 0;
  fprintf(stderr,
          "[simdram] bcast_w=%d (set PIM_BCAST_W=1 to use SIMDRAM BCAST_W extension)\n",
          g_bcast_w_enabled);

  last_program_id = last_program_id_y = last_program_id_z = -1;
  cur_load_class = PIM_LOAD_NORMAL;
  if (g_dedup_enabled) {
    /* Capacity hints: per-pid state sees one tile's worth of accesses
     * (small); persistent states accumulate across all pids in the
     * COMPUTE phase (big). 256k slots cover the largest workloads we
     * model with comfortable load factor. */
    const int perpid_cap = 4096;
    const int persistent_cap = 262144;
    g_dedup              = addr_dedup_create(perpid_cap);
    g_dedup_persistent   = addr_dedup_create(persistent_cap);
    g_dedup_invariant_yz = addr_dedup_create(persistent_cap);
    g_dedup_invariant_xz = addr_dedup_create(persistent_cap);
    g_dedup_invariant_xy = addr_dedup_create(persistent_cap);
    g_compute_row_dedup  = addr_dedup_create(persistent_cap);
  }

  /* Compute-row residency (corrected Lever 2). Defaults ON — models
   * SIMDRAM's column-parallel TRA so per-row compute cost matches
   * OptiPIM's per-temporal-step emission. PIM_ROW_COMPUTE_DEDUP=0
   * disables (ablation: every arithmetic op fires its full maj_cost). */
  const char *row_dedup_env = getenv("PIM_ROW_COMPUTE_DEDUP");
  g_compute_row_dedup_enabled = (row_dedup_env && row_dedup_env[0] == '0') ? 0 : 1;
  stat_compute_row_skips = 0;
  fprintf(stderr,
          "[simdram] row_compute_dedup=%d (set PIM_ROW_COMPUTE_DEDUP=0 to disable)\n",
          g_compute_row_dedup_enabled);

  /* K-axis batch amortization: emit MAJ-3 cost every BATCH
   * compute_traces. Default BATCH = column-parallel TRA width × bit-
   * serial depth = n_cols × dq_bits × ... wait, let me redo:
   * BATCH = cfg_num_cols * cfg_dq_bits / cfg_pe_bits. This represents
   * the number of arithmetic operations one row activation amortizes
   * across (cells-per-row × pe-bits). PIM_K_AMORT_BATCH overrides;
   * set to 1 for the no-amortization baseline. */
  g_k_amort_batch = (cfg_num_cols * cfg_dq_bits) / cfg_pe_bits;
  if (g_k_amort_batch < 1) g_k_amort_batch = 1;
  const char *kamort_env = getenv("PIM_K_AMORT_BATCH");
  if (kamort_env) {
    int v = atoi(kamort_env);
    if (v >= 1) g_k_amort_batch = v;
  }
  g_compute_trace_count = 0;
  stat_k_amort_emissions = 0;
  fprintf(stderr,
          "[simdram] k_amort_batch=%d (set PIM_K_AMORT_BATCH=N to override; 1 disables)\n",
          g_k_amort_batch);

  fprintf(stderr, "[simdram] Initialized. Trace: %s\n", trace_file);
  fprintf(stderr,
          "[simdram] HBM config: %d ch, %d pch, %d bg, %d banks/bg, "
          "%d sa, %d rows/sa (%d data + %d reserved), %d cols, "
          "%d-bit DQ, pe_bits=%d\n",
          cfg_num_channels, cfg_num_pch, cfg_num_bg, cfg_num_banks, cfg_num_sa,
          cfg_num_rows, cfg_data_rows_per_sa, SIMDRAM_RESERVED_ROWS,
          cfg_num_cols, cfg_dq_bits, cfg_pe_bits);
}

int simdram_register_tensor(void *ptr, const int *dims, int ndims,
                            int elem_size, simdram_role_t role) {
  if (num_tensors >= MAX_TENSORS) {
    fprintf(stderr, "[simdram] ERROR: too many tensors (max %d)\n",
            MAX_TENSORS);
    return -1;
  }

  tensor_info_t *t = &tensors[num_tensors];
  t->base_addr = ptr;
  t->elem_size = elem_size;
  t->role = role;
  t->emitted_R = t->emitted_W = 0;
  t->emitted_BR = t->emitted_BW = t->emitted_WB = 0;
  t->dedup_skips = t->persistent_skips = 0;

  int total = 1;
  for (int i = 0; i < ndims && i < 4; i++) {
    t->dims[i] = dims[i];
    total *= dims[i];
  }
  t->num_elements = total;
  t->total_bytes = (size_t)total * elem_size;

  /*
   * Bit-serial layout:
   *
   * One row group = pe_bits consecutive rows.  Within a row group:
   *   - Each column slot is dq_bits bit-cells wide.
   *   - One column slot across pe_bits rows stores dq_bits complete values
   *     (one bit per value per row, dq_bits values per slot).
   *   - A full row group stores num_cols × dq_bits complete values.
   *
   * A single TRA (triple-row activation) operates on the entire row width,
   * so all num_cols × dq_bits values are computed in parallel.
   */
  t->elems_per_col_slot = cfg_dq_bits;
  t->elems_per_row_group = cfg_num_cols * t->elems_per_col_slot;
  t->layout_scheme = cfg_layout_scheme;
  t->layout_linear_base = 0;

  const char *role_str = (role == SIMDRAM_ROLE_STREAMED)  ? "STREAMED"
                         : (role == SIMDRAM_ROLE_OPERAND) ? "OPERAND"
                                                          : "ACCUMULATOR";

  if (cfg_layout_scheme == SIMDRAM_LAYOUT_INTERLEAVED) {
    /* Align tensor base to a dq_bits boundary so a new tensor doesn't
     * partially share a packed col_slot with the previous one. */
    g_interleaved_next_linear =
        ((g_interleaved_next_linear + (uint64_t)cfg_dq_bits - 1) /
         (uint64_t)cfg_dq_bits) *
        (uint64_t)cfg_dq_bits;
    t->layout_linear_base = g_interleaved_next_linear;
    g_interleaved_next_linear += (uint64_t)total;

    /* All interleaved tensors live in (ch=0, pch=0). assigned_bg/bank
     * are placeholders; the actual physical placement is computed
     * per-access via map_element_interleaved. */
    t->assigned_ch = 0;
    t->assigned_pch = 0;
    t->assigned_bg = 0;
    t->assigned_bank = 0;
    t->base_row = 0;

    /* Per-tensor accumulator tracker. In interleaved mode the tracker's
     * (ch, pch, bg, bank, sa, row) is updated dynamically on every
     * accumulator access (each cell lives at a different bank). The
     * initial values are placeholders. */
    t->acc_tracker_idx = -1;
    if (role == SIMDRAM_ROLE_ACCUMULATOR && num_acc_trackers < MAX_ACC_TRACKERS) {
      int idx = num_acc_trackers++;
      t->acc_tracker_idx = idx;
      acc_trackers[idx].tensor_idx = num_tensors;
      acc_trackers[idx].ch = 0;
      acc_trackers[idx].pch = 0;
      acc_trackers[idx].bg = 0;
      acc_trackers[idx].bank = 0;
      acc_trackers[idx].sa = 0;
      acc_trackers[idx].row = 0;
      if (current_acc < 0)
        current_acc = idx;
    }

    fprintf(stderr,
            "[simdram] Tensor %d: %s, %d elems (%zu bytes), interleaved "
            "in (ch=0, pch=0), linear_base=%" PRIu64 "\n",
            num_tensors, role_str, total, t->total_bytes,
            t->layout_linear_base);
    return num_tensors++;
  }

  /* PINNED layout: each tensor lives at one assigned bank. */
  int total_flat_banks =
      cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
  int global_bank = num_tensors % total_flat_banks;

  int banks_per_pch = cfg_num_bg * cfg_num_banks;
  int banks_per_ch = cfg_num_pch * banks_per_pch;

  t->assigned_ch = global_bank / banks_per_ch;
  t->assigned_pch = (global_bank / banks_per_pch) % cfg_num_pch;
  t->assigned_bg = (global_bank / cfg_num_banks) % cfg_num_bg;
  t->assigned_bank = global_bank % cfg_num_banks;

  /* Linear rows needed: ceil(total / elems_per_row_group) × pe_bits.
   * Only D-group rows are usable for data (cfg_data_rows_per_sa per SA). */
  int row_groups =
      (total + t->elems_per_row_group - 1) / t->elems_per_row_group;
  int rows_needed = row_groups * cfg_pe_bits;
  int rows_capacity = cfg_num_sa * cfg_data_rows_per_sa;

  if (next_free_row[global_bank] + rows_needed > rows_capacity) {
    fprintf(stderr,
            "[simdram] ERROR: tensor %d needs %d rows but bank "
            "(ch=%d pch=%d bg=%d bank=%d) only has %d free of %d\n",
            num_tensors, rows_needed, t->assigned_ch, t->assigned_pch,
            t->assigned_bg, t->assigned_bank,
            rows_capacity - next_free_row[global_bank], rows_capacity);
    return -1;
  }

  t->base_row = next_free_row[global_bank];
  next_free_row[global_bank] += rows_needed;

  /* Initialize per-tensor accumulator tracker */
  t->acc_tracker_idx = -1;
  if (role == SIMDRAM_ROLE_ACCUMULATOR && num_acc_trackers < MAX_ACC_TRACKERS) {
    int idx = num_acc_trackers++;
    t->acc_tracker_idx = idx;
    acc_trackers[idx].tensor_idx = num_tensors; /* not yet incremented */
    acc_trackers[idx].ch = t->assigned_ch;
    acc_trackers[idx].pch = t->assigned_pch;
    acc_trackers[idx].bg = t->assigned_bg;
    acc_trackers[idx].bank = t->assigned_bank;
    acc_trackers[idx].sa = t->base_row / cfg_data_rows_per_sa;
    acc_trackers[idx].row = t->base_row % cfg_data_rows_per_sa;
    if (current_acc < 0)
      current_acc = idx; /* default to first registered accumulator */
  }

  int first_sa = t->base_row / cfg_data_rows_per_sa;
  int last_sa = (t->base_row + rows_needed - 1) / cfg_data_rows_per_sa;

  fprintf(stderr,
          "[simdram] Tensor %d: %s, %d elems (%zu bytes), "
          "ch=%d pch=%d bg=%d bank=%d sa=[%d..%d] rows=[%d..%d]\n",
          num_tensors, role_str, total, t->total_bytes, t->assigned_ch,
          t->assigned_pch, t->assigned_bg, t->assigned_bank, first_sa, last_sa,
          t->base_row, t->base_row + rows_needed - 1);

  return num_tensors++;
}

void simdram_set_phase(simdram_phase_t phase) {
  const char *names[] = {"IDLE", "COMPUTE", "HOST"};
  if (phase <= SIMDRAM_PHASE_HOST)
    fprintf(stderr, "[simdram] Phase -> %s\n", names[phase]);
  /* Phase change resets all dedup states, including the xyz-invariant
   * persistent one. The PE-register-cache model is bounded to a
   * single COMPUTE phase. */
  if (cur_phase != phase) {
    if (g_dedup)              addr_dedup_reset(g_dedup);
    if (g_dedup_persistent)   addr_dedup_reset(g_dedup_persistent);
    if (g_dedup_invariant_yz) addr_dedup_reset(g_dedup_invariant_yz);
    if (g_dedup_invariant_xz) addr_dedup_reset(g_dedup_invariant_xz);
    if (g_dedup_invariant_xy) addr_dedup_reset(g_dedup_invariant_xy);
    if (g_compute_row_dedup)  addr_dedup_reset(g_compute_row_dedup);
    g_compute_trace_count = 0;
  }
  cur_phase = phase;
}

void simdram_finalize(void) {
  if (finalized)
    return;
  finalized = 1;

  if (trace_fp) {
    fclose(trace_fp);
    trace_fp = NULL;
  }

  fprintf(stderr, "\n[simdram] === SIMDRAM Trace Statistics ===\n");
  fprintf(stderr, "[simdram]   Bank-reads  (BR): %" PRIu64 "\n",
          stat_bank_reads);
  fprintf(stderr, "[simdram]   Bank-writes (BW): %" PRIu64 "\n",
          stat_bank_writes);
  fprintf(stderr, "[simdram]   Reads       (R) : %" PRIu64 "\n", stat_reads);
  fprintf(stderr, "[simdram]   Writes      (W) : %" PRIu64 "\n", stat_writes);
  fprintf(stderr, "[simdram]   Compute ops     : %" PRIu64 "\n",
          stat_compute_ops);
  fprintf(stderr, "[simdram]   Ignored         : %" PRIu64 "\n", stat_ignored);
  fprintf(stderr, "[simdram]   Total SIMDRAM ops: %" PRIu64 "\n",
          stat_bank_reads + stat_bank_writes + stat_reads + stat_writes);
  fprintf(stderr, "[simdram]   Dedup hits     : %" PRIu64 " (enabled=%d)\n",
          addr_dedup_hits(g_dedup), g_dedup_enabled);
  fprintf(stderr, "[simdram]   Persistent skips (xyz invariant): %" PRIu64 "\n",
          stat_persistent_skips);
  fprintf(stderr, "[simdram]   Persistent skips (yz invariant) : %" PRIu64 "\n",
          stat_invariant_yz_skips);
  fprintf(stderr, "[simdram]   Persistent skips (xz invariant) : %" PRIu64 "\n",
          stat_invariant_xz_skips);
  fprintf(stderr, "[simdram]   Persistent skips (xy invariant) : %" PRIu64 "\n",
          stat_invariant_xy_skips);
  fprintf(stderr, "[simdram]   Input replication writes        : %" PRIu64 "\n",
          stat_input_replication_writes);
  fprintf(stderr, "[simdram]   Leader-bank skips               : %" PRIu64 " (enabled=%d)\n",
          stat_leader_bank_skips, g_leader_bank_only_enabled);
  fprintf(stderr, "[simdram]   Compute-row dedup skips         : %" PRIu64 " (enabled=%d)\n",
          stat_compute_row_skips, g_compute_row_dedup_enabled);
  fprintf(stderr, "[simdram]   K-amort emissions               : %" PRIu64 " (batch=%d)\n",
          stat_k_amort_emissions, g_k_amort_batch);

  fprintf(stderr,
          "\n[simdram] === Per-tensor breakdown ===\n"
          "[simdram]   tid  role             R          W         BR         BW         WB  "
          "dedup_skips  pers_skips\n");
  for (int i = 0; i < num_tensors; i++) {
    tensor_info_t *t = &tensors[i];
    const char *role_str = (t->role == SIMDRAM_ROLE_STREAMED)  ? "STREAMED"
                           : (t->role == SIMDRAM_ROLE_OPERAND) ? "OPERAND "
                                                               : "ACCUMUL.";
    fprintf(stderr,
            "[simdram]   %3d  %-9s  %10" PRIu64 " %10" PRIu64 " %10" PRIu64
            " %10" PRIu64 " %10" PRIu64 "  %11" PRIu64 "  %10" PRIu64 "\n",
            i, role_str,
            t->emitted_R, t->emitted_W,
            t->emitted_BR, t->emitted_BW, t->emitted_WB,
            t->dedup_skips, t->persistent_skips);
  }

  destroy_dedup_state();
  initialized = 0;
}

/* ================================================================
 *  MemTracePass hooks (memory access instrumentation)
 * ================================================================ */

void __mem_trace_init(void) {
  /* No-op: simdram_init() called explicitly by host driver */
}

void __mem_trace_fini(void) { simdram_finalize(); }

/*
 * SIMDRAM memory access trace mapping (matches OptiPIM SimDRAMCodeGen):
 *
 * COMPUTE phase:
 *   STREAMED (input) load      → pe_bits W ops (write to bit-rows)
 *   OPERAND (weight) load      → nothing (pre-stored in DRAM rows)
 *   ACCUMULATOR (output) load  → pe_bits R ops (read partial sums)
 *   ACCUMULATOR (output) store → nothing (results stay in subarray from MAJ)
 *
 * HOST phase: load → R, store → W
 */
static void simdram_trace_access(uint64_t addr, int is_write) {
  if (cur_phase == SIMDRAM_PHASE_IDLE) {
    stat_ignored++;
    return;
  }

  /* Track program-id transitions so per-pid dedup states reset at the
   * right cadence. The runtime calls __pim_get_program_id* (set by the
   * harness around each kernel launch). */
  advance_program_epoch_if_needed();

  int tidx = find_tensor(addr);
  if (tidx < 0) {
    stat_ignored++;
    return;
  }

  tensor_info_t *t = &tensors[tidx];
  int elem_idx = (int)((addr - (uint64_t)t->base_addr) / t->elem_size);
  if (elem_idx < 0 || elem_idx >= t->num_elements) {
    stat_ignored++;
    return;
  }

  int ch, pch, bg, bank, sa, base_row, col;
  map_element(t, elem_idx, &ch, &pch, &bg, &bank, &sa, &base_row, &col);
  int global_bank = compute_global_bank(ch, pch, bg, bank);

  if (cur_phase == SIMDRAM_PHASE_COMPUTE) {
    /* Leader-bank-only gate. COMPUTE-phase operand R/W happen in
     * lockstep with compute_trace; gate them on the same bank.
     * HOST phase (per-bank input setup) intentionally NOT gated. */
    if (g_leader_bank_only_enabled && __pim_get_bank_id() != 0) {
      stat_leader_bank_skips++;
      return;
    }
    if (!is_write) {
      addr_dedup_state_t *state = pick_load_dedup_state(t);
      switch (t->role) {
      case SIMDRAM_ROLE_STREAMED:
      case SIMDRAM_ROLE_ACCUMULATOR: {
        /* STREAMED: input → write bit-rows.
         * ACCUMULATOR (load): output → read bit-rows.
         *
         * Both are bit-serial expansions over pe_bits row offsets at
         * the same (bank, sa, col). Dedup is keyed by the per-bit
         * physical tuple; the first load of an operand emits all
         * pe_bits ops, a redundant load of the same operand collapses
         * at every bit (each bit's tuple is already in the state).
         *
         * Soundness: STREAMED/ACCUMULATOR loads represent operand
         * ingestion / partial-sum read. They are idempotent re-reads
         * from the same physical row buffer — collapsing them is
         * exactly the row-buffer-locality model PimCodeGen applies
         * via single_bank_opt + first_time_in_col. */
        const int is_acc = (t->role == SIMDRAM_ROLE_ACCUMULATOR);
        for (int bit = 0; bit < cfg_pe_bits; bit++) {
          int row = base_row + bit;
          uint64_t linear_row = (uint64_t)sa * (uint64_t)cfg_num_rows
                                + (uint64_t)row;
          if (state &&
              !addr_dedup_check_and_mark(state, (uint64_t)global_bank,
                                         linear_row, (uint64_t)col)) {
            account_load_dedup_skip(t, state);
            continue;
          }
          if (is_acc) {
            emit_trace("R", ch, pch, bg, bank, sa, row, col);
            stat_reads++; t->emitted_R++;
          } else {
            /* STREAMED input bit-row: emit one SIMDRAM broadcast-write
             * that delivers this bit-row's value to all banks in the
             * (ch, pch) via one channel dispatch. No separate primary
             * W — the broadcast covers the source bank too. */
            emit_input_broadcast(t, ch, pch, bg, bank, sa, row, col);
          }
        }
        if (is_acc && t->acc_tracker_idx >= 0) {
          acc_tracker_t *at = &acc_trackers[t->acc_tracker_idx];
          at->ch = ch; at->pch = pch; at->bg = bg; at->bank = bank;
          at->sa = sa; at->row = base_row;
          current_acc = t->acc_tracker_idx;
        }
        break;
      }
      case SIMDRAM_ROLE_OPERAND:
        /* Weight: pre-stored in DRAM, nothing to emit. */
        break;
      }
    } else {
      /* ACCUMULATOR store. SIMDRAM keeps results in-subarray after MAJ;
       * a store doesn't emit. We just track the location so that the
       * downstream __compute_trace knows where the accumulator lives. */
      if (t->role == SIMDRAM_ROLE_ACCUMULATOR && t->acc_tracker_idx >= 0) {
        acc_tracker_t *at = &acc_trackers[t->acc_tracker_idx];
        at->ch = ch; at->pch = pch; at->bg = bg; at->bank = bank;
        at->sa = sa; at->row = base_row;
        current_acc = t->acc_tracker_idx;
      }
    }
  } else if (cur_phase == SIMDRAM_PHASE_HOST) {
    /* HOST phase: dedup-eligible too. Same scoping as COMPUTE — no
     * special-casing for host R/W (the kernel that issued them runs
     * inside the same pid as its compute counterparts). */
    addr_dedup_state_t *state = pick_load_dedup_state(t);
    uint64_t linear_row = (uint64_t)sa * (uint64_t)cfg_num_rows
                          + (uint64_t)base_row;
    if (state && !is_write &&
        !addr_dedup_check_and_mark(state, (uint64_t)global_bank,
                                   linear_row, (uint64_t)col)) {
      account_load_dedup_skip(t, state);
      return;
    }
    if (!is_write) {
      emit_trace("R", ch, pch, bg, bank, sa, base_row, col);
      stat_reads++; t->emitted_R++;
    } else {
      emit_trace("W", ch, pch, bg, bank, sa, base_row, col);
      stat_writes++; t->emitted_W++;
    }
  }
}

void __mem_trace_load(void *addr, uint64_t size) {
  if (!trace_fp)
    return;
  simdram_trace_access((uint64_t)addr, 0);
}

void __mem_trace_store(void *addr, uint64_t size) {
  if (!trace_fp)
    return;
  simdram_trace_access((uint64_t)addr, 1);
}

/* MemTracePass classifies pointer-chain provenance and routes loads with
 * program-id-invariance to these dedicated entry points. The
 * classification ID lives in cur_load_class for the duration of the
 * single trace_access call below; pick_load_dedup_state consumes it
 * to choose the right dedup scope, then we reset to NORMAL.
 *
 * Caveat documented for future readers: the bit-serial column-lockstep
 * note that previously stubbed these out applies only to MAJ-3 GATE
 * ops (the cost-loop in __compute_trace), NOT to operand-load bit-row
 * reads. Operand re-reads of the same row buffer ARE idempotent and
 * dedup-safe; we route them through the dedup states like HBM-PIM. */
void __pim_load_persistent(void *addr, uint64_t size) {
  cur_load_class = PIM_LOAD_PERSISTENT_XYZ;
  __mem_trace_load(addr, size);
  cur_load_class = PIM_LOAD_NORMAL;
}
void __pim_load_persistent_yz(void *addr, uint64_t size) {
  cur_load_class = PIM_LOAD_INVARIANT_YZ;
  __mem_trace_load(addr, size);
  cur_load_class = PIM_LOAD_NORMAL;
}
void __pim_load_persistent_xz(void *addr, uint64_t size) {
  cur_load_class = PIM_LOAD_INVARIANT_XZ;
  __mem_trace_load(addr, size);
  cur_load_class = PIM_LOAD_NORMAL;
}
void __pim_load_persistent_xy(void *addr, uint64_t size) {
  cur_load_class = PIM_LOAD_INVARIANT_XY;
  __mem_trace_load(addr, size);
  cur_load_class = PIM_LOAD_NORMAL;
}

/* ================================================================
 *  ComputeTracePass hook (arithmetic instrumentation)
 * ================================================================ */

/*
 * Emit MAJ-3 gate operations as bank-read (BR) trace ops.
 * Cycles through pe_bits rows at the current accumulator location
 * (acc_trackers[current_acc]), matching OptiPIM's SimDRAMCodeGen pattern.
 *
 * KNOWN LIMITATION — Implicit accumulator association for non-accumulator ops
 * ─────────────────────────────────────────────────────────────────────────────
 * ComputeTracePass (ComputeTracePass.cpp) instruments every BinaryOperator in
 * LLVM IR with:
 *
 *     __compute_trace(opcode, bit_width)
 *
 * It passes ONLY the opcode and bit-width — NO destination address or SSA
 * value identity.  The runtime therefore cannot determine which accumulator
 * tensor a given arithmetic op feeds; instead, it relies on `current_acc`,
 * which is set by the most recent accumulator load or store in
 * simdram_trace_access().
 *
 * This works correctly for the typical tiled matmul access pattern:
 *
 *     load  acc[i,j]   →  sets current_acc to acc's tracker
 *     mul   a, b       →  BR ops emitted at acc[i,j]'s physical coords  ✓
 *     add   acc, tmp   →  BR ops emitted at acc[i,j]'s physical coords  ✓
 *     store acc[i,j]   →  updates acc's tracker (still correct)
 *
 * It BREAKS when:
 *
 *   1. Multiple accumulators are interleaved at the instruction level
 *      without an intervening accumulator load/store to switch current_acc.
 *      Example (fused kernel writing to both C and D):
 *
 *        load  C[i,j]     →  current_acc = C's tracker
 *        load  D[i,j]     →  current_acc = D's tracker  (overwritten!)
 *        mul   a, b       →  BR ops land at D's coords, but may feed C  ✗
 *
 *   2. Arithmetic ops that produce intermediate values consumed by later
 *      ops (e.g., `t = a * b; acc += t`): the `mul` has no inherent link
 *      to any accumulator.  It is attributed to whichever accumulator was
 *      most recently accessed — correct only if no other accumulator was
 *      touched in between.
 *
 *   3. Non-accumulator arithmetic (e.g., address calculations, loop
 *      induction variables) during COMPUTE phase: these get emitted as
 *      BR ops at the current accumulator location even though they aren't
 *      SIMDRAM in-memory operations.  This causes mild overcounting.
 *
 * POTENTIAL FIX: Extend ComputeTracePass to propagate the destination
 * address through LLVM's def-use chains — for each BinaryOperator, walk
 * the use chain forward to find the eventual StoreInst and pass that
 * store address to __compute_trace as a third argument.  The runtime can
 * then call find_tensor() + map_element() to determine the exact physical
 * coordinates, fully decoupling compute tracing from the load/store order.
 * This requires non-trivial LLVM analysis (phi nodes, select instructions,
 * multi-level GEPs) and is left as future work.
 */
void __compute_trace(int32_t opcode, int32_t bit_width, void *dest_addr) {
  (void)bit_width;
  if (!trace_fp || cur_phase != SIMDRAM_PHASE_COMPUTE)
    return;

  /* Leader-bank-only gate: SIMDRAM compute fires in channel-wide
   * lockstep; emit one bank's stream as the wallclock proxy. */
  if (g_leader_bank_only_enabled && __pim_get_bank_id() != 0) {
    stat_leader_bank_skips++;
    return;
  }

  int ch, pch, bg, bank, sa, base_row, col;
  int resolved = 0;
  int tensor_id = -1;

  /* Primary path: ComputeTracePass passed us the destination address
   * via its forward def-use walk. Resolve it to physical coords. */
  if (dest_addr) {
    int tidx = find_tensor((uint64_t)(uintptr_t)dest_addr);
    if (tidx >= 0) {
      tensor_info_t *t = &tensors[tidx];
      int elem_idx = (int)(((uint64_t)(uintptr_t)dest_addr -
                            (uint64_t)(uintptr_t)t->base_addr) /
                           (uint64_t)t->elem_size);
      if (elem_idx >= 0 && elem_idx < t->num_elements) {
        map_element(t, elem_idx, &ch, &pch, &bg, &bank, &sa, &base_row, &col);
        /* Update the per-tensor accumulator tracker so future
         * fall-back calls (dest_addr==NULL) attribute to the most
         * recently seen cell of this tensor. */
        if (t->acc_tracker_idx >= 0) {
          acc_tracker_t *at = &acc_trackers[t->acc_tracker_idx];
          at->ch = ch; at->pch = pch; at->bg = bg; at->bank = bank;
          at->sa = sa; at->row = base_row;
          current_acc = t->acc_tracker_idx;
        }
        tensor_id = tidx;
        resolved = 1;
      }
    }
  }

  /* Fall-back: ComputeTracePass couldn't determine a destination
   * (BinaryOp's result never reaches a store, or reaches one that
   * doesn't resolve to a registered tensor). Use current_acc — set
   * by the most recent accumulator load/store. */
  if (!resolved) {
    if (current_acc < 0)
      return;
    const acc_tracker_t *at = &acc_trackers[current_acc];
    ch = at->ch; pch = at->pch; bg = at->bg; bank = at->bank;
    sa = at->sa; base_row = at->row; col = 0;
    tensor_id = at->tensor_idx;
  }

  (void)tensor_id; /* reserved for future honest per-tensor stats */

  /* K-axis batch amortization (corrected A2). Count compute_traces
   * globally; emit MAJ-3 cost every g_k_amort_batch traces. This
   * approximates OptiPIM's `temporal_steps × 7n²+1` decomposition
   * without falsely collapsing distinct K iterations into one MAJ-3
   * the way the previous (bank, row) dedup did.
   *
   * BATCH = cfg_num_cols * cfg_dq_bits / cfg_pe_bits represents the
   * number of arithmetic ops one column-parallel TRA amortizes over
   * (cells-per-row × bit-serial depth). Different K-iter compute_traces
   * count toward the next batch; on every BATCH-th trace we emit one
   * MAJ-3 sequence at the current (bank, row). Cells in the same row
   * still amortize within a batch, K-iters across batches do not
   * collapse. */
  int cost;
  if (g_compute_row_dedup_enabled && g_k_amort_batch > 1) {
    g_compute_trace_count++;
    if ((g_compute_trace_count % (uint64_t)g_k_amort_batch) != 1) {
      stat_compute_row_skips++;
      return;
    }
    stat_k_amort_emissions++;
    cost = maj_cost(SIMDRAM_OP_MUL);
  } else if (g_compute_row_dedup_enabled) {
    /* BATCH=1 → emit a MUL's MAJ-3 every compute_trace (no
     * amortization). Same as the original honest per-arithmetic-op
     * baseline. */
    cost = maj_cost(SIMDRAM_OP_MUL);
  } else {
    /* row-dedup disabled → emit per the opcode that fired. */
    cost = maj_cost(opcode);
  }

  /* B-group rows start after D-group + C-group in each subarray. TRA
   * physically operates on B-group rows, not on data rows. */
  int cur_row_offset = 0;
  int b_group_base = cfg_data_rows_per_sa + SIMDRAM_C_GROUP_ROWS;

  tensor_info_t *attribute_t =
      (tensor_id >= 0 && tensor_id < MAX_TENSORS) ? &tensors[tensor_id] : NULL;
  for (int i = 0; i < cost; i++) {
    emit_trace("BR", ch, pch, bg, bank, sa,
               b_group_base + cur_row_offset, 0);
    cur_row_offset = (cur_row_offset + 1) % SIMDRAM_B_GROUP_ROWS;
    stat_bank_reads++;
    if (attribute_t) attribute_t->emitted_BR++;
  }

  stat_compute_ops++;
}
