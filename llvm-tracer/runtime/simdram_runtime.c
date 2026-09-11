#include "simdram_runtime.h"
#include "im_addr_dedup.h"
#include "im_runtime.h"
/* Shared with pim_runtime.c so the two runtimes cannot disagree about the record
 * width. The two are never linked together (pass_ablation.py picks one by target), so
 * this TU must supply the weak fallback definitions itself: on Mach-O a weak REFERENCE
 * with no definition fails to link, which is the normal case whenever
 * im-operand-residency-layout is skipped. */
#define PIM_LAYOUT_TABLE_DEFINE
#include "pim_layout_table.h"
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
/* HAND-COPY of the SIMULATOR's organization, same as pim_runtime.c and with the same
 * absence of a check. The shared SIMDRAM spec names HBM3_8Gb, whose preset has 64
 * subarrays; 16 is the HBM3_2Gb figure and was wrong here too (2026-09-10). Used only
 * for the capacity ceiling and the banner, never in the address slicing. */
static int cfg_num_sa = 64;
static int cfg_num_rows = 512; /* total physical rows per subarray  */
static int cfg_num_cols = 64;
static int cfg_dq_bits = 128;
static void check_compiler_dq_bits(int cfg_bits);
static int64_t simdram_derived_k_amort_batch(void);

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
  uint64_t dedup_skips;    /* (bank, sa·rows+row+bit, col) coalesced */
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
 * Reset cadence: PHASE change only, i.e. the collapse spans the whole kernel.
 *
 * That scope is chosen to match the baseline rather than by preference. OptiPIM's
 * simdram codegen declares its equivalent table before the temporal loop, under the
 * comment "We only need to load input once", and never clears it
 * (simulator/src/pim_codegen/impl/simdram.cpp:110). Resetting per program instance,
 * as this did until 2026-09-10, charged an input element once per instance where the
 * baseline charges it once per kernel. Measured cost of the asymmetry: matmul
 * 128x512x256 STREAMED writes 1,048,576 -> 65,536 and cycles -1.8%, conv 16x16 3x3
 * 65,536 -> 4,096 and cycles -0.2%, single-instance shapes unchanged. The table grows
 * on demand, so the wider scope needs no capacity hint.
 */
static addr_dedup_state_t *g_dedup = NULL;
static int g_dedup_enabled = 1;

/* K-AMORTIZATION BATCHING: one MAJ block emitted every g_k_amort_batch compute traces.
 * RENAMED 2026-09-09 from g_compute_row_dedup_enabled / PIM_ROW_COMPUTE_DEDUP.
 * benchmarks/run_runtime_ablation.py was updated with it. It was named after an addr_dedup table
 * that was created, reset, destroyed and printed but NEVER QUERIED, and that dead table
 * has been deleted. The old name cost real time: it reads as a second, separate
 * amortization stacked on this one, and the 128x/8x ablation ratios were briefly taken
 * as evidence of exactly that. They are just BATCH. What runs is the per-opcode
 * modulo counter below: one MAJ block emitted every g_k_amort_batch compute traces.
 *
 * PIM_K_AMORT=0 therefore disables k-amortization entirely (every arithmetic
 * op fires its full maj_cost), which is why turning it off costs 128x on matmul
 * 64x64x64 and 8x on conv 3x3 -- those ratios are exactly BATCH, not evidence of a
 * separate mechanism.
 *
 * CROSS-PID PERSISTENCE IS WORTH ZERO, contrary to the comment that used to sit here.
 * The counter resets only on a phase change so it does span the grid, but BATCH is the
 * per-pid per-pass lane count, so every batch boundary lands on a pid boundary and
 * persistent equals per-pid-reset exactly. VERIFIED by arithmetic on two shapes:
 * matmul 64x64x64 emits 123,008/1922 = 64 blocks = K with grid 1x1 (pids=1, so
 * persistence cannot fire at all), and conv 16x16x16x16 3x3 emits 4,428,288/1922 =
 * 2,304 = 16 pids x 144 reduction steps, i.e. NO cross-pid collapse occurred. The
 * runtime sits in the pids x depth x maj_cost branch its old comment claimed to avoid.
 *
 * NO DOUBLE-COUNTING with the leader-bank gate: that gate fires BEFORE this counter
 * increments, so the counter sees bank 0 only, and BATCH is denominated in cells PER
 * BANK. Numerator and denominator each divide by 32 once. */
static int g_k_amort_enabled = 1;
static uint64_t stat_k_amort_skips = 0;

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
 * Set PIM_K_AMORT=0 for the no-amortization
 * baseline (every compute_trace emits its full maj_cost). */
static int g_k_amort_batch = 0;
static uint64_t g_compute_trace_count = 0;
/* Per-opcode batch counters: a MUL and an ADD amortize independently because
 * their MAJ-3 costs differ by ~14x. */
static uint64_t g_compute_trace_count_by_op[SIMDRAM_OP_COUNT] = {0};
static uint64_t stat_k_amort_emissions = 0;

/* Count of per-bank input-replication writes.
 *
 * SIMDRAM compute needs an input's bit-rows physically resident in each consuming
 * bank's subarray, so one input bit-row costs one regular write per receiving bank.
 * That matches OptiPIM's simdram codegen, which pays a full write per receiving bank
 * (simdram.cpp:46-49).
 *
 * The SIMDRAM paper's BCAST_W extension would make this one pseudochannel dispatch
 * instead. It was implemented behind PIM_BCAST_W, defaulted off on 2026-05-27 for
 * symmetry with the baseline, never turned on, and deleted 2026-09-10. Turning it on
 * would have collapsed the fanout to a single event and taken us below a cost the
 * baseline pays. */
static uint64_t stat_input_replication_writes = 0;

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
 * PIM_LANE_MODEL selects the LANE MODEL as a whole, and replaced the old standalone
 * PIM_LEADER_BANK_ONLY on 2026-09-10. The gate and the occupancy denominator are two
 * halves of one convention and must move together: the gate keeps 1 lane of 32 while
 * the denominator divides the output tile by 32 banks, so setting either alone shifts
 * the charge by exactly 32x and measures nothing. One knob, self-consistent values.
 *
 *   leader  (default) gate ON,  denominator = output tile / banks.  What ships.
 *   paired            gate OFF, denominator = output tile.          Both /32 removed.
 *
 * A third model keyed on the PHYSICAL bank (simdram.cpp's own rule: keep the elements
 * whose local_bank_id is 0, break) was built and REFUTED on 2026-09-11. Their key
 * discounts only because they never replay; ours replays the kernel once per lane, so
 * swapping the key without also fixing the replay left all 32 lanes alive (+30.5x bank
 * reads, W=0, geomean over 6 shapes) -- for these tile widths every lane's elements
 * land in bank 0. Their gate needs their walk; see MECHANISM_CHECKLIST.
 *
 * `paired` is the diagnostic: if the gate really only drops replicas of lane 0's work
 * then the two arms emit the SAME bank reads, and if they differ the gate is dropping
 * distinct work. It keys on __pim_get_bank_id(), the harness replay lane, while every
 * physical bank in the trace comes from map_element(), which reads the element index
 * and never the lane, so the prediction is genuinely at risk. */
typedef enum { SIMDRAM_LANE_LEADER = 0, SIMDRAM_LANE_PAIRED = 1 } simdram_lane_model_t;
static simdram_lane_model_t g_lane_model = SIMDRAM_LANE_LEADER;
static int g_leader_bank_only_enabled = 1;
static uint64_t stat_leader_bank_skips = 0;

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

/* Emit one regular write per receiving bank: the input bit-row has to land in every
 * bank that will compute on it. See the note above for why this is a fanout and not
 * a broadcast.
 *
 * ACROSS BOTH PSEUDOCHANNELS since 2026-09-10, because the address map now spreads a
 * tile over all 32 banks and an operand in pch 0 is not visible to a consumer in
 * pch 1. An all-bank PIM command reaches the banks of ONE pseudochannel, so this is
 * two dispatches rather than one, and it costs twice the writes. That is the honest
 * price of using the whole 32-bank machine we already grant the baseline, and
 * pretending one dispatch covered both is exactly the BCAST_W mistake. src_pch is
 * therefore ignored: every consuming bank is written, wherever the source sat. */
static void emit_input_broadcast(tensor_info_t *t,
                                 int src_ch, int src_pch,
                                 int src_bg, int src_bank,
                                 int sa, int row, int col) {
  (void)src_pch;
  (void)src_bg;
  (void)src_bank;
  for (int pch = 0; pch < cfg_num_pch; pch++) {
    for (int bg = 0; bg < cfg_num_bg; bg++) {
      for (int bk = 0; bk < cfg_num_banks; bk++) {
        emit_trace("W", src_ch, pch, bg, bk, sa, row, col);
        stat_writes++;
        t->emitted_W++;
        stat_input_replication_writes++;
      }
    }
  }
}

/* The compiler bakes a bus width into every vector width it picks. Ours must match,
 * or the artifact was tuned for a machine this run is not modelling. */
static void check_compiler_dq_bits(int cfg_bits) {
  if (__pim_dq_bits && __pim_dq_bits != cfg_bits)
    fprintf(stderr,
            "[%s] WARN: bus-width disagreement. The compiler chose vector widths for "
            "a %d-bit bus; this run models %d-bit. Vector accesses are priced against "
            "a width the artifact never assumed.\n",
            "simdram", (int)__pim_dq_bits, cfg_bits);
}

static void destroy_dedup_state(void) {
  if (g_dedup)               { addr_dedup_destroy(g_dedup);              g_dedup = NULL; }
}

/* NULL when dedup is disabled (PIM_DEDUP=0 ablation). */
static addr_dedup_state_t *pick_load_dedup_state(const tensor_info_t *t) {
  (void)t;
  return g_dedup_enabled ? g_dedup : NULL;
}

static void account_load_dedup_skip(tensor_info_t *t,
                                    const addr_dedup_state_t *which) {
  (void)which;
  t->dedup_skips++;
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

  /* PSEUDOCHANNEL, added 2026-09-10. The map used to pin pch=0, so a tile spread
   * over 16 banks while the kernel was compiled for 32 lanes and the occupancy
   * divisor divided by 32. Three numbers that had to agree and did not, and we were
   * granting OptiPIM 32 banks through the num_banks attribute while placing our own
   * data in half that. Above bg and below col, so consecutive elements fill all 32
   * banks before advancing a column. */
  int log2_pch = ilog2_pow2(cfg_num_pch);
  *pch = (int)(linear & ((1ULL << log2_pch) - 1));
  linear >>= log2_pch;

  *col = (int)(linear & ((1ULL << log2_cols) - 1));
  linear >>= log2_cols;

  /* Remaining bits = row_group index within the bank's SAs. Each
   * row_group occupies pe_bits consecutive rows; advance the linear
   * row counter accordingly. */
  uint64_t linear_row = linear * (uint64_t)cfg_pe_bits;
  *sa = (int)(linear_row / (uint64_t)cfg_data_rows_per_sa);
  *base_row = (int)(linear_row % (uint64_t)cfg_data_rows_per_sa);

  /* One channel still. Spreading across channels would need the input fanout to
   * cross a channel boundary, which no PIM command does. */
  *ch = 0;
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
static int g_warned_unpriced_op = 0;

static int maj_cost(int opcode) {
  int n = cfg_pe_bits;
  switch (opcode) {
  /* FLOAT OPS SHARE THE FIXED-POINT POLYNOMIALS. SIMDRAM is bit-serial
   * processing-using-memory: a MAJ-3 gate count over n bit-planes, with no
   * mantissa or exponent hardware anywhere in this model. An fp16 kernel is
   * therefore priced as 16-bit FIXED POINT, which is what OptiPIM does too (one
   * dataWidth field, no float path, m_pe_bits used for every tensor). Charging
   * them identically is the symmetric choice.
   * They used to fall to the default and cost ZERO. That silently deleted the
   * entire arithmetic charge the moment kernels moved to fp16: matmul_64x64x64 on
   * SIMDRAM read 3,476,951 cycles at int32 and 32,783 at fp16, a 106x free
   * speedup that was pure metering loss. Caught by the invariants golden
   * 2026-09-09. */
  case SIMDRAM_OP_MUL:
  case SIMDRAM_OP_FMUL:
    return g_optipim_mul_formula ? (7 * n * n + 1)
                                 : (11 * n * n - 5 * n - 1);
  case SIMDRAM_OP_ADD:
  case SIMDRAM_OP_SUB:
  case SIMDRAM_OP_FADD:
  case SIMDRAM_OP_FSUB:
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
  case SIMDRAM_OP_FDIV:
  case SIMDRAM_OP_FREM:
    return 8 * n * n + 12 * n;
  default:
    /* Genuinely unmodelled opcode. Returning 0 here is how the float ops silently
     * went free, so make it LOUD rather than silent. */
    if (!g_warned_unpriced_op) {
      g_warned_unpriced_op = 1;
      fprintf(stderr, "[simdram] WARNING: opcode %d has no MAJ cost and is being "
                      "charged ZERO. Any cycle count including it understates the "
                      "arithmetic.\n", opcode);
    }
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
  check_compiler_dq_bits(cfg_dq_bits);
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
  /* Compiler decision first, env second and announced. Mirrors pim_runtime.c: the
   * scheme used to be a runtime default with only an env var to change it, so the
   * artifact had no say in its own placement. 1 = the legacy pinned scheme here,
   * 2 = interleaved, 0 = the kernel said nothing. */
  cfg_layout_scheme = SIMDRAM_LAYOUT_INTERLEAVED;
  if (__pim_layout_scheme == 1)
    cfg_layout_scheme = SIMDRAM_LAYOUT_PINNED;
  const char *layout_env = getenv("PIM_SIMDRAM_LAYOUT");
  if (layout_env) {
    simdram_layout_scheme_t want = (layout_env[0] == 'p')
                                       ? SIMDRAM_LAYOUT_PINNED
                                       : SIMDRAM_LAYOUT_INTERLEAVED;
    if (__pim_layout_scheme && want != cfg_layout_scheme)
      fprintf(stderr,
              "[simdram] WARN: PIM_SIMDRAM_LAYOUT=%s overrides the compiler's "
              "placement scheme. Addresses no longer follow the artifact.\n",
              layout_env);
    cfg_layout_scheme = want;
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
   * single_bank_opt). See the PIM_LANE_MODEL note at the declaration: the gate and the
   * occupancy denominator are one convention and this knob moves both. */
  const char *lane_env = getenv("PIM_LANE_MODEL");
  if (lane_env && strcmp(lane_env, "paired") == 0) {
    g_lane_model = SIMDRAM_LANE_PAIRED;
  } else if (lane_env && strcmp(lane_env, "leader") != 0) {
    fprintf(stderr,
            "[simdram] ERROR: PIM_LANE_MODEL=%s; expected 'leader' or 'paired'.\n",
            lane_env);
    exit(1);
  }
  g_leader_bank_only_enabled = (g_lane_model == SIMDRAM_LANE_LEADER) ? 1 : 0;
  stat_leader_bank_skips = 0;
  fprintf(stderr,
          "[simdram] lane_model=%s (leader_bank_only=%d; set PIM_LANE_MODEL=paired to "
          "drop the gate AND the matching /banks together)\n",
          g_lane_model == SIMDRAM_LANE_PAIRED ? "paired" : "leader",
          g_leader_bank_only_enabled);

  if (g_dedup_enabled) {
    const int perpid_cap = 4096;
    g_dedup              = addr_dedup_create(perpid_cap);
  }

  /* Compute-row residency (corrected Lever 2). Defaults ON — models
   * SIMDRAM's column-parallel TRA so per-row compute cost matches
   * OptiPIM's per-temporal-step emission. PIM_K_AMORT=0
   * disables (ablation: every arithmetic op fires its full maj_cost). */
  const char *k_amort_env = getenv("PIM_K_AMORT");
  g_k_amort_enabled = (k_amort_env && k_amort_env[0] == '0') ? 0 : 1;
  stat_k_amort_skips = 0;
  fprintf(stderr,
          "[simdram] k_amort=%d (set PIM_K_AMORT=0 to charge every arithmetic op in full)\n",
          g_k_amort_enabled);

  /* K-axis batch amortization: emit one MAJ-3 cost block every BATCH
   * compute_traces of the same opcode.
   *
   * BATCH = cfg_num_cols * cfg_dq_bits = bitlines in one open row = the number
   * of independent VALUES one triple-row-activate pass covers. NOT divided by
   * pe_bits: maj_cost already sweeps all n bit-planes (7n^2+1), so dividing here
   * would charge n twice. Identical to OptiPIM's n_values_per_row =
   * count[column]*dq (simdram.cpp:88-89), whose row_offset advances pe_bits rows
   * per 8192 values. (Was /cfg_pe_bits = 512 until 2026-09-05, a 16x
   * under-amortization that made us charge ~9x OptiPIM for the same conv.)
   * This is only the FALLBACK. The compiler's own figure overrides it below, and
   * PIM_K_AMORT_BATCH was deleted 2026-09-10 with the rest of the caller's authority
   * over the divisor: a knob on the denominator is a knob on the answer. PIM_K_AMORT
   * still turns amortization off wholesale, which is a real ablation. */
  g_k_amort_batch = cfg_num_cols * cfg_dq_bits;
  if (g_k_amort_batch < 1) g_k_amort_batch = 1;
  g_compute_trace_count = 0;
  memset(g_compute_trace_count_by_op, 0, sizeof(g_compute_trace_count_by_op));
  stat_k_amort_emissions = 0;
  fprintf(stderr,
          "[simdram] k_amort_batch=%d (fallback; the artifact overrides it below)\n",
          g_k_amort_batch);
  /* THE COMPILER'S VALUE WINS. Whatever the caller passed is an input to be checked,
   * not the answer: the occupancy is a property of the kernel's own output tile, and
   * a divisor the caller picks is a divisor on the cost of the thing being measured
   * (forcing 4096 in place of 8 moved one conv's bank reads 2,214,144 -> 5,766).
   * Agreed with the harness on 18 of 18 benchmarked shapes when this landed, so the
   * override is silent on everything we report; where it is not silent it says so.
   *
   * No footprint is the one case still at the caller's mercy, so THAT warns and
   * invariant 6 turns it into a failed run. */
  {
    int64_t derived = simdram_derived_k_amort_batch();
    if (derived > 0) {
      if (derived != (int64_t)g_k_amort_batch)
        fprintf(stderr,
                "[simdram] k_amort_batch %d -> %lld, taking the compiler's output "
                "footprint over the value in force.\n",
                g_k_amort_batch, (long long)derived);
      g_k_amort_batch = (int)derived;
    } else if ((int)__pim_layout_count > 0) {
      fprintf(stderr,
              "[simdram] WARN no store footprint in the artifact, so k_amort_batch=%d "
              "is whatever the caller supplied and the occupancy charge is unverified.\n",
              g_k_amort_batch);
    }
  }

  fprintf(stderr, "[simdram] Initialized. Trace: %s\n", trace_file);
  fprintf(stderr,
          "[simdram] HBM config: %d ch, %d pch, %d bg, %d banks/bg, "
          "%d sa, %d rows/sa (%d data + %d reserved), %d cols, "
          "%d-bit DQ, pe_bits=%d\n",
          cfg_num_channels, cfg_num_pch, cfg_num_bg, cfg_num_banks, cfg_num_sa,
          cfg_num_rows, cfg_data_rows_per_sa, SIMDRAM_RESERVED_ROWS,
          cfg_num_cols, cfg_dq_bits, cfg_pe_bits);
}

/* Does the compiler's layout analysis agree with the host's role string?
 *
 * REPORTS, never overrides. On HBM the same word drives the role, because there a
 * mislabel changes the charge (a replicated operand landed in the free role and
 * measured 31x too fast). Here the mapping is different: SIMDRAM's OPERAND emits
 * nothing by design and symmetrically with OptiPIM, whose simdram codegen also emits
 * nothing for the weight. So overriding the role from bank_replicated would invent a
 * cost the baseline does not charge. What SIMDRAM lacked was the ALARM, not the
 * override, so this is the alarm on its own.
 *
 * PIM_STRICT_ROLES=1 turns a disagreement into an abort. It must stay silent on every
 * benchmarked kernel; if it ever fires, the host tag and the layout have diverged and
 * the SIMDRAM numbers for that shape are describing a mapping nobody chose.
 */
/* Occupancy the COMPILER planned for, read out of the artifact.
 *
 * The charge divides by the live output tile per bank, and until now that number was
 * computed in Python and pushed in through PIM_K_AMORT_BATCH, which let the caller
 * choose the divisor on the cost of the thing being measured. The same quantity is
 * already in the kernel: a STORE's address footprint is one instance's output and
 * excludes the reduction axis by construction, which a load's footprint would not.
 *
 * REPORTS ONLY. Returns 0 when the kernel carries no table, no store record or no
 * footprint, and the caller then leaves the value in force alone. */
static int64_t simdram_derived_k_amort_batch(void) {
  if ((int)__pim_layout_count <= 0)
    return 0;
  int emitted = (int)__pim_layout_rec_words;
  if (emitted > 0 && emitted != PIM_LAYOUT_REC_WORDS)
    return 0; /* width drift; the loud abort for that lives in the role check */
  /* The gate's half of the pair. Under `leader` one lane of `banks` survives and the
   * tile is divided by `banks`; under `paired` every lane emits and the division comes
   * off with it. Reading g_lane_model here is what keeps the two from drifting. */
  int all_banks = cfg_num_pch * cfg_num_bg * cfg_num_banks;
  if (all_banks < 1) all_banks = 1;
  int banks = (g_lane_model == SIMDRAM_LANE_PAIRED) ? 1 : all_banks;
  int64_t row_values = __pim_row_values ? (int64_t)__pim_row_values
                                        : (int64_t)cfg_num_cols * cfg_dq_bits;
  if (row_values < 1) row_values = 1;
  /* THE CAP SCALES WITH THE DIVISOR, and forgetting that made the first paired sweep
   * lie. The row is a PER-BANK ceiling, so dropping the /banks has to raise it by the
   * same factor or the cap binds in one arm and not the other. Measured before the
   * fix: matmul 128x512x256 (tile 65,536) capped at 8,192 in paired against 2,048 in
   * leader, a ratio of 4 where 32 was owed, and bank reads moved 8x. That 8x was the
   * instrument, not the gate. Shapes under one row were unaffected either way. */
  if (g_lane_model == SIMDRAM_LANE_PAIRED)
    row_values *= all_banks;
  int64_t best = 0;
  for (int i = 0; i < (int)__pim_layout_count; i++) {
    const int32_t *rec = __pim_layout_table + (size_t)i * PIM_LAYOUT_REC_WORDS;
    if (rec[PIM_LW_IS_STORE] != 1)
      continue;
    int naxes = rec[PIM_LW_NUM_AXES];
    if (naxes <= 0 || naxes > PIM_MAX_FP_AXES)
      continue;
    int64_t cells = 1;
    for (int a = 0; a < naxes; a++)
      cells *= (int64_t)rec[PIM_LW_AXES_BASE + a * PIM_FP_AXIS_WORDS];
    /* Scale by whatever a rank-collapsing reduce hid from the store. A split-K
     * kernel keeps SPLIT_K partial sums live per output and sums them once at the
     * end, so the store sees BLOCK while the row holds BLOCK*SPLIT_K. Without this
     * the occupancy is frozen at the output width and the lever measures nothing:
     * on matvec 512x64 the derived batch read 16 at every SPLIT_K from 1 to 64.
     * 1 on every kernel whose stored value IS its accumulator, so this multiply is
     * the identity everywhere else. */
    int64_t split = rec[PIM_LW_LIVE_SPLIT];
    if (split < 1) split = 1;   /* absent or malformed reads as no collapse */
    cells *= split;
    int64_t occ = cells / banks;
    if (occ < 1) occ = 1;
    if (occ > row_values) occ = row_values;
    /* Several stores: the widest owes the passes, since each is charged its own. */
    if (occ > best) best = occ;
  }
  return best;
}

static void simdram_check_compiler_role(int tensor_id, simdram_role_t host_role) {
  /* Same cross-submodule width check as the HBM runtime: this TU strides the table
   * too, so it can misread it the same way. */
  static int width_checked = 0;
  if (!width_checked) {
    width_checked = 1;
    int emitted = (int)__pim_layout_rec_words;
    if (emitted > 0 && emitted != PIM_LAYOUT_REC_WORDS) {
      fprintf(stderr,
              "[simdram] FATAL: layout record width mismatch, compiler emitted %d, "
              "runtime strides %d. Submodules have drifted; rebuild both.\n",
              emitted, PIM_LAYOUT_REC_WORDS);
      abort();
    }
  }
  int n = (int)__pim_layout_count;
  if (n <= 0)
    return; /* weak fallback: this kernel carries no descriptor */
  for (int i = 0; i < n; i++) {
    const int32_t *rec = __pim_layout_table + (size_t)i * PIM_LAYOUT_REC_WORDS;
    if (rec[PIM_LW_OPERAND_ARG] != tensor_id)
      continue;
    int32_t br = rec[PIM_LW_BANK_REPLICATED];
    if (br < 0 || host_role == SIMDRAM_ROLE_ACCUMULATOR)
      return; /* pass said nothing, or a store whose role is what it is */
    /* bank_replicated=1 means every bank sees the same elements, which is what
     * OPERAND asserts here; 0 means bank-partitioned, which is STREAMED. */
    simdram_role_t derived = br ? SIMDRAM_ROLE_OPERAND : SIMDRAM_ROLE_STREAMED;
    if (derived == host_role)
      return;
    /* REPORTS, never overrides, and that asymmetry with pim_runtime.c is deliberate:
     * see the rationale above. Considered making both targets override on 2026-09-10
     * and reverted, because SIMDRAM's OPERAND emits nothing by design, symmetrically
     * with OptiPIM's simdram codegen, so taking the derived role here could invent a
     * cost the baseline never charges. Same word, different meaning per target.
     * The line is a WARN so invariant 6 turns it into a failure rather than a note
     * nobody reads. */
    const char *hs = (host_role == SIMDRAM_ROLE_OPERAND) ? "OPERAND" : "STREAMED";
    const char *ds = (derived == SIMDRAM_ROLE_OPERAND) ? "OPERAND" : "STREAMED";
    fprintf(stderr,
            "[simdram] WARN ROLE DISAGREEMENT tensor %d: host says %s, compiler "
            "layout says %s (bank_replicated=%d). SIMDRAM charges these differently "
            "(OPERAND is free), so this shape's cost is suspect.\n",
            tensor_id, hs, ds, (int)br);
    const char *strict = getenv("PIM_STRICT_ROLES");
    if (strict && strict[0] == '1') {
      fprintf(stderr, "[simdram] PIM_STRICT_ROLES=1, aborting.\n");
      abort();
    }
    return;
  }
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
  simdram_check_compiler_role(num_tensors, role);
  t->emitted_R = t->emitted_W = 0;
  t->emitted_BR = t->emitted_BW = 0;
  t->dedup_skips = 0;

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
  /* Phase change resets the dedup state. */
  if (cur_phase != phase) {
    if (g_dedup)              addr_dedup_reset(g_dedup);
    g_compute_trace_count = 0;
    memset(g_compute_trace_count_by_op, 0, sizeof(g_compute_trace_count_by_op));
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
  if (addr_dedup_saturations(g_dedup))
    fprintf(stderr,
            "[simdram]   WARN dedup table saturated      : %" PRIu64
            " (collapse stopped; trace over-emits)\n",
            addr_dedup_saturations(g_dedup));
  fprintf(stderr, "[simdram]   Input replication writes        : %" PRIu64 "\n",
          stat_input_replication_writes);
  fprintf(stderr, "[simdram]   Leader-bank skips               : %" PRIu64 " (enabled=%d)\n",
          stat_leader_bank_skips, g_leader_bank_only_enabled);
  fprintf(stderr, "[simdram]   K-amort skips                   : %" PRIu64 " (enabled=%d)\n",
          stat_k_amort_skips, g_k_amort_enabled);
  fprintf(stderr, "[simdram]   K-amort emissions               : %" PRIu64 " (batch=%d)\n",
          stat_k_amort_emissions, g_k_amort_batch);

  fprintf(stderr,
          "\n[simdram] === Per-tensor breakdown ===\n"
          "[simdram]   tid  role             R          W         BR         BW  "
          "dedup_skips\n");
  for (int i = 0; i < num_tensors; i++) {
    tensor_info_t *t = &tensors[i];
    const char *role_str = (t->role == SIMDRAM_ROLE_STREAMED)  ? "STREAMED"
                           : (t->role == SIMDRAM_ROLE_OPERAND) ? "OPERAND "
                                                               : "ACCUMUL.";
    fprintf(stderr,
            "[simdram]   %3d  %-9s  %10" PRIu64 " %10" PRIu64 " %10" PRIu64
            " %10" PRIu64 "  %11" PRIu64 "\n",
            i, role_str,
            t->emitted_R, t->emitted_W,
            t->emitted_BR, t->emitted_BW,
            t->dedup_skips);
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

/* SIMDRAM prices one bit-serial expansion per ELEMENT and discards both the byte
 * size and the lane count: a vector access is the same expansion repeated, and the
 * host replay hands us one element per call. Accepted so the ABI matches the pass
 * and the HBM runtime. */
void __mem_trace_load(void *addr, uint64_t size, uint64_t lanes) {
  (void)size; (void)lanes;
  if (!trace_fp)
    return;
  simdram_trace_access((uint64_t)addr, 0);
}

void __mem_trace_store(void *addr, uint64_t size, uint64_t lanes) {
  (void)size; (void)lanes;
  if (!trace_fp)
    return;
  simdram_trace_access((uint64_t)addr, 1);
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
  /* Amortize PER OPCODE: a MUL costs 7n^2+1 but an ADD only 8n+1, so batching
   * them on one counter and always charging MUL over-charges every accumulate
   * (half the instrumented sites) by ~14x. */
  int op_slot = (opcode >= 0 && opcode < SIMDRAM_OP_COUNT) ? opcode
                                                           : SIMDRAM_OP_OTHER;
  if (g_k_amort_enabled && g_k_amort_batch > 1) {
    g_compute_trace_count_by_op[op_slot]++;
    if ((g_compute_trace_count_by_op[op_slot] % (uint64_t)g_k_amort_batch) != 1) {
      stat_k_amort_skips++;
      return;
    }
    g_compute_trace_count++;
    stat_k_amort_emissions++;
    cost = maj_cost(opcode);
  } else if (g_k_amort_enabled) {
    /* BATCH=1 → emit this opcode's MAJ-3 every compute_trace (no
     * amortization). The honest per-arithmetic-op baseline. */
    cost = maj_cost(opcode);
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
