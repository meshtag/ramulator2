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
 * Accumulator read-out is emitted once per (lane, cell): a loop-carried value is
 * resident in its own bank after the first touch, and the slot's marker bit says so.
 * That is residency the compiler states, not an address-collapse heuristic.
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
static void flush_all_lanes(void);

static int cfg_pe_bits = 16;

/* Placement is lane placement, nothing else: a tensor the compiler stamped partitioned
 * or an accumulator lives in the executing lane's own bank, in row groups claimed as
 * its slab grows; an OPERAND is pre-stored and emits nothing. The interleaved and
 * pinned address maps that used to decide banks from element bits were deleted on
 * 2026-09-12 once no measured path read them. */


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
#define SIMDRAM_MAX_LANES 64

typedef struct {
  void *base_addr;
  size_t total_bytes;
  int elem_size;
  int dims[4];
  int num_elements;
  simdram_role_t role;

  int acc_tracker_idx;     /* index into acc_trackers[] or -1   */
  /* Compiler's partition bit from the layout table: 1 replicated, 0 bank-partitioned,
   * -1 not stated. Only 0 changes anything: that tensor is delivered to the executing
   * lane's own bank instead of fanned out, see note_lane_consumption. */
  int bank_replicated;
  /* Row groups of this tensor's per-lane slab, the same in every bank, CLAIMED AS THE
   * SLAB GROWS from a per-bank pool at the top of the row space. No size is guessed:
   * how much a lane touches depends on the launch grid, which no compile-time number
   * knows, and the old "twice the even share" overflowed on conv halos. */
  int *lane_rg;
  int lane_rg_n, lane_rg_cap;
  int lane_ord[SIMDRAM_MAX_LANES]; /* next free value slot per lane */
  int lane_flushed[SIMDRAM_MAX_LANES]; /* VALUES already written out, per lane */
  slot_map_t *lane_slot;   /* (lane, elem) -> slot; partitioned STREAMED and ACCUMULATOR */
  /* Cells in this lane that one CONSUMED value feeds through a tile broadcast, from the
   * compiler. Sharing that shows up as repeated loads is counted by the loads themselves;
   * this covers only the sharing that never reaches memory, such as matmul's b[k,n] being
   * multiplied into every row of the tile. 1 until the artifact says otherwise. */
  int cell_fanout;

  /* Per-tensor DRAM-opcode counters. Each emission in the trace file is
   * attributed to exactly one tensor by the call site (compute_trace BRs
   * resolve via dest_addr → acc tracker → tensor_idx; load/store ops
   * resolve via find_tensor(addr) at the access site). */
  uint64_t emitted_R;      /* regular DRAM read  (R)  */
  uint64_t emitted_W;      /* regular DRAM write (W)  */
  uint64_t emitted_BR;     /* bank-read MAJ-3 op (BR) */
  uint64_t dedup_skips;    /* re-touches of a resident (lane, cell): read out once */
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

/* The whole-kernel load dedup that lived here is gone (2026-09-12): delivery and
 * read-out are per (lane, element) slot, which is residency the compiler states
 * (partitioned load, loop-carried accumulator), not an address heuristic. */

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
 * kernel. With the gate on, only the bank_id == 0 iteration emits the
 * gated COMPUTE-phase ops; bank_id 1..N skip. HOST-phase accesses have
 * no placement model since 2026-09-12 and are ignored with a WARN.
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
 * The gate covers compute, partial-sum readout and the delivery of REPLICATED or
 * unstated inputs. A bank-PARTITIONED input (compiler bank_replicated == 0) bypasses it
 * since 2026-09-12: every lane places its own elements in its own bank, see
 * note_lane_consumption and flush_all_lanes, so there is nothing for the gate to
 * sample there and no fanout to cancel against it.
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
 * physical bank in the trace now comes from lane placement, so the two arms differ
 * only by the gate and the divisor. */
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

static uint64_t stat_lane_placed_writes = 0;
/* Per-bank pool of row groups for lane slabs, claimed downward from the top of the row
 * space. Nothing else places rows on SIMDRAM any more (OPERAND emits nothing, STREAMED
 * and ACCUMULATOR are lane-placed), so there is no allocator below to collide with. */
static int g_lane_rg_top = -1;
static uint64_t stat_lane_rg_claimed = 0;

/* Physical row group for slab row group `idx` of tensor t, claiming from the top of the
 * row space on first use. Returns -1 once the pool is exhausted (WARNs once). */
static int lane_rg_for(tensor_info_t *t, int idx) {
  static int warned = 0;
  while (idx >= t->lane_rg_n) {
    if (t->lane_rg_n == t->lane_rg_cap) {
      int ncap = t->lane_rg_cap ? t->lane_rg_cap * 2 : 8;
      int *n = (int *)realloc(t->lane_rg, (size_t)ncap * sizeof(int));
      if (!n) return -1;
      t->lane_rg = n; t->lane_rg_cap = ncap;
    }
    if (g_lane_rg_top < 0)
      g_lane_rg_top = (cfg_num_sa * cfg_data_rows_per_sa) / cfg_pe_bits;
    if (g_lane_rg_top <= 0) {
      if (!warned++)
        fprintf(stderr, "[simdram] WARN row space exhausted while growing a lane slab; "
                        "placement beyond this point aliases row group 0.\n");
      t->lane_rg[t->lane_rg_n++] = 0;
      continue;
    }
    t->lane_rg[t->lane_rg_n++] = --g_lane_rg_top;
    stat_lane_rg_claimed++;
  }
  return t->lane_rg[idx];
}
extern uint64_t __pim_program_epoch; /* im_runtime.c, bumped per set_program_id */
static uint64_t g_last_epoch = 0;


/* Flat bank id -> (ch, pch, bg, bank), bank innermost. A replay lane in 0..31 is a
 * flat bank id, so this is how "lane b" becomes "bank b". */
static void decompose_global_bank(int g, int *ch, int *pch, int *bg, int *bank) {
  *bank = g % cfg_num_banks; g /= cfg_num_banks;
  *bg = g % cfg_num_bg;      g /= cfg_num_bg;
  *pch = g % cfg_num_pch;    g /= cfg_num_pch;
  *ch = g;
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
  stat_bank_reads = stat_bank_writes = stat_reads = stat_writes = 0;
  stat_compute_ops = stat_ignored = 0;
  num_acc_trackers = 0;
  current_acc = -1;
  initialized = 1;
  finalized = 0;

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
  if (__pim_lanes > 0 && (int)__pim_lanes != cfg_num_pch * cfg_num_bg * cfg_num_banks)
    fprintf(stderr, "[simdram] WARN kernel compiled for %d lanes, machine has %d banks; "
                    "the occupancy divisor follows the kernel.\n",
            (int)__pim_lanes, cfg_num_pch * cfg_num_bg * cfg_num_banks);
  stat_leader_bank_skips = 0;
  stat_lane_placed_writes = 0;
  stat_lane_rg_claimed = 0;
  g_lane_rg_top = -1; /* fresh row space per run */
  g_last_epoch = __pim_program_epoch;
  fprintf(stderr,
          "[simdram] lane_model=%s (leader_bank_only=%d; set PIM_LANE_MODEL=paired to "
          "drop the gate AND the matching /banks together)\n",
          g_lane_model == SIMDRAM_LANE_PAIRED ? "paired" : "leader",
          g_leader_bank_only_enabled);


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
    } else {
      /* Gating this on layout_count > 0 made the NO-TABLE case the quiet one, where the
       * 8192 fallback stands and the occupancy charge is 64x too cheap (matmul 64x64x64
       * BR 123,008 -> 1,922). A pass ablation that skips the residency pass then reads
       * that as the pass's own win. Review 2026-09-12. */
      fprintf(stderr,
              "[simdram] WARN no store footprint in the artifact%s, so k_amort_batch=%d "
              "is the fallback and the occupancy charge is unverified.\n",
              (int)__pim_layout_count > 0 ? "" : " (no layout table at all)",
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
  /* The divisor is the COMPILER'S lane count when it travels: the gate keeps one lane
   * of that many, so the same number must divide the tile. The configured bank count
   * is checked against it at init, not silently substituted for it. */
  int all_banks = __pim_lanes > 0 ? (int)__pim_lanes
                                  : cfg_num_pch * cfg_num_bg * cfg_num_banks;
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

/* The map is the ACCUMULATOR's alone: it maps a cell to its slab slot and carries the
 * read-out bit. STREAMED counts consumptions and never looks an element up. */
static void simdram_ensure_lane_slot(int tensor_id) {
  tensor_info_t *t = &tensors[tensor_id];
  if (t->role != SIMDRAM_ROLE_ACCUMULATOR || t->lane_slot)
    return;
  int all_banks = cfg_num_pch * cfg_num_bg * cfg_num_banks;
  t->lane_slot = slot_map_create((size_t)(t->num_elements / all_banks) + 1);
  if (!t->lane_slot)
    fprintf(stderr, "[simdram] WARN tensor %d: slot map allocation failed; nothing will "
                    "be placed for it.\n", tensor_id);
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
  if (n <= 0) {
    /* No descriptor at all (the residency pass was skipped). The host role is all we
     * have, so placement follows it, and the WARN turns the run into an invariant-6
     * failure. Returning silently here charged ZERO delivery for STREAMED tensors,
     * a 64x under-charge that the review of 2026-09-12 measured. */
    if (host_role != SIMDRAM_ROLE_OPERAND)
      fprintf(stderr, "[simdram] WARN tensor %d (%s): kernel carries no layout table; "
                      "placement follows the host role.\n", tensor_id,
              host_role == SIMDRAM_ROLE_ACCUMULATOR ? "ACCUMULATOR" : "STREAMED");
    simdram_ensure_lane_slot(tensor_id);
    return;
  }
  for (int i = 0; i < n; i++) {
    const int32_t *rec = __pim_layout_table + (size_t)i * PIM_LAYOUT_REC_WORDS;
    if (rec[PIM_LW_OPERAND_ARG] != tensor_id)
      continue;
    int32_t br = rec[PIM_LW_BANK_REPLICATED];
    tensors[tensor_id].bank_replicated = (int)br;
    int32_t cf = rec[PIM_LW_CELL_FANOUT];
    tensors[tensor_id].cell_fanout = cf > 0 ? (int)cf : 1;
    /* THE ROLE IS THE COMPILER'S, same rule as pim_runtime.c: is_store -> ACCUMULATOR,
     * replicated -> OPERAND (pre-stored, free, symmetric with simdram.cpp's weight),
     * partitioned -> STREAMED. The host string is checked, not obeyed. This used to
     * only report, from when the derivation missed six of nine fixture tensors; it is
     * complete now (invariant 8) and the compiler reads the layout it produced. */
    simdram_role_t role = host_role;
    if (rec[PIM_LW_IS_STORE] == 1)
      role = SIMDRAM_ROLE_ACCUMULATOR;
    else if (br >= 0)
      role = br ? SIMDRAM_ROLE_OPERAND : SIMDRAM_ROLE_STREAMED;
    if (role != host_role) {
      static const char *names[] = {"STREAMED", "OPERAND", "ACCUMULATOR"};
      fprintf(stderr, "[simdram] WARN tensor %d: host role %s, compiler layout says %s; "
                      "taking the compiler's.\n", tensor_id,
              (host_role >= 0 && host_role < 3) ? names[host_role] : "?", names[role]);
      tensors[tensor_id].role = role;
    }
    if (br < 0 && role != SIMDRAM_ROLE_OPERAND)
      fprintf(stderr, "[simdram] WARN tensor %d (%s): compiler stated no partition bit; "
                      "placement stays on the interleaved map.\n", tensor_id,
              role == SIMDRAM_ROLE_ACCUMULATOR ? "ACCUMULATOR" : "STREAMED");
    simdram_ensure_lane_slot(tensor_id);
    return;
  }
  if (host_role != SIMDRAM_ROLE_OPERAND)
    fprintf(stderr, "[simdram] WARN tensor %d: no layout record; placement stays on the "
                    "interleaved map.\n", tensor_id);
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
  t->cell_fanout = 1;
  t->emitted_R = t->emitted_W = 0;
  t->emitted_BR = 0;
  t->dedup_skips = 0;

  int total = 1;
  for (int i = 0; i < ndims && i < 4; i++) {
    t->dims[i] = dims[i];
    total *= dims[i];
  }
  t->num_elements = total;
  t->total_bytes = (size_t)total * elem_size;



  t->bank_replicated = -1;
  if (t->lane_rg) { free(t->lane_rg); t->lane_rg = NULL; }
  t->lane_rg_n = 0;
  t->lane_rg_cap = 0;
  memset(t->lane_ord, 0, sizeof(t->lane_ord));
  memset(t->lane_flushed, 0, sizeof(t->lane_flushed));
  if (t->lane_slot) { slot_map_destroy(t->lane_slot); t->lane_slot = NULL; }
  /* After num_elements and the fields above: the check stores the compiler's
   * partition bit and sizes the per-lane slab from the element count. */
  simdram_check_compiler_role(num_tensors, role);

  /* The tracker names the accumulator's most recent cell for compute traces the pass
   * could not resolve; placement fills it on first touch. */
  t->acc_tracker_idx = -1;
  if (t->role == SIMDRAM_ROLE_ACCUMULATOR && num_acc_trackers < MAX_ACC_TRACKERS) {
    int idx = num_acc_trackers++;
    t->acc_tracker_idx = idx;
    acc_trackers[idx].tensor_idx = num_tensors;
    acc_trackers[idx].ch = 0; acc_trackers[idx].pch = 0;
    acc_trackers[idx].bg = 0; acc_trackers[idx].bank = 0;
    acc_trackers[idx].sa = 0; acc_trackers[idx].row = 0;
    if (current_acc < 0)
      current_acc = idx;
  }
  fprintf(stderr, "[simdram] Tensor %d: %s, %d elems (%zu bytes), lane-placed\n",
          num_tensors,
          t->role == SIMDRAM_ROLE_OPERAND ? "OPERAND"
          : t->role == SIMDRAM_ROLE_STREAMED ? "STREAMED" : "ACCUMULATOR",
          total, t->total_bytes);
  return num_tensors++;
}

void simdram_set_phase(simdram_phase_t phase) {
  const char *names[] = {"IDLE", "COMPUTE", "HOST"};
  if (phase <= SIMDRAM_PHASE_HOST)
    fprintf(stderr, "[simdram] Phase -> %s\n", names[phase]);
  /* Phase change: flush pending slabs, reset the k-amort counters. */
  if (cur_phase != phase) {
    if (cur_phase == SIMDRAM_PHASE_COMPUTE)
      flush_all_lanes(); /* the last lane has no successor to flush it */
    memset(g_compute_trace_count_by_op, 0, sizeof(g_compute_trace_count_by_op));
  }
  cur_phase = phase;
}

void simdram_finalize(void) {
  if (finalized)
    return;
  finalized = 1;

  flush_all_lanes(); /* a harness that never leaves COMPUTE still gets its slabs */
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
  fprintf(stderr, "[simdram]   Lane row groups claimed         : %" PRIu64 "\n",
          stat_lane_rg_claimed);
  fprintf(stderr, "[simdram]   Lane-placed input writes        : %" PRIu64 "\n",
          stat_lane_placed_writes);
  fprintf(stderr, "[simdram]   Leader-bank skips               : %" PRIu64 " (enabled=%d)\n",
          stat_leader_bank_skips, g_leader_bank_only_enabled);
  fprintf(stderr, "[simdram]   K-amort skips                   : %" PRIu64 " (enabled=%d)\n",
          stat_k_amort_skips, g_k_amort_enabled);
  fprintf(stderr, "[simdram]   K-amort emissions               : %" PRIu64 " (batch=%d)\n",
          stat_k_amort_emissions, g_k_amort_batch);

  fprintf(stderr,
          "\n[simdram] === Per-tensor breakdown ===\n"
          "[simdram]   tid  role             R          W         BR  "
          "res_skips\n");
  for (int i = 0; i < num_tensors; i++) {
    tensor_info_t *t = &tensors[i];
    const char *role_str = (t->role == SIMDRAM_ROLE_STREAMED)  ? "STREAMED"
                           : (t->role == SIMDRAM_ROLE_OPERAND) ? "OPERAND "
                                                               : "ACCUMUL.";
    fprintf(stderr,
            "[simdram]   %3d  %-9s  %10" PRIu64 " %10" PRIu64 " %10" PRIu64
            "  %11" PRIu64 "\n",
            i, role_str,
            t->emitted_R, t->emitted_W, t->emitted_BR,
            t->dedup_skips);
  }

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
/* Delivery for a bank-PARTITIONED input. Lane b is bank b's PE (threads_per_warp ==
 * num_banks, gpu.thread_id lowers to __pim_get_bank_id), so the element it loads
 * belongs in bank b, in the next slot of that lane's slab (OptiPIM's alloc_col++ made
 * explicit). Nothing is emitted here: slabs are written out at the program-instance
 * boundary by flush_all_lanes, banks innermost. Emitting each slot's 16 bit-rows back
 * to back at access time was 16 row misses per column and +40% on matvec, an ordering
 * artifact not a cost.
 * The (lane, elem) slot map is the only dedup: a re-load finds its slot and adds
 * nothing. There is no fanout path any more; -1 WARNs at registration. */
/* One column slot per CONSUMPTION, which is what OptiPIM allocates (simdram.cpp,
 * alloc_col++ per output cell per tensor). Deduplicating by element instead forgave every
 * value a lane reads more than once for different cells; on conv that is the R-by-S halo
 * and it measured a 2.21x under-charge (review 2026-09-13). A separate load op is a
 * separate consumption too, so unrolled channel chains need no multiplier of their own. */
static void note_lane_consumption(tensor_info_t *t) {
  static int warned_lane = 0, warned_wide = 0;
  int all_banks = cfg_num_pch * cfg_num_bg * cfg_num_banks;
  int lane = __pim_get_bank_id();
  if (lane < 0 || lane >= all_banks || lane >= SIMDRAM_MAX_LANES) {
    if (!warned_lane++)
      fprintf(stderr, "[simdram] WARN replay lane %d outside the %d banks; "
                      "lane-placed delivery dropped.\n", lane, all_banks);
    return;
  }
  t->lane_ord[lane]++;
  if (t->lane_ord[lane] >= (1 << 24) && !warned_wide++)
    fprintf(stderr, "[simdram] WARN lane slab exceeds 2^24 slots.\n");
}

/* Accumulator twin of the above, but the address is needed NOW: the partial sum is
 * read and written every K step and every touch must land on the same slot. Bank is
 * the executing lane's; (sa, row, col) from the slot. Overrides the tuple in place. */
static int simdram_place_acc(tensor_info_t *t, int elem_idx, int peek, int *ch, int *pch,
                             int *bg, int *bank, int *sa, int *base_row, int *col) {
  int all_banks = cfg_num_pch * cfg_num_bg * cfg_num_banks;
  int lane = __pim_get_bank_id();
  if (!t->lane_slot || lane < 0 || lane >= all_banks || lane >= SIMDRAM_MAX_LANES)
    return 0;
  int32_t slot;
  if (peek) {
    /* A compute trace names the tensor, not necessarily a cell it has touched; an
     * absent cell is charged at the slab's first slot and no slot is created. */
    slot = slot_map_peek(t->lane_slot, lane, elem_idx);
    if (slot < 0)
      slot = 0;
  } else {
    int inserted = 0;
    slot = slot_map_get_or_put(t->lane_slot, lane, elem_idx, t->lane_ord[lane], &inserted);
    if (slot < 0)
      return 0;
    if (inserted)
      t->lane_ord[lane]++;
  }
  int col_slot = slot / cfg_dq_bits;
  int rgp = lane_rg_for(t, col_slot / cfg_num_cols);
  if (rgp < 0)
    return 0;
  int linear_row = rgp * cfg_pe_bits;
  *sa = linear_row / cfg_data_rows_per_sa;
  *base_row = linear_row % cfg_data_rows_per_sa;
  *col = col_slot % cfg_num_cols;
  decompose_global_bank(lane, ch, pch, bg, bank);
  return 1;
}

/* Write every partitioned slab pending since the last flush, banks INNERMOST: for each
 * (row group, bit-row, column) the lanes' banks are written round-robin, so the row
 * misses of different banks overlap instead of adding. That is simdram.cpp's order,
 * replica loop inside row and column. One lane's whole slab at a time was 32 serial
 * banks and +17.8% on matvec; the old fanout was bank-parallel by accident and this
 * makes it so on purpose, on the right addresses. Runs at the program-instance
 * boundary, at phase end and at finalize. */
static void flush_all_lanes(void) {
  if (!trace_fp)
    return;
  int all_banks = cfg_num_pch * cfg_num_bg * cfg_num_banks;
  int nl = all_banks < SIMDRAM_MAX_LANES ? all_banks : SIMDRAM_MAX_LANES;
  for (int i = 0; i < num_tensors; i++) {
    tensor_info_t *t = &tensors[i];
    if (t->role != SIMDRAM_ROLE_STREAMED)
      continue;
    /* lane_flushed counts VALUES, not columns. A column that was written while
     * partly full and has since gained values is written again: physically the
     * 128-bit word of every bit-row is rewritten, which is what OptiPIM's per-step
     * first_time_in_col charges too. Column-granular bookkeeping here delivered every
     * value added after its column's first flush for free (review 2026-09-12: matmul
     * 64x128x32 grid (1,2) wrote 512 where 1,024 were owed). */
    /* Each consumed value occupies cell_fanout column slots, one per output cell in
     * this lane that reads it. OptiPIM allocates the same way (simdram.cpp, alloc_col++
     * per output cell per tensor, memo keyed by cell), and charging one slot per bank
     * instead was a 64x discount on matmul. */
    long long F = t->cell_fanout > 0 ? t->cell_fanout : 1;
    long long slots[SIMDRAM_MAX_LANES], first[SIMDRAM_MAX_LANES];
    long long lo = -1, hi = -1;
    for (int l = 0; l < nl; l++) {
      long long used = (long long)t->lane_ord[l] * F;
      long long done = (long long)t->lane_flushed[l] * F;
      slots[l] = (used + cfg_dq_bits - 1) / cfg_dq_bits;
      first[l] = done / cfg_dq_bits;
      if (used > done) {
        if (lo < 0 || first[l] < lo) lo = first[l];
        if (slots[l] - 1 > hi) hi = slots[l] - 1;
      } else {
        first[l] = slots[l]; /* nothing pending: exclude this lane below */
      }
    }
    if (hi < 0)
      continue;
    for (long long rg = lo / cfg_num_cols; rg <= hi / cfg_num_cols; rg++) {
      int rgp = lane_rg_for(t, (int)rg);
      if (rgp < 0)
        break;
      int linear_row = rgp * cfg_pe_bits;
      int sa = linear_row / cfg_data_rows_per_sa;
      int base_row = linear_row % cfg_data_rows_per_sa;
      for (int bit = 0; bit < cfg_pe_bits; bit++)
        for (int col = 0; col < cfg_num_cols; col++) {
          long long slot = rg * cfg_num_cols + col;
          for (int l = 0; l < nl; l++) {
            if (slot < first[l] || slot >= slots[l])
              continue;
            int ch, pch, bg, bank;
            decompose_global_bank(l, &ch, &pch, &bg, &bank);
            emit_trace("W", ch, pch, bg, bank, sa, base_row + bit, col);
            stat_writes++;
            t->emitted_W++;
            stat_lane_placed_writes++;
          }
        }
    }
    for (int l = 0; l < nl; l++)
      t->lane_flushed[l] = t->lane_ord[l];
  }
}

/* A new program instance means every lane of the previous one has finished. */
/* A program instance owns its own output cells, so the cells of the next instance are
 * different cells and owe their own copy of whatever they read. OptiPIM has no instances
 * at all: its spatial loop covers every output cell and re-allocates a column per cell,
 * so keeping the slabs across the boundary made a re-read free that the baseline charges
 * (worth ceil(M/BLOCK_M) on matmul's B). HBM dropped the same reuse on 2026-09-10.
 * The row groups are kept: the next instance rewrites the same rows, it does not need new
 * ones. */
static void reset_streamed_slabs(void) {
  for (int i = 0; i < num_tensors; i++) {
    tensor_info_t *t = &tensors[i];
    if (t->role != SIMDRAM_ROLE_STREAMED)
      continue;
    memset(t->lane_ord, 0, sizeof(t->lane_ord));
    memset(t->lane_flushed, 0, sizeof(t->lane_flushed));
  }
}

static void note_epoch_change(void) {
  uint64_t e = __pim_program_epoch;
  if (e == g_last_epoch)
    return;
  flush_all_lanes();
  reset_streamed_slabs();
  g_last_epoch = e;
}

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

  if (cur_phase == SIMDRAM_PHASE_HOST) {
    static int warned = 0;
    if (!warned++)
      fprintf(stderr, "[simdram] WARN HOST phase has no placement model; its accesses "
                      "are ignored.\n");
    stat_ignored++;
    return;
  }
  /* Only an accumulator needs a tuple here: STREAMED is noted into its slab and written
   * at the instance boundary, OPERAND emits nothing. Placed before the gate so the
   * read-out, the tracker and the compute BR all agree on the lane's bank. */
  int ch = 0, pch = 0, bg = 0, bank = 0, sa = 0, base_row = 0, col = 0;
  if (cur_phase == SIMDRAM_PHASE_COMPUTE && t->role == SIMDRAM_ROLE_ACCUMULATOR &&
      !(t->lane_slot &&
        simdram_place_acc(t, elem_idx, 0, &ch, &pch, &bg, &bank, &sa, &base_row, &col))) {
    stat_ignored++; /* no slab: allocation failed and WARNed at registration */
    return;
  }

  if (cur_phase == SIMDRAM_PHASE_COMPUTE) {
    note_epoch_change();
    if (!is_write && t->role == SIMDRAM_ROLE_STREAMED) {
      note_lane_consumption(t); /* before the gate: every lane */
      return;
    }
    /* Leader-bank-only gate for what is left: accumulator read-out, in lockstep with
     * compute_trace. HOST phase not gated. */
    if (g_leader_bank_only_enabled && __pim_get_bank_id() != 0) {
      stat_leader_bank_skips++;
      return;
    }
    if (!is_write) {
      switch (t->role) {
      case SIMDRAM_ROLE_ACCUMULATOR: {
        /* Partial-sum read-out: pe_bits bit-rows, once per (lane, cell). The accumulator
         * is loop-carried, so after the first touch it is resident in its own bank and a
         * later load reads nothing from DRAM. The marker is the slot's flag bit, not an
         * address set: a compute trace naming this tensor may have created the slot
         * already, and that must not count as the read-out. */
        int lane = __pim_get_bank_id();
        int done = t->lane_slot ? slot_map_test_and_set(t->lane_slot, lane, elem_idx, 30) : 0;
        if (done == 1) {
          t->dedup_skips++;
          break;
        }
        for (int bit = 0; bit < cfg_pe_bits; bit++) {
          emit_trace("R", ch, pch, bg, bank, sa, base_row + bit, col);
          stat_reads++; t->emitted_R++;
        }
        if (t->acc_tracker_idx >= 0) {
          acc_tracker_t *at = &acc_trackers[t->acc_tracker_idx];
          at->ch = ch; at->pch = pch; at->bg = bg; at->bank = bank;
          at->sa = sa; at->row = base_row;
          current_acc = t->acc_tracker_idx;
        }
        break;
      }
      case SIMDRAM_ROLE_STREAMED:
        break; /* diverted above */
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
  }
}

/* SIMDRAM prices one bit-serial expansion per ELEMENT and discards both the byte
 * size and the lane count: a vector access is the same expansion repeated, and the
 * host replay hands us one element per call. Accepted so the ABI matches the pass
 * and the HBM runtime. */
/* One call is one machine instruction, which on a vector access moves `lanes` values.
 * Discarding the width priced a <8 x half> load as a single element and under-charged
 * delivery up to 8x (review 2026-09-12); the HBM twin has always expanded. The column
 * packing in flush_all_lanes is what collapses word-mates, so expanding here does not
 * double-charge a bus transaction: 128 values still cost one column of bit-row writes. */
static void simdram_trace_access_range(uint64_t base_addr, uint64_t size, int is_write,
                                       uint64_t compiler_lanes) {
  int tidx = find_tensor(base_addr);
  int elem_size = (tidx >= 0 && tensors[tidx].elem_size > 0) ? tensors[tidx].elem_size : 1;
  uint64_t n = size / (uint64_t)elem_size;
  if (n == 0)
    n = 1;
  if (compiler_lanes && compiler_lanes != n) {
    static int warned = 0;
    if (!warned++)
      fprintf(stderr, "[simdram] WARN lane-count disagreement: the compiler emitted %llu "
                      "lanes for a %llu-byte access, elem_size=%d makes that %llu. "
                      "Pricing the runtime's count.\n",
              (unsigned long long)compiler_lanes, (unsigned long long)size, elem_size,
              (unsigned long long)n);
  }
  for (uint64_t i = 0; i < n; i++)
    simdram_trace_access(base_addr + i * (uint64_t)elem_size, is_write);
}

void __mem_trace_load(void *addr, uint64_t size, uint64_t lanes) {
  if (!trace_fp)
    return;
  simdram_trace_access_range((uint64_t)addr, size, 0, lanes);
}

void __mem_trace_store(void *addr, uint64_t size, uint64_t lanes) {
  if (!trace_fp)
    return;
  simdram_trace_access_range((uint64_t)addr, size, 1, lanes);
}

/* ================================================================
 *  ComputeTracePass hook (arithmetic instrumentation)
 * ================================================================ */

/*
 * Emit MAJ-3 gate operations as bank-read (BR) trace ops, cycling through
 * pe_bits rows at the destination accumulator, matching OptiPIM's
 * SimDRAMCodeGen pattern.
 *
 * dest_addr is ComputeTracePass's forward def-use walk to the eventual store.
 * When that walk fails the pass passes NULL and we fall back to `current_acc`,
 * the tracker of the most recently touched accumulator cell. The fall-back
 * misattributes when two accumulators interleave with no load or store between
 * them, and charges BR for arithmetic that is not a SIMDRAM operation at all
 * (address math, induction variables) during COMPUTE. bit_width is part of the
 * ABI but unused: the MAJ-3 cost comes from the tensor's own pe_bits.
 */
void __compute_trace(int32_t opcode, int32_t bit_width, void *dest_addr) {
  (void)bit_width;
  if (!trace_fp || cur_phase != SIMDRAM_PHASE_COMPUTE)
    return;
  note_epoch_change();

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
      if (elem_idx >= 0 && elem_idx < t->num_elements &&
          t->role == SIMDRAM_ROLE_ACCUMULATOR && t->lane_slot &&
          simdram_place_acc(t, elem_idx, 1, &ch, &pch, &bg, &bank, &sa, &base_row, &col)) {
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
    /* Last resort. The pass names the destination tensor even when it cannot name the
     * element, so reaching here means arithmetic whose result never reaches any
     * argument. The bank is then a guess from trace order; say so once. */
    static int warned_guess = 0;
    if (!warned_guess++)
      fprintf(stderr, "[simdram] WARN compute trace with no destination; charged at the "
                      "most recently touched accumulator's bank.\n");
    const acc_tracker_t *at = &acc_trackers[current_acc];
    ch = at->ch; pch = at->pch; bg = at->bg; bank = at->bank;
    sa = at->sa; base_row = at->row; col = 0;
    tensor_id = at->tensor_idx;
  }


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
    stat_k_amort_emissions++;
    cost = maj_cost(opcode);
  } else {
    /* Either BATCH=1 or PIM_K_AMORT=0. Both mean the same thing, every op pays its
     * own MAJ-3, which is the honest per-arithmetic-op baseline. They were two
     * branches with identical bodies until 2026-09-22. */
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
