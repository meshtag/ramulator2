#include "simdram_runtime.h"
#include "im_addr_dedup.h"
#include "im_runtime.h"
/* Shared with pim_runtime.c so the two runtimes cannot disagree about the record
 * width. The two are never linked together (pipeline.py picks one by target), so
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
 * polynomials in pe_bits). Arithmetic on cells that share a row is
 * batched per opcode, one MAJ block per row pass (g_k_amort_batch).
 *
 * Accumulator read-out is emitted once per (lane, cell): a loop-carried value is
 * resident in its own bank after the first touch, and the slot's marker bit says so.
 * That is residency the compiler states, not an address-collapse heuristic. A cell
 * never loaded is read out at its store, with every partial sum a split-K reduce
 * combined, since the host does that sum (simdram_read_out_cell).
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
 *  HBM3_PIM organization constants. The defaults match OptiPIM's
 *  simdram_config.yaml exactly.
 * ================================================================ */
static const int cfg_num_pch = 2; /* rank: 2 */
static const int cfg_num_bg = 4;
static const int cfg_num_banks = 4; /* per bank group */
/* Copy of the simulator's organization, as in pim_runtime.c, and nothing checks that the
 * two agree. The shared SIMDRAM spec names HBM3_8Gb, whose preset has 64 subarrays. Used
 * only for the capacity ceiling and the banner, never in the address slicing. */
static const int cfg_num_sa = 64;
static const int cfg_num_rows = 512; /* total physical rows per subarray  */
static const int cfg_num_cols = 64;
static const int cfg_dq_bits = 128;
static int64_t simdram_derived_k_amort_batch(void);
static void flush_all_lanes(void);

static const int cfg_pe_bits = 16;

/* Placement is lane placement. A streamed input or an accumulator lives in the executing
 * lane's own bank, in row groups claimed as its slab grows. An operand is pre-stored and
 * emits nothing. */

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

/* Usable data rows per subarray. */
static const int cfg_data_rows_per_sa = cfg_num_rows - SIMDRAM_RESERVED_ROWS;

/* ================================================================
 *  Tensor registry
 * ================================================================ */
#define MAX_TENSORS 16
#define SIMDRAM_MAX_LANES 64

typedef struct {
  void *base_addr;
  size_t total_bytes;
  int elem_size;
  int num_elements;
  simdram_role_t role;

  /* Row groups of this tensor's per-lane slab, the same in every bank, claimed as the slab
   * grows from a per-bank pool at the top of the row space. Nothing is reserved up front,
   * since how much a lane touches depends on the launch grid. */
  int *lane_rg;
  int lane_rg_n, lane_rg_cap;
  int lane_ord[SIMDRAM_MAX_LANES]; /* next free value slot per lane */
  int lane_flushed[SIMDRAM_MAX_LANES]; /* VALUES already written out, per lane */
  slot_map_t *lane_slot;   /* (lane, elem) -> slot, for an accumulator only */
  /* Cells in this lane that one CONSUMED value feeds through a tile broadcast, from the
   * compiler. Sharing that shows up as repeated loads is counted by the loads themselves;
   * this covers only the sharing that never reaches memory, such as matmul's b[k,n] being
   * multiplied into every row of the tile. 1 until the artifact says otherwise. */
  int cell_fanout;
  /* Partial sums a store's cell combined, from the compiler's LIVE_SPLIT word: SPLIT_K on
   * a split-K store, 1 on every other. The host reads all of them out. */
  int live_split;
  slot_map_t *readout_cols; /* (lane, partial column) already read out */
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

/* Where an accumulator cell lives: the executing lane's bank, the cell's row group and its
 * column slot. */
typedef struct {
  int ch, pch, bg, bank, sa, base_row, col;
} acc_cell_t;

/* The accumulator cell touched last, where __compute_trace charges arithmetic whose
 * destination it cannot resolve. Valid once an accumulator has registered. */
static struct {
  int valid;
  acc_cell_t cell;
} last_acc;

/* The program epoch of the last access. A new one means every lane of the previous
 * instance has finished. */
static uint64_t g_last_epoch = 0;

/* Per-run counts and warn-once flags. simdram_init clears both, because the harness
 * inits one library more than once per process (the ladder verifies, then traces), and
 * each run must report its own counts and WARNs. */
static struct {
  uint64_t bank_reads, reads, writes, compute_ops, ignored;
  uint64_t lane_placed_writes, lane_rg_claimed;
} stats;
/* One flag per WARN condition, so one condition cannot hide another. */
static struct {
  int row_space, unpriced_op, lane_outside, lane_wide, lane_count, no_destination,
      vector_op;
} warned;

/* K-amortization batching: one MAJ block per opcode every g_k_amort_batch compute
 * traces. BATCH is the per-pass lane count, so every batch boundary lands on a pid
 * boundary and the counter's persistence across pids is worth nothing. The leader-bank
 * gate fires before the counter increments, so the counter sees one bank and BATCH is
 * denominated in cells per bank. Numerator and denominator each divide by 32 once.
 * simdram_init sets BATCH from the compiler's output footprint. */
static int g_k_amort_batch = 0;
/* Per-opcode batch counters: a MUL and an ADD amortize independently because
 * their MAJ-3 costs differ by ~14x. */
static uint64_t g_compute_trace_count_by_op[SIMDRAM_OP_COUNT] = {0};

/* The leader-bank gate, OptiPIM's single_bank_opt (simdram.cpp:136-142). SIMDRAM computes
 * on every bank of a channel in lockstep, each bank running the same loop body on its own
 * cells, so one bank's stream stands for the channel. With the gate on, only lane 0's
 * replay emits compute and the accumulator's read-out. Streamed inputs bypass it, since
 * every lane writes its own values into its own bank.
 *
 * PIM_LANE_MODEL moves the gate and the occupancy divisor together, as either alone shifts
 * the charge by the lane count. leader, the default, gates and divides the output tile by
 * the lanes. paired does neither, and it is the diagnostic: if the gate drops only replicas
 * of lane 0's work, both models emit the same bank reads. */
typedef enum { SIMDRAM_LANE_LEADER = 0, SIMDRAM_LANE_PAIRED = 1 } simdram_lane_model_t;
static simdram_lane_model_t g_lane_model = SIMDRAM_LANE_LEADER;

/* Record decoding, the drift checks, tensor lookup, trace output and the MemTracePass
 * hooks, shared with pim_runtime.c. */
#include "im_trace_common.h"

/* ================================================================
 *  Helpers
 * ================================================================ */

/* Per-bank pool of row groups for lane slabs, claimed downward from the top of the row
 * space. Lane slabs are the only rows SIMDRAM places, so nothing grows from below. */
static int g_lane_rg_top = 0;

/* Physical row group for slab row group `idx` of tensor t, claiming from the top of the
 * row space on first use. An exhausted pool aliases row group 0 and WARNs once. Returns
 * -1 only when the table of row groups cannot grow. */
static int lane_rg_for(tensor_info_t *t, int idx) {
  while (idx >= t->lane_rg_n) {
    if (t->lane_rg_n == t->lane_rg_cap) {
      int ncap = t->lane_rg_cap ? t->lane_rg_cap * 2 : 8;
      int *n = (int *)realloc(t->lane_rg, (size_t)ncap * sizeof(int));
      if (!n) return -1;
      t->lane_rg = n; t->lane_rg_cap = ncap;
    }
    if (g_lane_rg_top <= 0) {
      if (!warned.row_space++)
        fprintf(stderr, "[simdram] WARN row space exhausted while growing a lane slab; "
                        "placement beyond this point aliases row group 0.\n");
      t->lane_rg[t->lane_rg_n++] = 0;
      continue;
    }
    t->lane_rg[t->lane_rg_n++] = --g_lane_rg_top;
    stats.lane_rg_claimed++;
  }
  return t->lane_rg[idx];
}

/*
 * Compute SIMDRAM AAP/AP cost for a given integer operation.
 * Costs from the SIMDRAM paper (Table 1), in units of AAPs/APs,
 * where n = pe_bits (number of bits per operand, stored bit-serially).
 *
 *   Operation       Cost (AAPs/APs)     Class
 *   ─────────────   ─────────────────   ─────────
 *   addition        8n + 1              Linear
 *   subtraction     8n + 1              Linear
 *   multiplication  7n² + 1             Quadratic  (OptiPIM's emulated
 *                                                   formula, simdram.cpp:58,
 *                                                   not the paper's 11n²−5n−1)
 *   division        8n² + 12n           Quadratic
 *   bitwise AND     n                   Linear   (1 MAJ per bit)
 *   bitwise OR      n                   Linear   (1 MAJ per bit)
 *   bitwise XOR     5n                  Linear   (~5 MAJ per bit)
 *   shift           n                   Linear
 *
 * Float opcodes take the fixed-point polynomial of their integer twin, see below.
 */

static int maj_cost(int opcode) {
  int n = cfg_pe_bits;
  switch (opcode) {
  /* Float opcodes take the fixed-point polynomials. SIMDRAM has no mantissa or exponent
   * hardware, only MAJ-3 over n bit-planes, so fp16 is priced as 16-bit fixed point, as
   * OptiPIM prices it with one dataWidth for every tensor. */
  case SIMDRAM_OP_MUL:
  case SIMDRAM_OP_FMUL:
    return 7 * n * n + 1;
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
    /* An opcode with no polynomial is charged nothing, so that is said once. */
    if (!warned.unpriced_op) {
      warned.unpriced_op = 1;
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

/* PIM_LANE_MODEL, leader when unset. */
static simdram_lane_model_t parse_lane_model(void) {
  const char *env = getenv("PIM_LANE_MODEL");
  if (!env || strcmp(env, "leader") == 0)
    return SIMDRAM_LANE_LEADER;
  if (strcmp(env, "paired") == 0)
    return SIMDRAM_LANE_PAIRED;
  fprintf(stderr, "[simdram] ERROR: PIM_LANE_MODEL=%s; expected 'leader' or 'paired'.\n",
          env);
  exit(1);
}

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

  check_compiler_dq_bits("simdram", cfg_dq_bits);
  check_layout_rec_words("simdram");

  num_tensors = 0;
  cur_phase = SIMDRAM_PHASE_IDLE;
  memset(&stats, 0, sizeof stats);
  memset(&warned, 0, sizeof warned);
  memset(&last_acc, 0, sizeof last_acc);
  initialized = 1;
  finalized = 0;

  g_lane_model = parse_lane_model();
  if (__pim_lanes > 0 && (int)__pim_lanes != cfg_num_pch * cfg_num_bg * cfg_num_banks)
    fprintf(stderr, "[simdram] WARN kernel compiled for %d lanes, machine has %d banks; "
                    "the occupancy divisor follows the kernel.\n",
            (int)__pim_lanes, cfg_num_pch * cfg_num_bg * cfg_num_banks);
  g_lane_rg_top = (cfg_num_sa * cfg_data_rows_per_sa) / cfg_pe_bits;
  g_last_epoch = __pim_program_epoch;
  fprintf(stderr,
          "[simdram] lane_model=%s (leader_bank_only=%d; set PIM_LANE_MODEL=paired to "
          "drop the gate AND the matching /banks together)\n",
          g_lane_model == SIMDRAM_LANE_PAIRED ? "paired" : "leader",
          g_lane_model == SIMDRAM_LANE_LEADER);

  /* The batch is the compiler's occupancy. The fallback is the values one open row covers,
   * cfg_num_cols * cfg_dq_bits bitlines, OptiPIM's n_values_per_row (simdram.cpp:88-89),
   * and it is not divided by pe_bits since maj_cost already sweeps every bit-plane. */
  memset(g_compute_trace_count_by_op, 0, sizeof(g_compute_trace_count_by_op));
  int64_t derived = simdram_derived_k_amort_batch();
  g_k_amort_batch = derived > 0 ? (int)derived : cfg_num_cols * cfg_dq_bits;
  if (derived > 0)
    fprintf(stderr, "[simdram] k_amort_batch=%d from the compiler's output footprint\n",
            g_k_amort_batch);
  else
    fprintf(stderr,
            "[simdram] WARN no store footprint in the artifact%s, so k_amort_batch=%d "
            "is the fallback and the occupancy charge is unverified.\n",
            (int)__pim_layout_count > 0 ? "" : " (no layout table at all)",
            g_k_amort_batch);

  fprintf(stderr, "[simdram] Initialized. Trace: %s\n", trace_file);
  fprintf(stderr,
          "[simdram] HBM config: %d pch, %d bg, %d banks/bg, "
          "%d sa, %d rows/sa (%d data + %d reserved), %d cols, "
          "%d-bit DQ, pe_bits=%d\n",
          cfg_num_pch, cfg_num_bg, cfg_num_banks, cfg_num_sa,
          cfg_num_rows, cfg_data_rows_per_sa, SIMDRAM_RESERVED_ROWS,
          cfg_num_cols, cfg_dq_bits, cfg_pe_bits);
}

/* The occupancy the compiler planned for, read out of the artifact. The charge divides by
 * the live output tile per bank, and a store's footprint is one instance's output, which
 * excludes the reduction axis by construction, as a load's would not. Returns 0 when the
 * kernel carries no store record with a footprint. */
static int64_t simdram_derived_k_amort_batch(void) {
  /* The gate keeps one lane of the compiler's lane count, so the same count divides the
   * tile, and the configured bank count stands in only when the kernel did not say. */
  int all_banks = __pim_lanes > 0 ? (int)__pim_lanes
                                  : cfg_num_pch * cfg_num_bg * cfg_num_banks;
  int banks = (g_lane_model == SIMDRAM_LANE_PAIRED) ? 1 : all_banks;
  int64_t row_values = (int64_t)cfg_num_cols * cfg_dq_bits;
  /* The row is a per-bank ceiling, so dropping the division raises it by the same factor,
   * or the cap would bind in one lane model and not the other. */
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
     * end, so the store sees BLOCK while the row holds BLOCK*SPLIT_K. */
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

/* Takes the compiler's record for one tensor, its role, cell fanout and live split. A
 * tensor without a record keeps the host's role, and a WARN says so. Every path gives an
 * accumulator its slot map. */
static void apply_compiler_layout(int tensor_id, simdram_role_t host_role) {
  tensor_info_t *t = &tensors[tensor_id];
  const int32_t *rec = layout_record(tensor_id);
  if (!rec) {
    if (host_role != SIMDRAM_ROLE_OPERAND) {
      if ((int)__pim_layout_count <= 0)
        fprintf(stderr, "[simdram] WARN tensor %d (%s): kernel carries no layout table; "
                        "placement follows the host role.\n", tensor_id,
                host_role == SIMDRAM_ROLE_ACCUMULATOR ? "ACCUMULATOR" : "STREAMED");
      else
        fprintf(stderr, "[simdram] WARN tensor %d: no layout record, so the compiler "
                        "stated no placement for it.\n", tensor_id);
    }
    simdram_ensure_lane_slot(tensor_id);
    return;
  }
  int32_t cf = rec[PIM_LW_CELL_FANOUT];
  t->cell_fanout = cf > 0 ? (int)cf : 1;
  int32_t ls = rec[PIM_LW_LIVE_SPLIT];
  t->live_split = ls > 0 ? (int)ls : 1;
  /* A replicated input is an operand, pre-stored and free as simdram.cpp's weight is. */
  t->role = (simdram_role_t)role_from_record("simdram", tensor_id, rec, host_role);
  if (rec[PIM_LW_BANK_REPLICATED] < 0 && t->role != SIMDRAM_ROLE_OPERAND)
    fprintf(stderr, "[simdram] WARN tensor %d (%s): compiler stated no partition bit; "
                    "placement follows the role.\n", tensor_id,
            t->role == SIMDRAM_ROLE_ACCUMULATOR ? "ACCUMULATOR" : "STREAMED");
  simdram_ensure_lane_slot(tensor_id);
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
  t->live_split = 1;

  int total = 1;
  for (int i = 0; i < ndims && i < 4; i++)
    total *= dims[i];
  t->num_elements = total;
  t->total_bytes = (size_t)total * elem_size;

  if (t->lane_rg) { free(t->lane_rg); t->lane_rg = NULL; }
  t->lane_rg_n = 0;
  t->lane_rg_cap = 0;
  memset(t->lane_ord, 0, sizeof(t->lane_ord));
  memset(t->lane_flushed, 0, sizeof(t->lane_flushed));
  if (t->lane_slot) { slot_map_destroy(t->lane_slot); t->lane_slot = NULL; }
  if (t->readout_cols) { slot_map_destroy(t->readout_cols); t->readout_cols = NULL; }
  /* After num_elements and the defaults above, which the record overwrites and the slot
   * map is sized from. */
  apply_compiler_layout(num_tensors, role);
  if (t->role == SIMDRAM_ROLE_ACCUMULATOR)
    last_acc.valid = 1;
  fprintf(stderr, "[simdram] Tensor %d: %s, %d elems (%zu bytes), lane-placed\n",
          num_tensors, role_name(t->role), total, t->total_bytes);
  return num_tensors++;
}

void simdram_set_phase(simdram_phase_t phase) {
  const char *names[] = {"IDLE", "COMPUTE"};
  if (phase <= SIMDRAM_PHASE_COMPUTE)
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
          stats.bank_reads);
  fprintf(stderr, "[simdram]   Reads       (R) : %" PRIu64 "\n", stats.reads);
  fprintf(stderr, "[simdram]   Writes      (W) : %" PRIu64 "\n", stats.writes);
  fprintf(stderr, "[simdram]   Compute ops     : %" PRIu64 "\n",
          stats.compute_ops);
  fprintf(stderr, "[simdram]   Ignored         : %" PRIu64 "\n", stats.ignored);
  fprintf(stderr, "[simdram]   Total SIMDRAM ops: %" PRIu64 "\n",
          stats.bank_reads + stats.reads + stats.writes);
  fprintf(stderr, "[simdram]   Lane row groups claimed         : %" PRIu64 "\n",
          stats.lane_rg_claimed);
  fprintf(stderr, "[simdram]   Lane-placed input writes        : %" PRIu64 "\n",
          stats.lane_placed_writes);
  initialized = 0;
}

/* ================================================================
 *  MemTracePass hooks (memory access instrumentation)
 * ================================================================ */

/*
 * SIMDRAM memory access trace mapping (matches OptiPIM SimDRAMCodeGen):
 *
 * COMPUTE phase:
 *   STREAMED (input) load      → noted in the lane's slab, pe_bits W per column at
 *                                the instance boundary (flush_all_lanes)
 *   OPERAND (weight) load      → nothing (pre-stored in DRAM rows)
 *   ACCUMULATOR (output) load  → pe_bits R ops, once per (lane, cell)
 *   ACCUMULATOR (output) store → read-out of the cell's partials
 */
/* Lane b is bank b's PE, so a value it loads takes the next column slot of its own slab,
 * one slot per consumption, as OptiPIM allocates a column per output cell (alloc_col++).
 * Nothing is emitted here. flush_all_lanes writes the slabs at the instance boundary. */
static void note_lane_consumption(tensor_info_t *t) {
  int all_banks = cfg_num_pch * cfg_num_bg * cfg_num_banks;
  int lane = __pim_get_bank_id();
  if (lane < 0 || lane >= all_banks || lane >= SIMDRAM_MAX_LANES) {
    if (!warned.lane_outside++)
      fprintf(stderr, "[simdram] WARN replay lane %d outside the %d banks; "
                      "lane-placed delivery dropped.\n", lane, all_banks);
    return;
  }
  t->lane_ord[lane]++;
  if (t->lane_ord[lane] >= (1 << 24) && !warned.lane_wide++)
    fprintf(stderr, "[simdram] WARN lane slab exceeds 2^24 slots.\n");
}

/* Accumulator twin of the above, but the address is needed now, since the partial sum is
 * read and written every K step and every touch must land on the same slot. The bank is
 * the executing lane's and (sa, row, col) come from the slot. */
static int simdram_place_acc(tensor_info_t *t, int elem_idx, int peek, acc_cell_t *c) {
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
  c->sa = linear_row / cfg_data_rows_per_sa;
  c->base_row = linear_row % cfg_data_rows_per_sa;
  c->col = col_slot % cfg_num_cols;
  decompose_global_bank(lane, &c->ch, &c->pch, &c->bg, &c->bank);
  return 1;
}

static void note_acc_cell(const acc_cell_t *c) { last_acc.cell = *c; }

/* Writes every streamed slab pending since the last flush, banks innermost. For each (row
 * group, bit-row, column) the lanes' banks take turns, so the row misses of different
 * banks overlap, as simdram.cpp orders its replica loop inside row and column. Runs at the
 * program-instance boundary, at phase end and at finalize. */
static void flush_all_lanes(void) {
  if (!trace_fp)
    return;
  int all_banks = cfg_num_pch * cfg_num_bg * cfg_num_banks;
  int nl = all_banks < SIMDRAM_MAX_LANES ? all_banks : SIMDRAM_MAX_LANES;
  for (int i = 0; i < num_tensors; i++) {
    tensor_info_t *t = &tensors[i];
    if (t->role != SIMDRAM_ROLE_STREAMED)
      continue;
    /* lane_flushed counts values, so a column written while partly full is written again
     * once it gains values, as OptiPIM's per-step first_time_in_col charges. Each consumed
     * value takes cell_fanout column slots, one per output cell in this lane that reads it. */
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
            stats.writes++;
            stats.lane_placed_writes++;
          }
        }
    }
    for (int l = 0; l < nl; l++)
      t->lane_flushed[l] = t->lane_ord[l];
  }
}

/* A program instance owns its output cells, so the next instance's cells owe their own
 * copy of whatever they read, as OptiPIM re-allocates a column per cell. The row groups
 * are kept, since the next instance rewrites the same rows. */
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

/* The two read-outs of an accumulator cell, which the slot's bit 30 keeps to one per
 * (lane, cell). They differ because they serve different readers. A loaded cell is
 * loop-carried, so its first load reads the cell's own pe_bits bit-rows, and the value
 * then stays resident in the lane's bank. Returns whether it read the cell out. */
static int read_out_on_first_load(tensor_info_t *t, int elem_idx, const acc_cell_t *c) {
  if (slot_map_test_and_set(t->lane_slot, __pim_get_bank_id(), elem_idx, 30) == 1)
    return 0;
  for (int bit = 0; bit < cfg_pe_bits; bit++) {
    emit_trace("R", c->ch, c->pch, c->bg, c->bank, c->sa, c->base_row + bit, c->col);
    stats.reads++;
  }
  return 1;
}

/* A cell first touched by its store goes to the host, which reads out every partial sum
 * it combines, as OptiPIM's codegen reads the output's partials (simdram.cpp:39-43). That
 * is pe_bits bit-rows per DRAM column, so cells sharing a column share its reads, and a
 * split-K cell sends all live_split partials, packed after its slot, because SIMDRAM
 * cannot add across bitlines (SIMDRAM, Sec. 5.6). */
static void simdram_read_out_cell(tensor_info_t *t, int elem_idx) {
  int lane = __pim_get_bank_id();
  if (!t->lane_slot || slot_map_test_and_set(t->lane_slot, lane, elem_idx, 30) != 0)
    return;
  int32_t slot = slot_map_peek(t->lane_slot, lane, elem_idx);
  if (slot < 0)
    return;
  if (!t->readout_cols)
    t->readout_cols = slot_map_create(64);
  if (!t->readout_cols)
    return;
  int64_t s = t->live_split > 0 ? t->live_split : 1;
  int64_t first = (int64_t)slot * s / cfg_dq_bits;
  int64_t last = ((int64_t)slot * s + s - 1) / cfg_dq_bits;
  int ch, pch, bg, bank;
  decompose_global_bank(lane, &ch, &pch, &bg, &bank);
  for (int64_t c = first; c <= last; c++) {
    int inserted = 0;
    if (slot_map_get_or_put(t->readout_cols, lane, (int)c, 0, &inserted) < 0 || !inserted)
      continue;
    int rgp = lane_rg_for(t, (int)(c / cfg_num_cols));
    if (rgp < 0)
      return;
    int linear_row = rgp * cfg_pe_bits;
    int sa = linear_row / cfg_data_rows_per_sa;
    int base_row = linear_row % cfg_data_rows_per_sa;
    for (int bit = 0; bit < cfg_pe_bits; bit++) {
      emit_trace("R", ch, pch, bg, bank, sa, base_row + bit, (int)(c % cfg_num_cols));
      stats.reads++;
    }
  }
}

/* One element of an access. An accumulator is placed first and the epoch checked after,
 * since both claim row groups and the order of the claims sets the addresses. */
static void trace_element(uint64_t addr, int is_write) {
  if (cur_phase != SIMDRAM_PHASE_COMPUTE) {
    stats.ignored++;
    return;
  }

  int tidx = find_tensor(addr);
  if (tidx < 0) {
    stats.ignored++;
    return;
  }

  tensor_info_t *t = &tensors[tidx];
  int elem_idx = (int)((addr - (uint64_t)t->base_addr) / t->elem_size);
  if (elem_idx < 0 || elem_idx >= t->num_elements) {
    stats.ignored++;
    return;
  }

  acc_cell_t cell;
  if (t->role == SIMDRAM_ROLE_ACCUMULATOR && !simdram_place_acc(t, elem_idx, 0, &cell)) {
    stats.ignored++; /* no slab: allocation failed and WARNed at registration */
    return;
  }
  note_epoch_change();

  switch (t->role) {
  case SIMDRAM_ROLE_OPERAND:
    return;
  case SIMDRAM_ROLE_STREAMED:
    if (!is_write)
      note_lane_consumption(t); /* every lane, before the gate */
    return;
  case SIMDRAM_ROLE_ACCUMULATOR:
    if (g_lane_model == SIMDRAM_LANE_LEADER && __pim_get_bank_id() != 0)
      return;
    if (is_write) {
      simdram_read_out_cell(t, elem_idx);
      note_acc_cell(&cell);
    } else if (read_out_on_first_load(t, elem_idx, &cell)) {
      note_acc_cell(&cell);
    }
    return;
  }
}

/* One call is one machine instruction, which on a vector access moves `lanes` values.
 * Each value is traced, and the column packing in flush_all_lanes collapses word-mates,
 * so 128 values still cost one column of bit-row writes. */
static void trace_access_range(uint64_t base_addr, uint64_t size, int is_write,
                               uint64_t compiler_lanes) {
  int tidx = find_tensor(base_addr);
  int elem_size = (tidx >= 0 && tensors[tidx].elem_size > 0) ? tensors[tidx].elem_size : 1;
  uint64_t n = size / (uint64_t)elem_size;
  if (n == 0)
    n = 1;
  check_access_lanes("simdram", &warned.lane_count, compiler_lanes, size, elem_size, n);
  for (uint64_t i = 0; i < n; i++)
    trace_element(base_addr + i * (uint64_t)elem_size, is_write);
}

/* ================================================================
 *  ComputeTracePass hook (arithmetic instrumentation)
 * ================================================================ */

/* The accumulator cell dest_addr names, in the executing lane's bank and without creating
 * a slot. Returns 0 when it names no lane-placed accumulator. */
static int place_destination(void *dest_addr, acc_cell_t *c) {
  int tidx = dest_addr ? find_tensor((uint64_t)(uintptr_t)dest_addr) : -1;
  if (tidx < 0)
    return 0;
  tensor_info_t *t = &tensors[tidx];
  int elem_idx = (int)(((uint64_t)(uintptr_t)dest_addr - (uint64_t)(uintptr_t)t->base_addr) /
                       (uint64_t)t->elem_size);
  return elem_idx >= 0 && elem_idx < t->num_elements &&
         t->role == SIMDRAM_ROLE_ACCUMULATOR && simdram_place_acc(t, elem_idx, 1, c);
}

/*
 * Emits MAJ-3 gate operations as bank-read (BR) ops on the B-group rows of the
 * destination's subarray, as OptiPIM's SimDRAMCodeGen does.
 *
 * dest_addr is normally the accumulator tensor's argument, since the accumulator is stored
 * after the loop, and the element pointer only when it dominates the arithmetic. NULL comes
 * only when no tensor is named, and such arithmetic is charged at the accumulator cell
 * touched last. lanes is the operation's vector width, and the charge is one operation
 * whatever the width.
 */
void __compute_trace(int32_t opcode, int32_t lanes, void *dest_addr) {
  if (!trace_fp || cur_phase != SIMDRAM_PHASE_COMPUTE)
    return;
  note_epoch_change();
  if (lanes > 1 && !warned.vector_op++)
    fprintf(stderr, "[simdram] WARN a %d-lane vector operation is charged as one operation, "
                    "so the arithmetic of the other lanes goes unpriced.\n", (int)lanes);

  /* Leader-bank-only gate: SIMDRAM compute fires in channel-wide
   * lockstep; emit one bank's stream as the wallclock proxy. */
  if (g_lane_model == SIMDRAM_LANE_LEADER && __pim_get_bank_id() != 0)
    return;

  acc_cell_t cell;
  if (place_destination(dest_addr, &cell)) {
    note_acc_cell(&cell);
  } else {
    if (!last_acc.valid)
      return;
    /* Last resort. The pass names the destination tensor even when it cannot name the
     * element, so reaching here means arithmetic whose result never reaches any
     * argument. The bank is then a guess from trace order; say so once. */
    if (!warned.no_destination++)
      fprintf(stderr, "[simdram] WARN compute trace with no destination; charged at the "
                      "most recently touched accumulator's bank.\n");
    cell = last_acc.cell;
  }

  /* K-axis batch amortization: one MAJ-3 sequence per g_k_amort_batch compute traces
   * of an opcode, at the current (bank, row). This approximates OptiPIM's
   * `temporal_steps × 7n²+1` without collapsing distinct K iterations. */
  /* Amortize PER OPCODE: a MUL costs 7n^2+1 but an ADD only 8n+1, so batching
   * them on one counter and always charging MUL over-charges every accumulate
   * (half the instrumented sites) by ~14x. */
  int op_slot = (opcode >= 0 && opcode < SIMDRAM_OP_COUNT) ? opcode
                                                           : SIMDRAM_OP_OTHER;
  /* BATCH=1, which the derived batch reaches when the occupancy clamps to one, skips
   * the count: the modulo test would skip every op, so each op pays its own MAJ-3. */
  if (g_k_amort_batch > 1) {
    g_compute_trace_count_by_op[op_slot]++;
    if ((g_compute_trace_count_by_op[op_slot] % (uint64_t)g_k_amort_batch) != 1)
      return;
  }
  int cost = maj_cost(opcode);

  /* B-group rows start after D-group + C-group in each subarray. TRA
   * physically operates on B-group rows, not on data rows. */
  int cur_row_offset = 0;
  int b_group_base = cfg_data_rows_per_sa + SIMDRAM_C_GROUP_ROWS;

  for (int i = 0; i < cost; i++) {
    emit_trace("BR", cell.ch, cell.pch, cell.bg, cell.bank, cell.sa,
               b_group_base + cur_row_offset, 0);
    cur_row_offset = (cur_row_offset + 1) % SIMDRAM_B_GROUP_ROWS;
    stats.bank_reads++;
  }

  stats.compute_ops++;
}
