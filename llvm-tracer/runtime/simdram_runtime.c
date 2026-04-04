#include "simdram_runtime.h"
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
 * SIMDRAM computation: multiply = 7*pe_bits^2+1 MAJ-3 gate operations,
 * emitted as bank-read (BR) ops cycling through pe_bits rows.
 * Matches OptiPIM's SimDRAMCodeGen exactly.
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
static int cfg_num_rows = 512;
static int cfg_num_cols = 64;
static int cfg_dq_bits = 128;
static int cfg_pe_bits = 16;

/* ================================================================
 *  Tensor registry
 * ================================================================ */
#define MAX_TENSORS 16

typedef struct {
  void *base_addr;
  size_t total_bytes;
  int elem_size;
  int dims[4];
  int ndims;
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
static int find_tensor(uint64_t addr) {
  for (int i = 0; i < num_tensors; i++) {
    uint64_t base = (uint64_t)tensors[i].base_addr;
    if (addr >= base && addr < base + tensors[i].total_bytes)
      return i;
  }
  return -1;
}

/*
 * Map a tensor element to physical HBM coordinates for SIMDRAM.
 *
 * Bit-serial layout: each element occupies pe_bits consecutive rows.
 * Within a "row group" (pe_bits rows), each column stores dq_bits values.
 * Returns the base row of the pe_bits-row group containing this element.
 */
static void map_element(const tensor_info_t *t, int elem_idx, int *ch, int *pch,
                        int *bg, int *bank, int *sa, int *base_row, int *col) {
  *ch = t->assigned_ch;
  *pch = t->assigned_pch;
  *bg = t->assigned_bg;
  *bank = t->assigned_bank;

  int row_group_idx = elem_idx / t->elems_per_row_group;
  *col = (elem_idx % t->elems_per_row_group) / t->elems_per_col_slot;

  int linear_base = t->base_row + row_group_idx * cfg_pe_bits;
  *sa = linear_base / cfg_num_rows;
  *base_row = linear_base % cfg_num_rows;
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
 *   multiplication  11n² − 5n − 1       Quadratic
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
    return 11 * n * n - 5 * n - 1;
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

  num_tensors = 0;
  cur_phase = SIMDRAM_PHASE_IDLE;
  memset(next_free_row, 0, sizeof(next_free_row));
  stat_bank_reads = stat_bank_writes = stat_reads = stat_writes = 0;
  stat_compute_ops = stat_ignored = 0;
  num_acc_trackers = 0;
  current_acc = -1;
  initialized = 1;
  finalized = 0;

  fprintf(stderr, "[simdram] Initialized. Trace: %s\n", trace_file);
  fprintf(stderr,
          "[simdram] HBM config: %d ch, %d pch, %d bg, %d banks/bg, "
          "%d sa, %d rows, %d cols, %d-bit DQ, pe_bits=%d\n",
          cfg_num_channels, cfg_num_pch, cfg_num_bg, cfg_num_banks, cfg_num_sa,
          cfg_num_rows, cfg_num_cols, cfg_dq_bits, cfg_pe_bits);
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
  t->ndims = ndims;
  t->role = role;

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

  /* Assign bank — round-robin across all ch/pch/bg/banks */
  int total_flat_banks =
      cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
  int global_bank = num_tensors % total_flat_banks;

  int banks_per_pch = cfg_num_bg * cfg_num_banks;
  int banks_per_ch = cfg_num_pch * banks_per_pch;

  t->assigned_ch = global_bank / banks_per_ch;
  t->assigned_pch = (global_bank / banks_per_pch) % cfg_num_pch;
  t->assigned_bg = (global_bank / cfg_num_banks) % cfg_num_bg;
  t->assigned_bank = global_bank % cfg_num_banks;

  /* Linear rows needed: ceil(total / elems_per_row_group) × pe_bits */
  int row_groups =
      (total + t->elems_per_row_group - 1) / t->elems_per_row_group;
  int rows_needed = row_groups * cfg_pe_bits;
  int rows_capacity = cfg_num_sa * cfg_num_rows;

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
    acc_trackers[idx].sa = t->base_row / cfg_num_rows;
    acc_trackers[idx].row = t->base_row % cfg_num_rows;
    if (current_acc < 0)
      current_acc = idx; /* default to first registered accumulator */
  }

  int first_sa = t->base_row / cfg_num_rows;
  int last_sa = (t->base_row + rows_needed - 1) / cfg_num_rows;

  const char *role_str = (role == SIMDRAM_ROLE_STREAMED)  ? "STREAMED"
                         : (role == SIMDRAM_ROLE_OPERAND) ? "OPERAND"
                                                          : "ACCUMULATOR";
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

  if (cur_phase == SIMDRAM_PHASE_COMPUTE) {
    if (!is_write) {
      /* Load */
      switch (t->role) {
      case SIMDRAM_ROLE_STREAMED:
        /* Input: write to subarray bit-rows (bit-serial) */
        for (int bit = 0; bit < cfg_pe_bits; bit++) {
          emit_trace("W", ch, pch, bg, bank, sa, base_row + bit, col);
          stat_writes++;
        }
        break;
      case SIMDRAM_ROLE_OPERAND:
        /* Weight: pre-stored in DRAM, nothing to emit */
        break;
      case SIMDRAM_ROLE_ACCUMULATOR:
        /* Output: read partial sums from bit-rows */
        for (int bit = 0; bit < cfg_pe_bits; bit++) {
          emit_trace("R", ch, pch, bg, bank, sa, base_row + bit, col);
          stat_reads++;
        }
        /* Track accumulator location for __compute_trace */
        if (t->acc_tracker_idx >= 0) {
          acc_tracker_t *at = &acc_trackers[t->acc_tracker_idx];
          at->ch = ch;
          at->pch = pch;
          at->bg = bg;
          at->bank = bank;
          at->sa = sa;
          at->row = base_row;
          current_acc = t->acc_tracker_idx;
        }
        break;
      }
    } else {
      /* Store — in SIMDRAM, results stay in subarray from MAJ ops.
       * Only update accumulator location tracking. */
      if (t->role == SIMDRAM_ROLE_ACCUMULATOR && t->acc_tracker_idx >= 0) {
        acc_tracker_t *at = &acc_trackers[t->acc_tracker_idx];
        at->ch = ch;
        at->pch = pch;
        at->bg = bg;
        at->bank = bank;
        at->sa = sa;
        at->row = base_row;
        current_acc = t->acc_tracker_idx;
      }
    }
  } else if (cur_phase == SIMDRAM_PHASE_HOST) {
    if (!is_write) {
      emit_trace("R", ch, pch, bg, bank, sa, base_row, col);
      stat_reads++;
    } else {
      emit_trace("W", ch, pch, bg, bank, sa, base_row, col);
      stat_writes++;
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
void __compute_trace(int32_t opcode, int32_t bit_width) {
  if (!trace_fp || cur_phase != SIMDRAM_PHASE_COMPUTE)
    return;
  if (current_acc < 0)
    return;

  const acc_tracker_t *at = &acc_trackers[current_acc];
  int cost = maj_cost(opcode);
  int cur_row_offset = 0;

  for (int i = 0; i < cost; i++) {
    emit_trace("BR", at->ch, at->pch, at->bg, at->bank, at->sa,
               at->row + cur_row_offset, 0);
    cur_row_offset = (cur_row_offset + 1) % cfg_pe_bits;
    stat_bank_reads++;
  }

  stat_compute_ops++;
}
