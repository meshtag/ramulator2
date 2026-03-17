#include "pim_runtime.h"
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

/* ================================================================
 *  Tensor registry
 * ================================================================ */
#define MAX_TENSORS 16

typedef struct {
  void *base_addr;
  size_t total_bytes;
  int elem_size;
  // The maximum number of dimensions is 4 for now.
  int dims[4];
  int ndims;
  int num_elements;
  pim_role_t role;

  /* Physical placement in HBM */
  int assigned_ch;
  int assigned_pch;
  int assigned_bg;
  int assigned_bank;
  int base_row; /* linear row across all subarrays in the bank */
  int values_per_col;
  int values_per_row;
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
#define MAX_BANKS 512
static int next_free_row[MAX_BANKS];

/* Statistics */
static uint64_t stat_bank_reads = 0;
static uint64_t stat_bank_writes = 0;
static uint64_t stat_reads = 0;
static uint64_t stat_writes = 0;
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

/*
 * Map a tensor element index to physical HBM coordinates.
 * base_row is a linear row across all subarrays in the bank;
 * we decompose it into (subarray, row-within-subarray).
 */
static void map_element(const tensor_info_t *t, int elem_idx, int *ch, int *pch,
                        int *bg, int *bank, int *sa, int *row, int *col) {
  *ch = t->assigned_ch;
  *pch = t->assigned_pch;
  *bg = t->assigned_bg;
  *bank = t->assigned_bank;

  int linear_row = t->base_row + (elem_idx / t->values_per_row);
  *sa = linear_row / cfg_num_rows;
  *row = linear_row % cfg_num_rows;
  *col = (elem_idx % t->values_per_row) / t->values_per_col;
}

/* ================================================================
 *  Public API
 * ================================================================ */

void pim_init(const char *trace_file) {
  /* If already initialized, close old trace and reinitialize */
  if (initialized && trace_fp) {
    fclose(trace_fp);
    trace_fp = NULL;
  }

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

  num_tensors = 0;
  cur_phase = PIM_PHASE_IDLE;
  memset(next_free_row, 0, sizeof(next_free_row));
  stat_bank_reads = stat_bank_writes = stat_reads = stat_writes = stat_ignored =
      0;
  initialized = 1;
  finalized = 0;

  fprintf(stderr, "[pim-runtime] Initialized. Trace: %s\n", trace_file);
  fprintf(stderr,
          "[pim-runtime] HBM config: %d ch, %d pch, %d bg, %d banks/bg, "
          "%d sa, %d rows, %d cols, %d-bit DQ\n",
          cfg_num_channels, cfg_num_pch, cfg_num_bg, cfg_num_banks, cfg_num_sa,
          cfg_num_rows, cfg_num_cols, cfg_dq_bits);
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
  t->ndims = ndims;
  t->role = role;

  int total = 1;
  for (int i = 0; i < ndims && i < 4; i++) {
    t->dims[i] = dims[i];
    total *= dims[i];
  }
  t->num_elements = total;
  t->total_bytes = (size_t)total * elem_size;

  /* Compute how many elements fit per column and per row */
  t->values_per_col = cfg_dq_bits / (elem_size * 8);
  // if (t->values_per_col < 1) t->values_per_col = 1;
  t->values_per_row = cfg_num_cols * t->values_per_col;

  /* Assign to a bank - round-robin across all ch/pch/bg/banks */
  int total_flat_banks =
      cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
  int global_bank = num_tensors % total_flat_banks;

  int banks_per_pch = cfg_num_bg * cfg_num_banks;
  int banks_per_ch = cfg_num_pch * banks_per_pch;

  t->assigned_ch = global_bank / banks_per_ch;
  t->assigned_pch = (global_bank / banks_per_pch) % cfg_num_pch;
  t->assigned_bg = (global_bank / cfg_num_banks) % cfg_num_bg;
  t->assigned_bank = global_bank % cfg_num_banks;

  /* Allocate linear rows (spanning subarrays) from this bank's pool */
  int rows_needed = (total + t->values_per_row - 1) / t->values_per_row;
  int rows_capacity = cfg_num_sa * cfg_num_rows;

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

  int first_sa = t->base_row / cfg_num_rows;
  int last_sa = (t->base_row + rows_needed - 1) / cfg_num_rows;

  const char *role_str = (role == PIM_ROLE_STREAMED)  ? "STREAMED"
                         : (role == PIM_ROLE_OPERAND) ? "OPERAND"
                                                      : "ACCUMULATOR";
  fprintf(stderr,
          "[pim-runtime] Tensor %d: %s, %d elems (%zu bytes), "
          "ch=%d pch=%d bg=%d bank=%d sa=[%d..%d] rows=[%d..%d]\n",
          num_tensors, role_str, total, t->total_bytes, t->assigned_ch,
          t->assigned_pch, t->assigned_bg, t->assigned_bank, first_sa, last_sa,
          t->base_row, t->base_row + rows_needed - 1);

  return num_tensors++;
}

void pim_set_phase(pim_phase_t phase) {
  const char *names[] = {"IDLE", "COMPUTE", "HOST"};
  if (phase <= PIM_PHASE_HOST) {
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
static void pim_trace_access(uint64_t addr, int is_write) {
  if (cur_phase == PIM_PHASE_IDLE) {
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

  int ch, pch, bg, bank, sa, row, col;
  map_element(t, elem_idx, &ch, &pch, &bg, &bank, &sa, &row, &col);

  if (cur_phase == PIM_PHASE_COMPUTE) {
    if (!is_write) {
      /* Load */
      switch (t->role) {
      case PIM_ROLE_STREAMED:
        emit_trace("BR", ch, pch, bg, bank, sa, row, col);
        stat_bank_reads++;
        break;
      case PIM_ROLE_OPERAND:
        emit_trace("W", ch, pch, bg, bank, sa, row, col);
        stat_writes++;
        break;
      case PIM_ROLE_ACCUMULATOR:
        emit_trace("R", ch, pch, bg, bank, sa, row, col);
        stat_reads++;
        break;
      }
    } else {
      /* Store */
      if (t->role == PIM_ROLE_ACCUMULATOR) {
        emit_trace("BW", ch, pch, bg, bank, sa, row, col);
        stat_bank_writes++;
      }
      /* Stores to STREAMED/OPERAND in COMPUTE phase are ignored (read-only) */
    }
  } else if (cur_phase == PIM_PHASE_HOST) {
    if (!is_write) {
      emit_trace("R", ch, pch, bg, bank, sa, row, col);
      stat_reads++;
    } else {
      emit_trace("W", ch, pch, bg, bank, sa, row, col);
      stat_writes++;
    }
  }
}

void __mem_trace_load(void *addr, uint64_t size) {
  if (!trace_fp)
    return;
  pim_trace_access((uint64_t)addr, 0);
}

void __mem_trace_store(void *addr, uint64_t size) {
  if (!trace_fp)
    return;
  pim_trace_access((uint64_t)addr, 1);
}
