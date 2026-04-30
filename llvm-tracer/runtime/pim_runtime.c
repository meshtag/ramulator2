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
  int ndims;
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

/* Statistics */
static uint64_t stat_bank_reads = 0;
static uint64_t stat_bank_writes = 0;
static uint64_t stat_reads = 0;
static uint64_t stat_writes = 0;
static uint64_t stat_ignored = 0;
static uint64_t stat_store_coalesced = 0; /* per-call BWs collapsed when a
                                             vector store had multiple lanes
                                             land on the same (bank,row,col) */

/* Physical-address dedup: collapses spatially-redundant reads (multiple PEs
 * in a bank-group hitting the same row buffer) into one trace line per
 * unique (global_bank, sa+row, col). Resets per program-id. See
 * im_addr_dedup.h. */
static addr_dedup_state_t *g_dedup = NULL;
static int g_dedup_enabled = 1;

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
   * resets). */
  if (g_dedup) {
    addr_dedup_reset(g_dedup);
  }
  if (g_dedup_store_perpid) {
    addr_dedup_reset(g_dedup_store_perpid);
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

/*
 * Map a tensor element index to physical HBM coordinates.
 * Each logical tensor row is striped across flat banks round-robin so the
 * memory system can use the full bank fabric rather than hot-spotting one bank
 * per tensor.
 */
static void map_element(const tensor_info_t *t, int elem_idx, int *ch, int *pch,
                        int *bg, int *bank, int *sa, int *row, int *col) {
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

  num_tensors = 0;
  cur_phase = PIM_PHASE_IDLE;
  memset(next_free_row, 0, sizeof(next_free_row));
  stat_bank_reads = stat_bank_writes = stat_reads = stat_writes = stat_ignored =
      0;
  last_program_id = -1;
  last_program_id_y = -1;
  last_program_id_z = -1;

  /* Allow disabling the physical-address dedup for ablation: IM_DEDUP=0. */
  const char *dedup_env = getenv("IM_DEDUP");
  g_dedup_enabled = (dedup_env && dedup_env[0] == '0') ? 0 : 1;

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
  if (g_dedup_enabled) {
    int perpid_cap = 16384;
    int persistent_cap = 262144;
    if ((v = getenv("IM_DEDUP_CAP"))      ) perpid_cap = atoi(v);
    if ((v = getenv("IM_PERSISTENT_CAP")) ) persistent_cap = atoi(v);
    if (perpid_cap < 16) perpid_cap = 16;
    if (persistent_cap < 16) persistent_cap = 16;

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
   *   - Only in COMPUTE phase.
   *   - Only for STREAMED/OPERAND loads. ACCUMULATOR loads are partial-sum
   *     reads that the kernel computes into; they are not idempotent
   *     re-reads and so are not deduped here.
   *   - Stores are handled separately (per-program-id store dedup, below). */
  if (g_dedup_enabled && g_dedup && cur_phase == PIM_PHASE_COMPUTE &&
      !is_write &&
      (t->role == PIM_ROLE_STREAMED || t->role == PIM_ROLE_OPERAND)) {
    int global_bank = compute_global_bank(loc.ch, loc.pch, loc.bg, loc.bank);
    uint64_t linear_row =
        (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    if (!addr_dedup_check_and_mark(g_dedup, (uint64_t)global_bank, linear_row,
                                   (uint64_t)loc.col)) {
      t->dedup_skips++;
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

  emit_access_by_role_phase(t, &loc, is_write);
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

  if (g_dedup_enabled && dedup_state && cur_phase == PIM_PHASE_COMPUTE &&
      (t->role == PIM_ROLE_STREAMED || t->role == PIM_ROLE_OPERAND)) {
    int global_bank = compute_global_bank(loc.ch, loc.pch, loc.bg, loc.bank);
    uint64_t linear_row =
        (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    if (!addr_dedup_check_and_mark(dedup_state, (uint64_t)global_bank,
                                   linear_row, (uint64_t)loc.col)) {
      t->persistent_skips++;
      if (state_skips)
        (*state_skips)++;
      return;
    }
  }

  emit_access_by_role_phase(t, &loc, 0);
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
    if (g_dedup_enabled && g_dedup_store_perpid) {
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

/* Explicit reset of all persistent dedup states, e.g. when leaving a logical
 * tile region. Currently unused (phase changes provide an implicit reset),
 * but exposed so future MLIR-level work can scope persistent-load regions
 * more tightly than the COMPUTE phase. */
void __pim_scope_release(int32_t scope_tag) {
  (void)scope_tag;
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
}
