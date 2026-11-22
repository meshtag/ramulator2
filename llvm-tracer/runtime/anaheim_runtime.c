/* anaheim_runtime.c: TritonIM's trace runtime for Anaheim's near-bank PIM unit (Kim et al.,
 * HPCA'25), the anaheim target of benchmarks/pipeline.py.
 *
 * The lanes are the banks of one channel. Anaheim runs every bank of a channel in lockstep,
 * so an access that all lanes make at the same point of a program is ONE all-bank command.
 * Bank 0's replay emits the commands, and every other bank's replay must make the same
 * accesses in the same order, or the run is refused. The trace is ramulator2_anaheim's
 * AnaheimPIMTrace format, one command per line:
 *   PIM_RD_AB <addr>  one 256-bit chunk from every bank into its unit
 *   PIM_WR_AB <addr>  one buffer entry back to every bank
 *   PIM_CMP <addr>    one PIM cycle of MMACs on buffered operands only
 * with addr = row << 24 | chunk << 8 | channel, the simulator's AnaheimLinear layout.
 *
 * Placement is the compiler's: im_polygroups, emitted as __anaheim_polygroups with placement
 * scheme 3. It is Anaheim's column partitioning (Sec. VI-B), the formula of
 * ramulator2_anaheim/anaheim/gen_trace.py. A tensor is u32[limbs][banks][C][8]. PolyGroup g
 * owns the rows from g << 12. Its n tensors split each row into column groups of
 * w = min(cg_max, row_chunks / next_pow2(n)) chunks, and limb l of its k-th tensor wraps
 * over ceil(C / w) rows of column group k. PolyGroup -1 is a constant the PIM decoder
 * broadcasts, so its accesses issue nothing.
 *
 * Arithmetic comes from ComputeTracePass. Every 8 elements of `% q` (urem) are one MMAC,
 * (+-a*b +- c) mod q on the unit's 8 lanes, and the mul, add and sub feeding it are part of
 * it. MMACs ride the run of reads just before them, one MMAC per read chunk, taking that
 * chunk as their bank operand as the unit's datapath does. A run is the consecutive reads of
 * one tensor, since the compiler splits a lane's block into one load per chunk. Any other
 * MMAC costs one PIM_CMP cycle. Any other operation is refused, since the unit has none.
 *
 * The buffer holds every read chunk no MMAC took on arrival, plus the results the program
 * writes back. A program that needs more than B entries is refused, since Anaheim has no
 * spill.
 */
#define PIM_LAYOUT_TABLE_DEFINE
#include "im_runtime.h"
#include "simdram_runtime.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TENSORS 64
#define CHUNK_WORDS 8 /* one 256-bit chunk of u32 */

typedef struct {
  void *base_addr;
  size_t total_bytes;
  int elem_size;
  int polygroup; /* -1: broadcast by the decoder, in no bank */
  int cg_index;  /* its column group within the PolyGroup */
  int cg_width;  /* chunks per column group */
  int rows_per_limb;
} anaheim_tensor_t;

/* What im_trace_common.h reads. */
static FILE *trace_fp = NULL;
static anaheim_tensor_t tensors[MAX_TENSORS];
static int num_tensors = 0;
static int cfg_num_pch = 1, cfg_num_bg = 1, cfg_num_banks = 16;
#include "im_trace_common.h"

/* The compiler's PolyGroups, weak defaults overridden by the kernel's table. */
__attribute__((weak)) extern const int32_t __anaheim_polygroups[1];
__attribute__((weak)) extern const int32_t __anaheim_polygroup_count;
const int32_t __anaheim_polygroups[1] = {0};
const int32_t __anaheim_polygroup_count = 0;

/* The machine, from anaheim_set_machine. */
static int cfg_chunks_per_limb = 32;
static int cfg_row_chunks = 64;
static int cfg_cg_max = 16;
static int cfg_buffer = 32;

/* The leader's accesses in the current program, which every other lane must repeat. */
typedef struct {
  int is_write, row, col;
} access_t;
static access_t *prog = NULL;
static size_t prog_len = 0, prog_cap = 0;
static size_t lane_pos = 0;
static uint64_t cur_epoch = (uint64_t)-1;
static int cur_lane = -1;

/* The leader's MMAC and buffer accounting for the current program. */
static int pending_chunks = 0; /* chunks of the current read run no MMAC has taken */
static int pending_tensor = -1; /* the run's tensor, -1 once an MMAC or a write ends it */
static int bypass_chunks = 0;   /* read chunks no MMAC took on arrival, held in the buffer */
static int rem_elems = 0;       /* urem elements not yet a whole chunk */
static int acc_chunks = 0;      /* results written back, held in the buffer until then */

static long long s_rd, s_wr, s_cmp, s_mmac, s_paired, s_programs;
static int s_max_buffer;
static int phase;

static void die(const char *msg) {
  fprintf(stderr, "[anaheim] ERROR: %s\n", msg);
  exit(1);
}

void anaheim_set_machine(int banks, int chunks_per_limb, int row_chunks, int cg_max,
                         int buffer) {
  cfg_num_banks = banks;
  cfg_chunks_per_limb = chunks_per_limb;
  cfg_row_chunks = row_chunks;
  cfg_cg_max = cg_max;
  cfg_buffer = buffer;
}

static int next_pow2(int n) {
  int p = 1;
  while (p < n)
    p <<= 1;
  return p;
}

/* Close out the leader's program: unconsumed reads stay in the buffer, which must fit. */
static void finish_program(void) {
  if (cur_epoch == (uint64_t)-1)
    return;
  if (cur_lane > 0 && lane_pos != prog_len) {
    fprintf(stderr, "[anaheim] ERROR: lane %d made %zu bank accesses where bank 0 made %zu, "
                    "so the banks would not run in lockstep\n", cur_lane, lane_pos, prog_len);
    exit(1);
  }
  bypass_chunks += pending_chunks;
  int used = bypass_chunks + acc_chunks;
  if (used > s_max_buffer)
    s_max_buffer = used;
  if (used > cfg_buffer) {
    fprintf(stderr, "[anaheim] ERROR: a program holds %d chunks (%d read ahead, %d MMAC "
                    "results) in a buffer of %d entries, and Anaheim's unit cannot spill. "
                    "Shrink the tile.\n", used, bypass_chunks, acc_chunks, cfg_buffer);
    exit(1);
  }
  s_programs++;
}

/* Start a new program at each program-id change, and a new lane at each bank change. */
static int sync_program(void) {
  int lane = __pim_get_bank_id();
  if (__pim_program_epoch != cur_epoch) {
    finish_program();
    cur_epoch = __pim_program_epoch;
    prog_len = 0;
    cur_lane = -1;
    pending_chunks = bypass_chunks = rem_elems = acc_chunks = 0;
    pending_tensor = -1;
  }
  if (lane != cur_lane) {
    if (cur_lane > 0 && lane_pos != prog_len) {
      fprintf(stderr, "[anaheim] ERROR: lane %d made %zu bank accesses where bank 0 made "
                      "%zu, so the banks would not run in lockstep\n",
              cur_lane, lane_pos, prog_len);
      exit(1);
    }
    cur_lane = lane;
    lane_pos = 0;
  }
  return lane;
}

static void place(const anaheim_tensor_t *t, size_t chunk_idx, int *limb, int *bank,
                  int *row, int *col) {
  size_t per_bank = (size_t)cfg_chunks_per_limb;
  *limb = (int)(chunk_idx / (per_bank * cfg_num_banks));
  *bank = (int)((chunk_idx / per_bank) % cfg_num_banks);
  int chunk = (int)(chunk_idx % per_bank);
  *row = (t->polygroup << 12) + *limb * t->rows_per_limb + chunk / t->cg_width;
  *col = t->cg_index * t->cg_width + chunk % t->cg_width;
}

static void trace_access_range(uint64_t addr, uint64_t size, int is_write, uint64_t lanes) {
  (void)lanes;
  if (phase == 0)
    return;
  int lane = sync_program();
  int tid = find_tensor(addr);
  if (tid < 0)
    die("an access outside every registered tensor");
  const anaheim_tensor_t *t = &tensors[tid];
  if (t->polygroup < 0)
    return; /* the decoder broadcasts it */
  uint64_t first = (addr - (uint64_t)t->base_addr) / (uint64_t)t->elem_size;
  uint64_t n = size / (uint64_t)t->elem_size;
  if (first % CHUNK_WORDS || n % CHUNK_WORDS)
    die("an access that is not whole 256-bit chunks. Size each lane's block in multiples "
        "of 8 u32 and compile with im_dq_bits 256");
  int nchunks = (int)(n / CHUNK_WORDS);
  for (int j = 0; j < nchunks; j++) {
    int limb, bank, row, col;
    place(t, first / CHUNK_WORDS + (uint64_t)j, &limb, &bank, &row, &col);
    if (bank != lane) {
      fprintf(stderr, "[anaheim] ERROR: lane %d touched bank %d of tensor %d. Put the banks "
                      "on the lane axis (im_bank_axis).\n", lane, bank, tid);
      exit(1);
    }
    if (lane == 0) {
      if (prog_len == prog_cap) {
        prog_cap = prog_cap ? 2 * prog_cap : 256;
        prog = (access_t *)realloc(prog, prog_cap * sizeof(access_t));
      }
      prog[prog_len++] = (access_t){is_write, row, col};
      fprintf(trace_fp, "%s 0x%llx\n", is_write ? "PIM_WR_AB" : "PIM_RD_AB",
              (unsigned long long)(((uint64_t)row << 24) | ((uint64_t)col << 8)));
      is_write ? s_wr++ : s_rd++;
    } else {
      const access_t *a = lane_pos < prog_len ? &prog[lane_pos] : NULL;
      if (!a || a->is_write != is_write || a->row != row || a->col != col) {
        fprintf(stderr, "[anaheim] ERROR: lane %d's access %zu differs from bank 0's, so the "
                        "banks would not run in lockstep\n", lane, lane_pos);
        exit(1);
      }
      lane_pos++;
    }
  }
  if (lane == 0) {
    if (is_write) {
      bypass_chunks += pending_chunks;
      pending_chunks = 0;
      pending_tensor = -1;
      acc_chunks += nchunks;
    } else if (tid == pending_tensor) {
      pending_chunks += nchunks;
    } else {
      bypass_chunks += pending_chunks;
      pending_chunks = nchunks;
      pending_tensor = tid;
    }
  }
}

void __compute_trace(int32_t op, int32_t lanes, void *dest_addr) {
  (void)dest_addr;
  if (!trace_fp || phase == 0)
    return;
  int lane = sync_program();
  switch (op) {
  case SIMDRAM_OP_ADD:
  case SIMDRAM_OP_SUB:
  case SIMDRAM_OP_MUL:
    return; /* part of the MMAC that reduces it */
  case SIMDRAM_OP_REM:
    break;
  default:
    fprintf(stderr, "[anaheim] ERROR: compute op %d has no form on Anaheim's MMAC lanes, "
                    "which do only (+-a*b +- c) mod q\n", op);
    exit(1);
  }
  if (lane != 0)
    return;
  rem_elems += lanes;
  int chunks = rem_elems / CHUNK_WORDS;
  rem_elems %= CHUNK_WORDS;
  int paired = chunks < pending_chunks ? chunks : pending_chunks;
  pending_chunks -= paired;
  if (chunks)
    pending_tensor = -1; /* the run is over, so the next read starts another */
  s_mmac += chunks;
  s_paired += paired;
  for (int i = paired; i < chunks; i++) {
    fprintf(trace_fp, "PIM_CMP 0x0\n");
    s_cmp++;
  }
}

void anaheim_init(const char *trace_file) {
  trace_fp = fopen(trace_file, "w");
  if (!trace_fp)
    die("cannot open the trace file");
  num_tensors = 0;
  prog_len = lane_pos = 0;
  cur_epoch = (uint64_t)-1;
  cur_lane = -1;
  s_rd = s_wr = s_cmp = s_mmac = s_paired = s_programs = 0;
  s_max_buffer = 0;
  phase = 0;
  if (__pim_layout_scheme && __pim_layout_scheme != 3) {
    fprintf(stderr, "[anaheim] ERROR: the kernel was placed for scheme %d, and this runtime "
                    "takes only Anaheim's column partitioning (scheme 3, im_polygroups)\n",
            (int)__pim_layout_scheme);
    exit(1);
  }
  check_compiler_dq_bits("anaheim", 256);
}

int anaheim_register_tensor(void *ptr, const int *dims, int ndims, int elem_size, int role) {
  (void)role;
  if (num_tensors >= MAX_TENSORS)
    die("too many tensors");
  int id = num_tensors;
  if (id >= (int)__anaheim_polygroup_count)
    die("a tensor without a PolyGroup. Compile with im_polygroups, one per pointer argument");
  anaheim_tensor_t *t = &tensors[num_tensors++];
  size_t total = 1;
  for (int i = 0; i < ndims; i++)
    total *= (size_t)dims[i];
  t->base_addr = ptr;
  t->elem_size = elem_size;
  t->total_bytes = total * (size_t)elem_size;
  t->polygroup = (int)__anaheim_polygroups[id];
  if (t->polygroup >= 0) {
    if (elem_size != 4)
      die("a banked tensor that is not u32. Anaheim stores 32-bit words");
    int n = 0, k = 0;
    for (int i = 0; i < (int)__anaheim_polygroup_count; i++) {
      if (__anaheim_polygroups[i] != t->polygroup)
        continue;
      if (i < id)
        k++;
      n++;
    }
    int w = cfg_row_chunks / next_pow2(n);
    t->cg_width = w < cfg_cg_max ? w : cfg_cg_max;
    t->cg_index = k;
    t->rows_per_limb = (cfg_chunks_per_limb + t->cg_width - 1) / t->cg_width;
  }
  return id;
}

void anaheim_set_phase(int p) { phase = p; }

void anaheim_finalize(void) {
  finish_program();
  cur_epoch = (uint64_t)-1;
  if (trace_fp) {
    fclose(trace_fp);
    trace_fp = NULL;
  }
  fprintf(stderr, "[anaheim] %lld programs: %lld PIM_RD_AB, %lld PIM_WR_AB, %lld PIM_CMP; "
                  "%lld MMAC chunks, %lld on their read; buffer peak %d of %d\n",
          s_programs, s_rd, s_wr, s_cmp, s_mmac, s_paired, s_max_buffer, cfg_buffer);
  free(prog);
  prog = NULL;
  prog_cap = 0;
}
