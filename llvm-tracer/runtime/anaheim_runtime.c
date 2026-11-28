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
 * ramulator2_anaheim/anaheim/gen_trace.py. A polynomial is u32[limbs][banks][C][8], and a
 * tensor registered with two or more dims stacks dims[0] of them. PolyGroup g owns the rows
 * from g << 12. Its n polynomials split each row into column groups of
 * w = min(cg_max, row_chunks / next_pow2(n)) chunks, and limb l of its k-th polynomial wraps
 * over ceil(C / w) rows of column group k. PolyGroup -1 is a constant the PIM decoder
 * broadcasts, so its accesses issue nothing. A tensor the compiler splits
 * (__anaheim_polygroup_split, n > 0) puts each n of its stacked polynomials in a PolyGroup
 * of their own, from its PolyGroup g on, as a tensor of n polynomials in each.
 *
 * Arithmetic comes from ComputeTracePass. Every 8 elements of `% q` (urem) are one MMAC,
 * (+-a*b +- c) mod q on the unit's 8 lanes, and the mul, add and sub feeding it are part of
 * it. MMACs ride the run of reads just before them, one MMAC per read chunk, taking that
 * chunk as their bank operand as the unit's datapath does. A run is the consecutive reads of
 * one tensor, since the compiler splits a lane's block into one load per chunk. Any other
 * MMAC costs one PIM_CMP cycle. Any other operation is refused, since the unit has none, and
 * so is a program with more multiplies than reductions, since the MMAC reduces every product.
 *
 * The buffer holds every read chunk no MMAC took on arrival, for the rest of the program, and
 * each batch of results until it is written back. It keeps one copy of an address, so a chunk
 * read again while the buffer holds it lands in the held copy's entry, as a program that reads
 * a value twice needs it stored once. A program that needs more than B entries is refused,
 * since Anaheim has no spill.
 */
#define PIM_LAYOUT_TABLE_DEFINE
#include "im_runtime.h"
#include "simdram_runtime.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TENSORS 64
#define MAX_SLICES 64
#define CHUNK_WORDS 8 /* one 256-bit chunk of u32 */

typedef struct {
  void *base_addr;
  size_t total_bytes;
  int elem_size;
  int polygroup; /* -1 for a constant the decoder broadcasts, else a split tensor's first */
  int split;     /* polynomials per PolyGroup of a split tensor, 0 unsplit */
  int n_polys;   /* polynomials stacked along dims[0] */
  size_t poly_elems;
  int n_slices;  /* PolyGroups it spans */
  /* Per slice: the column group of its first polynomial within its PolyGroup, the chunks per
   * column group and the rows each limb takes. */
  int cg_index[MAX_SLICES];
  int cg_width[MAX_SLICES];
  int rows_per_limb[MAX_SLICES];
} anaheim_tensor_t;

/* What im_trace_common.h reads. */
static FILE *trace_fp = NULL;
static anaheim_tensor_t tensors[MAX_TENSORS];
static int num_tensors = 0;
static int cfg_num_pch = 1, cfg_num_bg = 1, cfg_num_banks = 16;
#include "im_trace_common.h"

/* The compiler's PolyGroups and splits, weak defaults overridden by the kernel's tables. */
__attribute__((weak)) extern const int32_t __anaheim_polygroups[1];
__attribute__((weak)) extern const int32_t __anaheim_polygroup_count;
__attribute__((weak)) extern const int32_t __anaheim_polygroup_split[1];
__attribute__((weak)) extern const int32_t __anaheim_polygroup_split_count;
const int32_t __anaheim_polygroups[1] = {0};
const int32_t __anaheim_polygroup_count = 0;
const int32_t __anaheim_polygroup_split[1] = {0};
const int32_t __anaheim_polygroup_split_count = 0;

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
static int rem_elems = 0;       /* urem elements not yet a whole chunk */
static int batch_chunks = 0;    /* the current run of writes, held in the buffer until then */
static long long mul_count = 0, rem_count = 0; /* elements, for the reduction check */

/* The leader's read chunks in order, each held until an MMAC takes it. The current run is the
 * chunks from run_next on, and MMACs take them first in, first out, as pending_chunks counts
 * them. held_addrs counts the addresses with a held chunk, through a table of held chunks per
 * address that a new generation empties at each program. */
typedef struct {
  uint64_t addr;
  int held;
} chunk_t;
static chunk_t *reads = NULL;
static size_t n_reads = 0, reads_cap = 0, run_next = 0;
#define HELD_SLOTS (1 << 16)
static uint64_t held_key[HELD_SLOTS];
static uint32_t held_gen[HELD_SLOTS];
static int held_count[HELD_SLOTS];
static uint32_t cur_gen = 1;
static int held_addrs = 0;

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

/* The buffer holds the reads no MMAC took, one entry per address, and the current batch of
 * results. */
static void check_buffer(void) {
  int used = held_addrs + batch_chunks;
  if (used > s_max_buffer)
    s_max_buffer = used;
  if (used > cfg_buffer) {
    fprintf(stderr, "[anaheim] ERROR: a program holds %d chunks (%d read ahead, %d results) "
                    "in a buffer of %d entries, and Anaheim's unit cannot spill. Shrink the "
                    "tile.\n", used, held_addrs, batch_chunks, cfg_buffer);
    exit(1);
  }
}

/* The held chunks of addr, by linear probing in this program's generation. */
static int *held_slot(uint64_t addr) {
  uint32_t h = (uint32_t)((addr * 0x9E3779B97F4A7C15ull) >> 48);
  for (;;) {
    if (held_gen[h] != cur_gen) {
      held_gen[h] = cur_gen;
      held_key[h] = addr;
      held_count[h] = 0;
      return &held_count[h];
    }
    if (held_key[h] == addr)
      return &held_count[h];
    h = (h + 1) & (HELD_SLOTS - 1);
  }
}

static void hold_read(uint64_t addr) {
  if (n_reads == reads_cap) {
    reads_cap = reads_cap ? 2 * reads_cap : 256;
    reads = (chunk_t *)realloc(reads, reads_cap * sizeof(chunk_t));
  }
  reads[n_reads++] = (chunk_t){addr, 1};
  if ((*held_slot(addr))++ == 0)
    held_addrs++;
}

/* MMACs took n chunks of the current run. */
static void take_from_run(int n) {
  for (; run_next < n_reads && n > 0; run_next++) {
    if (!reads[run_next].held)
      continue;
    reads[run_next].held = 0;
    if (--(*held_slot(reads[run_next].addr)) == 0)
      held_addrs--;
    n--;
  }
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
  pending_chunks = 0;
  run_next = n_reads;
  check_buffer();
  if (mul_count > rem_count) {
    fprintf(stderr, "[anaheim] ERROR: a program multiplies %lld elements but reduces %lld. "
                    "Anaheim's MMAC reduces every product, so write `%% q` after each "
                    "multiply.\n", mul_count, rem_count);
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
    pending_chunks = rem_elems = batch_chunks = held_addrs = 0;
    pending_tensor = -1;
    n_reads = run_next = 0;
    cur_gen++;
    mul_count = rem_count = 0;
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

static void place(const anaheim_tensor_t *t, uint64_t elem, int *limb, int *bank, int *row,
                  int *col) {
  int poly = (int)(elem / t->poly_elems);
  int s = t->split ? poly / t->split : 0;
  if (t->split)
    poly %= t->split;
  size_t chunk_idx = (elem % t->poly_elems) / CHUNK_WORDS;
  size_t per_bank = (size_t)cfg_chunks_per_limb;
  *limb = (int)(chunk_idx / (per_bank * cfg_num_banks));
  *bank = (int)((chunk_idx / per_bank) % cfg_num_banks);
  int chunk = (int)(chunk_idx % per_bank);
  *row = ((t->polygroup + s) << 12) + *limb * t->rows_per_limb[s] + chunk / t->cg_width[s];
  *col = (t->cg_index[s] + poly) * t->cg_width[s] + chunk % t->cg_width[s];
}

/* Slice s of tensor t: its PolyGroup and polynomial count. */
static int slice_group(const anaheim_tensor_t *t, int s) {
  return t->polygroup + (t->split ? s : 0);
}
static int slice_polys(const anaheim_tensor_t *t) {
  return t->split ? t->split : t->n_polys;
}

/* Lay out every PolyGroup once all its tensors are registered, a split tensor's slices as
 * tensors of their own, in registration order. */
static void layout_polygroups(void) {
  for (int i = 0; i < num_tensors; i++) {
    anaheim_tensor_t *t = &tensors[i];
    if (t->polygroup < 0)
      continue;
    for (int s = 0; s < t->n_slices; s++) {
      int g = slice_group(t, s), n = 0, k = 0;
      for (int j = 0; j < num_tensors; j++) {
        const anaheim_tensor_t *u = &tensors[j];
        for (int r = 0; u->polygroup >= 0 && r < u->n_slices; r++) {
          if (slice_group(u, r) != g)
            continue;
          if (j < i || (j == i && r < s))
            k += slice_polys(u);
          n += slice_polys(u);
        }
      }
      int w = cfg_row_chunks / next_pow2(n);
      t->cg_width[s] = w < cfg_cg_max ? w : cfg_cg_max;
      if (t->cg_width[s] < 1)
        die("a PolyGroup with more polynomials than chunks in a row");
      t->cg_index[s] = k;
      t->rows_per_limb[s] = (cfg_chunks_per_limb + t->cg_width[s] - 1) / t->cg_width[s];
    }
  }
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
  if (lane == 0 && !is_write) {
    batch_chunks = 0; /* the batch went back to the banks */
    if (tid == pending_tensor) {
      pending_chunks += nchunks;
    } else {
      run_next = n_reads; /* the last run's chunks no MMAC took stay held */
      pending_chunks = nchunks;
      pending_tensor = tid;
    }
  }
  for (int j = 0; j < nchunks; j++) {
    int limb, bank, row, col;
    place(t, first + (uint64_t)j * CHUNK_WORDS, &limb, &bank, &row, &col);
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
      uint64_t chunk_addr = ((uint64_t)row << 24) | ((uint64_t)col << 8);
      fprintf(trace_fp, "%s 0x%llx\n", is_write ? "PIM_WR_AB" : "PIM_RD_AB",
              (unsigned long long)chunk_addr);
      is_write ? s_wr++ : s_rd++;
      if (!is_write)
        hold_read(chunk_addr);
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
  if (lane == 0 && is_write) {
    pending_chunks = 0; /* the run's chunks no MMAC took stay held */
    pending_tensor = -1;
    run_next = n_reads;
    batch_chunks += nchunks;
    check_buffer();
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
    return; /* part of the MMAC that reduces it */
  case SIMDRAM_OP_MUL:
    if (lane == 0)
      mul_count += lanes;
    return;
  case SIMDRAM_OP_REM:
    break;
  default:
    fprintf(stderr, "[anaheim] ERROR: compute op %d has no form on Anaheim's MMAC lanes, "
                    "which do only (+-a*b +- c) mod q\n", op);
    exit(1);
  }
  if (lane != 0)
    return;
  rem_count += lanes;
  batch_chunks = 0;
  rem_elems += lanes;
  int chunks = rem_elems / CHUNK_WORDS;
  rem_elems %= CHUNK_WORDS;
  int paired = chunks < pending_chunks ? chunks : pending_chunks;
  pending_chunks -= paired;
  take_from_run(paired);
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
  t->split = id < (int)__anaheim_polygroup_split_count ? (int)__anaheim_polygroup_split[id] : 0;
  t->n_polys = ndims >= 2 ? dims[0] : 1;
  t->poly_elems = total / (size_t)t->n_polys;
  t->n_slices = t->split ? t->n_polys / t->split : 1;
  if (t->polygroup >= 0 && elem_size != 4)
    die("a banked tensor that is not u32. Anaheim stores 32-bit words");
  if (t->split && (t->polygroup < 0 || t->n_polys % t->split))
    die("a split tensor whose polynomials do not divide into its PolyGroups");
  if (t->n_slices > MAX_SLICES)
    die("a tensor split over more PolyGroups than the runtime holds");
  return id;
}

void anaheim_set_phase(int p) {
  if (p && !phase)
    layout_polygroups();
  phase = p;
}

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
  free(reads);
  reads = NULL;
  reads_cap = n_reads = run_next = 0;
}
