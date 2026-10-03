#include "pim_runtime.h"
#include "im_addr_dedup.h"
#include "im_runtime.h"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define PIM_LAYOUT_TABLE_DEFINE
#include "pim_layout_table.h"

/*
 * PIM Runtime Library, DCC-parity tracer.
 *
 * Intercepts LLVM-instrumented loads/stores and emits DCC's opcodes at DCC's
 * addresses, for DCC's Ramulator build. Only PIM_TRACE_FORMAT=dcc is supported.
 */

/* ================================================================
 *  HBM3_PIM organization, fixed to the DCC-matched machine
 * ================================================================ */
static const int cfg_num_channels = 1;
static const int cfg_num_pch = 2;
static const int cfg_num_bg = 4;
static const int cfg_num_banks = 4; /* per bank group */
/* Hand copy of the simulator's org preset (HBM3_8Gb in HBM3_PIM.cpp). Nothing checks
 * the two agree. */
static const int cfg_num_sa = 64;
static const int cfg_num_rows = 512;
/* A column command moves prefetch 2 x dq 128 = 256 bits, so columns count 256-bit words.
 * cfg_num_cols and cfg_dq_bits move together, and with the simulator org. */
static const int cfg_num_cols = 32;
static const int cfg_dq_bits = 256;
static void check_compiler_dq_bits(int cfg_bits);

/* Modelled value width, a hardware property (the spec's pe_bits), not the host dtype.
 * Column packing is dq_bits / data_width. */
static const int cfg_data_width_bits = 16;

/* Where each tensor starts. DQ aligns to the data-bus word only. ROW_PACK applies to
 * lane slabs: each tensor reserves exactly its per-lane share in one shared column space,
 * so small tensors sit side by side in the same rows. */
typedef enum { PIM_ALIGN_DQ = 1, PIM_ALIGN_ROW_PACK = 3 } pim_place_align_t;
static pim_place_align_t cfg_place_align = PIM_ALIGN_DQ;

/* Next free position in the global linear-element space. Tensors are appended and
 * aligned to a values_per_col boundary only, so they share rows on different banks. */
static uint64_t g_interleaved_next_linear = 0;
/* Lane slabs take rows from the TOP of the per-bank row space; interleaved placement
 * grows from the bottom, and the two must never meet. */
static int g_lane_row_top = -1;
static long g_pack_col_top = -1;   /* row-pack: next free column, counted down */
static uint64_t g_pack_overflow = 0; /* accesses beyond a lane's share, reported as uncovered */
static uint64_t g_emit_drops = 0;    /* compute accesses that reached no record */
static uint64_t stat_lane_placed = 0;

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
  int num_elements;
  pim_role_t role;

  int values_per_col;
  int values_per_row;

  /* Offset into the global bit-interleaved element space, see map_element. */
  uint64_t layout_linear_base;
  /* Compiler's partition bit: 1 replicated, 0 bank-partitioned, -1 not stated. Only 0
   * changes anything: that tensor's accesses go to the executing lane's own bank at a
   * per-lane slab address instead of the element's interleaved home, see
   * place_in_lane_slab. The lockstep key drops bank, so counts do not move; addresses do. */
  int bank_replicated;
  int lane_row_base;        /* first row of this tensor's per-lane slab, or -1 */
  int lane_row_count;
  int lane_ord[MAX_BANKS];  /* next free value slot per lane */
  slot_map_t *lane_slot;    /* (lane, elem) -> slot, partitioned tensors only */
  long pack_base_col;       /* row-pack: first column of this tensor's share, or -1 */
  int pack_cols;

  /* Per-tensor diagnostic counters (Stage 0). Emitted = trace lines actually
   * written; dedup_skips = times check_and_mark returned 0 for this tensor;
   * range_calls = number of __mem_trace_load/store invocations into this
   * tensor (vs. expanded element accesses). */
  uint64_t emitted_br;
  uint64_t emitted_bw;
  uint64_t emitted_r;
  uint64_t emitted_w;
  uint64_t dedup_skips;
  uint64_t range_calls;
  /* DCC tile MAC: column reads one command covers (0 = one per column), and the count of
   * surviving reads in the current dispatch. */
  int tile_mac_reads;
  uint64_t tile_ord, tile_disp;
  /* __pim_dcc_acc: -1 when the compiler stated nothing for this tensor. */
  int acc_reset, acc_wb;
  /* __pim_dcc_mac_addr: head and reduce strides and the reduce extent, 0 when unstated;
   * addr_last is the previous kept tile's order key within addr_disp. */
  int64_t addr_head, addr_k, addr_kext, addr_last;
  uint64_t addr_disp;
  /* __pim_dcc_return_from: the input this tensor's return stage is sized from, -1 none,
   * and under RET the columns per bank that return stage spans. */
  int ret_src, ret_cols;
} tensor_info_t;

static tensor_info_t tensors[MAX_TENSORS];
static int num_tensors = 0;

/* DCC-parity tracer (branch dcc-parity). Group-level commands span the CHANNEL, DCC's
 * PIM group, where the OptiPIM-matched tree scopes them to one pseudochannel. Their MAC
 * count reconciles only at that scope (cmds x 16 lanes x 32 cores = M*K). */

/* Bank named, so replicas do NOT fold, but (row,col) still key the entry so values in
 * one column still collapse to one command. */
static inline uint64_t perbank_ns(const tensor_info_t *t, int ch, int pch, int bg,
                                  int bank, int is_write) {
  /* MUST FIT IN 16 BITS: addr_dedup's pack_keys gives key1 exactly 16 and truncates
   * silently. */
  uint64_t ns = ((uint64_t)(t - tensors) & 0xFULL)        /* bits 0-3   16 tensors  */
                | ((uint64_t)(is_write ? 1 : 0) << 4)     /* bit  4                 */
                | (((uint64_t)ch & 0x3ULL) << 5)          /* bits 5-6    4 channels */
                | (((uint64_t)pch & 0x3ULL) << 7)         /* bits 7-8    4 pch      */
                | (((uint64_t)bg & 0x7ULL) << 9)          /* bits 9-11   8 bg       */
                | (((uint64_t)bank & 0xFULL) << 12);      /* bits 12-15 16 banks/bg */
  if (ns > 0xFFFFULL) {
    static int warned = 0;
    if (!warned++)
      fprintf(stderr, "[pim-runtime] ERROR: per-bank dedup namespace %llu exceeds the "
                      "16 bits addr_dedup gives it; banks are aliasing and the trace "
                      "under-counts. Widen pack_keys or shrink a field.\n",
              (unsigned long long)ns);
  }
  return ns;
}

/* Bank and bank group stay out, so one command covers the channel. The write bit keeps
 * an accumulator load from marking the key its store then consults. */
static inline uint64_t channel_ns(const tensor_info_t *t, int ch, int is_write) {
  return ((uint64_t)(t - tensors) & 0xFULL) | (((uint64_t)ch & 0x1FULL) << 4) |
         ((uint64_t)(is_write ? 1 : 0) << 11);
}

/* ================================================================
 *  Runtime state
 * ================================================================ */
static FILE *trace_fp = NULL;

static pim_phase_t cur_phase = PIM_PHASE_IDLE;
static int initialized = 0;
static int finalized = 0;

/* Statistics */
static uint64_t stat_bank_reads = 0;
static uint64_t stat_bank_writes = 0;
static uint64_t stat_reads = 0;
static uint64_t stat_writes = 0;
static uint64_t stat_ignored = 0;
/* Loads folded by the within-call DQ-word coalescer: lanes of ONE vector load that
 * land on the same DRAM column. */
static uint64_t stat_load_coalesced = 0;
/* Distinct keys past the 64-entry per-call buffer, i.e. coalescing this call could
 * not complete. Must stay 0; nonzero means a kernel emits a vector wider than the
 * buffer and some duplicate lanes were charged twice. */
static uint64_t stat_coalesce_overflow = 0;

/* Lockstep collapse, an emulation correction and not reuse. The host replays the kernel
 * once per bank, so one SIMD instruction arrives as one event per bank. Events with the
 * same lockstep_key (namespace, dispatch, lane access ordinal, row, col) fold into one
 * command. The namespace is channel_ns for group-level commands and perbank_ns for an
 * accumulator store, which DCC charges per bank. Operand loads bypass it. */
static addr_dedup_state_t *g_lockstep_collapse = NULL;
static uint64_t stat_lockstep_skips = 0;
/* Accumulator ops DCC's model charges nothing for, so this tree does not emit
 * them. Counted so the omission is visible rather than silent. */
static uint64_t stat_dcc_free_acc_ops = 0;

/* ACTIVATION STAGING, DCC's convention, this tree only.
 *
 * Their model stages activation vectors into the banks per invocation and leaves the
 * weight matrix resident: GEMV emits ST for the vector and nothing for the matrix, VA
 * emits LD for both its inputs because both are activations. We already match them on
 * GEMV, because our operand path stages the vector -- which is why parity is 0.973x --
 * but an elementwise kernel's inputs look partitioned to the residency classifier, get
 * read as bank-local group commands, and are never staged. That is the whole reason our
 * unfused VA looked 1.5-2.1x faster than theirs: 34 costed commands against 128.
 *
 * Which tensors are activations is the COMPILER's call, not the host's: it is
 * reduction_tiled = false in the im.residency stamp. The host only forwards it, so no
 * new decision is invented here and no artifact format changes.
 *
 * Staging is emitted once per distinct (tensor, bank, row, col), into the stage phase,
 * on first read -- which is banks x columns-per-bank, exactly the vector. */
static unsigned char g_dcc_activation[MAX_TENSORS];

/* GRF-SIDE OPERAND, DCC's convention, this tree only.
 *
 * HBM-PIM's MAC takes one operand from a DRAM column and the other from the register
 * file, so exactly one input needs a bank->GRF load. The residency classifier decides
 * only REPLICATION, and a GEMV vector partitioned across a parallel axis comes back
 * with the same verdict as the matrix (ReductionStridedMatrix, partitioned), so neither
 * was staged and the GRF load went uncharged entirely -- the whole 1.28x on DCC's own
 * gemv_single shapes.
 *
 * Discriminator is RANK, already in the artifact: the operand is broadcast along a
 * parallel axis the streamed tensor carries, so it has fewer footprint axes. Equal
 * ranks disqualify everything, which is every elementwise kernel, so the VA/RELU parity
 * this tree exists for cannot move.
 *
 * Deduped per (bank, row, col), so a value loaded once stays resident. DCC instead
 * reloads per output GRF block because n_grf is a fixed 8 in their generators; that
 * difference is the residency LEVER and is deliberately left visible rather than
 * emulated here. */
static unsigned char g_dcc_grf_operand[MAX_TENSORS];
static addr_dedup_state_t *g_dcc_grf_staged = NULL;
static int g_dcc_grf_decided = 0;

/* GRF_A, the operand register file: 8 entries of one column each (Aquabolt-XL GRF_A,
 * and DCC's n_grf). Every operand load the kernel emits is charged. Reuse across tiles is
 * the compiler's to realize (im-operand-hoist moves the loads above the tile loop), so
 * this LRU only counts how many loads a resident file would have absorbed. */
#define GRF_A_ENTRIES 8
typedef struct { uint64_t key[GRF_A_ENTRIES]; uint64_t dispatch; uint8_t n; } grf_a_t;
static grf_a_t g_grf_a[MAX_BANKS];
static uint64_t stat_grf_a_loads = 0;    /* first touch of a column in a dispatch */
static uint64_t stat_grf_a_refetch = 0;  /* loaded again into the same bank */
static uint64_t stat_grf_a_lru_hits = 0; /* reloads an 8-entry LRU would absorb, per dispatch */

/* GRF_B, the accumulator register file: 8 entries. The compiler states the widest
 * loop-carried accumulator per lane (__pim_acc_cells_per_lane). Past capacity the kernel
 * as written cannot run here: no opcode loads a partial sum back into GRF_B, so the only
 * realization is a blocked kernel, and that is the compiler's to write. FLAGGED, NOT
 * CHARGED. An invented spill cost would credit a transformation nobody made. */
#define GRF_B_ENTRIES 8
static int g_grf_b_overflow = 0;   /* cells beyond capacity, 0 = fits */

/* Lazy: registration is one tensor at a time and this needs to compare them all.
 *
 * Rank would be the principled discriminator -- the operand is broadcast along a
 * parallel axis the streamed tensor carries -- but PIM_LW_NUM_AXES reads 0 on these
 * kernels because the residency pass does not populate `footprint`, so the comparison
 * is vacuous. Size stands in, and it is DCC's own stated rule rather than an invention:
 * they stage the activation vector per invocation and leave the weight matrix resident.
 * STRICTLY smaller, so an elementwise kernel whose inputs match in size selects nothing
 * and the VA/RELU parity this tree exists for cannot move. */
static void dcc_pick_grf_operand(void) {
  if (__pim_dcc_decided) {  /* the compiler elected, possibly nothing */
    for (int i = 0; i < (int)__pim_dcc_grf_a_count && i < PIM_MAX_DCC_ACC; i++) {
      int a = __pim_dcc_grf_a[i];
      if (a < 0 || a >= num_tensors || tensors[a].role == PIM_ROLE_ACCUMULATOR) {
        fprintf(stderr, "[pim-runtime] ERROR: compiler elected GRF_A operand %d\n", a);
        exit(1);
      }
      g_dcc_grf_operand[a] = 1;
    }
    return;
  }
  int best = -1;
  for (int i = 0; i < num_tensors; i++) {
    if (tensors[i].role == PIM_ROLE_ACCUMULATOR) continue;
    if (tensors[i].bank_replicated != 0) continue;  /* replicated takes the W path */
    if (best < 0 || tensors[i].num_elements < tensors[best].num_elements) best = i;
  }
  if (best < 0) return;
  /* The larger tensor must be PARTITIONED too. Comparing against a replicated one elects
   * the streamed matrix whenever a replicated operand is bigger: measured on matmul
   * 64x256x32 with bank_axis=0 (A partitioned 2,048, B replicated 8,192) as PIM_MAC_OP1
   * 0 -- the entire compute stream replaced by operand loads. A replicated tensor is
   * already the GRF side by way of the W path and is not a candidate on either side. */
  for (int i = 0; i < num_tensors; i++)
    if (i != best && tensors[i].role != PIM_ROLE_ACCUMULATOR &&
        tensors[i].bank_replicated == 0 &&
        tensors[i].num_elements > tensors[best].num_elements) {
      /* Which operand GRF_A holds is a residency decision, so it is the compiler's. */
      fprintf(stderr, "[pim-runtime] ERROR: tensor %d would be elected GRF_A by size; compile "
                      "with im_dcc_acc_grf so the compiler makes that call\n", best);
      exit(1);
    }
}
static addr_dedup_state_t *g_dcc_staged = NULL;
static uint64_t stat_dcc_staged = 0;

void pim_dcc_mark_activation(int tensor_id) {
  if (tensor_id >= 0 && tensor_id < MAX_TENSORS)
    g_dcc_activation[tensor_id] = 1;
}

/* The dispatch this runtime is currently seeing, taken from __pim_program_epoch.
 *
 * It is part of the lockstep KEY rather than a signal to reset the table. Those are
 * equivalent while the host drives one tile per program instance, because the bank
 * loop sits inside the instance loop, so an instance's keys are all adjacent. They
 * stop being equivalent for a persistent kernel, where the bank loop is OUTSIDE and
 * one replay walks every tile: bank 0 finishes tile T before bank 1 starts tile 0, so
 * resetting on a tile boundary would put the 32 replays of one tile in different
 * epochs and the collapse would never fire at all. Keying works for both. */
static uint64_t cur_dispatch = 0;

/* Bits of the dispatch field given to the tile index, with the program epoch
 * above it. Packed rather than summed: summing let instance p's tile 1 collide
 * with instance p+1's tile 0, so two distinct dispatches folded into one and the
 * trace under-emitted. Both components are policed at pack time. */
#define PIM_TILE_BITS 10
static int g_warned_dispatch_overflow = 0;

/* ================================================================
 *  Helpers
 * ================================================================ */

/* ---------------------------------------------------------------------------
 * DCC TRACE FORMAT (PIM_TRACE_FORMAT=dcc, this tree only).
 *
 * Emits DCC's own opcodes at DCC's own addresses so the trace runs through THEIR
 * Ramulator build. Pricing is then theirs by construction, with no mapping layer left
 * to get wrong -- which matters, because every error in this campaign bar one was a
 * vocabulary-mapping error.
 *
 * Every mapping below is pinned by counting, not by judgement:
 *   BR (group-collapsed column read)  -> PIM_MAC_OP1   one 16-lane SIMD op across 32
 *        cores; cmds x 16 x 32 = M*K exactly on 9/10 matvec10 shapes.
 *   W  (operand into a bank)          -> ST + PIM_LD_OP1   staged, then bank->GRF;
 *        their PIM_LD_OP1 count equals our W count exactly.
 *   BW (accumulator column)           -> PIM_WB_ACC + ST + PIM_ACC_RESET, 128 each on
 *        matvec 512x64, which is their whole output path.
 *   R  (accumulator read)             -> LD
 *
 * Addresses follow THEIR geometry, not ours: a 32-byte column, 32 columns to a row,
 * and the bank/bankgroup strides their generator derives in HBM_GS. Our (sa,row) is
 * flattened and wrapped into their 16384-row bank, which is stated below because it is
 * the one place the two machines are not the same shape.
 * --------------------------------------------------------------------------- */

/* Their HBM_GS, in bytes. Mirrors align_geometry() in run_dcc_matvec_comparison.py. */
#define DCC_COL_BYTES   32
#define DCC_COLS_PER_ROW 32
#define DCC_ROWS_PER_BANK 16384

static void decompose_global_bank(int g, int *ch, int *pch, int *bg, int *bank);

static uint64_t dcc_addr(int ch, int pch, int bg, int bank, int sa, int row, int col) {
  static int wrapped = 0;
  uint64_t g_row  = (uint64_t)DCC_COLS_PER_ROW * DCC_COL_BYTES;
  uint64_t g_ba   = (uint64_t)DCC_ROWS_PER_BANK * g_row;
  uint64_t g_bg   = (uint64_t)cfg_num_banks * g_ba;
  uint64_t g_rank = (uint64_t)cfg_num_bg * g_bg;
  uint64_t g_pch  = g_rank;                   /* their rank level is dropped to match 32 banks */
  uint64_t g_ch   = (uint64_t)cfg_num_pch * g_pch;
  uint64_t lrow = (uint64_t)sa * (uint64_t)cfg_num_rows + (uint64_t)row;
  if (lrow >= DCC_ROWS_PER_BANK) {
    if (!wrapped++)
      fprintf(stderr, "[pim-runtime] NOTE linear row %llu exceeds DCC's %d rows per "
                      "bank and wraps; relative placement is preserved, absolute is "
                      "not.\n", (unsigned long long)lrow, DCC_ROWS_PER_BANK);
    lrow %= DCC_ROWS_PER_BANK;
  }
  return (uint64_t)ch * g_ch + (uint64_t)pch * g_pch + (uint64_t)bg * g_bg
       + (uint64_t)bank * g_ba + lrow * g_row
       + (uint64_t)(col % DCC_COLS_PER_ROW) * DCC_COL_BYTES;
}

/* PHASE ORDERING. DCC's trace is grouped by phase -- stage the vector, then per round
 * reset / load operand / MAC / write back, then write the outputs -- and their controller
 * depends on it: MAC_OP1 requires every bank's row open, and our kernel-order interleaving
 * starves that prerequisite until the refresh manager aborts ("Failed to send refresh").
 * Each opcode class simulates cleanly alone, so it is the interleaving and not any one
 * command that their model rejects.
 *
 * Commands are therefore bucketed at emission and flushed in their order: staging, then
 * one round per GRF block of (reset, operand load, MAC, write-back) with no barrier
 * between rounds, then every output store. That is their generator's structure, and
 * within a phase their nesting too, position outer and bank inner. Any levers-on number
 * must be reported against this same baseline or the delta is not attributable. */
enum { DP_STAGE = 0, DP_RESET, DP_LDOP, DP_MAC, DP_WB, DP_OUT, DP_N };
static char  *dp_buf[DP_N];
static size_t dp_len[DP_N], dp_cap[DP_N];

static void dp_push(int phase, const char *op, uint64_t addr) {
  char line[64];
  int n = snprintf(line, sizeof line, "%s 0x%08llx\n", op, (unsigned long long)addr);
  if (dp_len[phase] + n + 1 > dp_cap[phase]) {
    size_t cap = dp_cap[phase] ? dp_cap[phase] * 2 : (1u << 16);
    while (cap < dp_len[phase] + n + 1) cap *= 2;
    char *nb = (char *)realloc(dp_buf[phase], cap);
    if (!nb) { fprintf(stderr, "[pim-runtime] ERROR: phase buffer alloc failed\n"); exit(1); }
    dp_buf[phase] = nb; dp_cap[phase] = cap;
  }
  memcpy(dp_buf[phase] + dp_len[phase], line, n);
  dp_len[phase] += n;
}

/* GRF block rounds. A block ends at its store, which only the ACCESSES show: the lockstep
 * collapse drops lanes 1..N's MACs, so counting emitted commands would leave every later
 * lane in round 0. A map's rounds are the strips its program stores in. */
static int g_rnd_lane = -1;
static uint64_t g_rnd_disp = UINT64_MAX;
static uint32_t g_rnd = 0;
static int g_rnd_closed = 0;

/* Per-lane access ordinal within a dispatch. Lanes replay one instruction stream, so
 * the k-th access of every lane is the same instruction. */
static int g_ord_lane = -1;
static uint64_t g_ord_disp = UINT64_MAX;
static uint64_t g_access_ord = 0;

static void access_ord_next(void) {
  int lane = __pim_get_bank_id();
  if (lane != g_ord_lane || cur_dispatch != g_ord_disp) {
    g_ord_lane = lane; g_ord_disp = cur_dispatch; g_access_ord = 0;
  }
  g_access_ord++;
}

static inline uint64_t lk_mix(uint64_t x) {
  x ^= x >> 33; x *= 0xFF51AFD7ED558CCDULL;
  x ^= x >> 33; x *= 0xC4CEB9FE1A85EC53ULL;
  return x ^ (x >> 33);
}

static uint64_t lockstep_key(uint64_t ns, uint64_t disp, uint64_t ord, uint64_t row,
                             uint64_t col) {
  uint64_t h = lk_mix(ns ^ 0x9E3779B97F4A7C15ULL);
  h = lk_mix(h ^ disp);
  h = lk_mix(h ^ ord);
  return lk_mix(h ^ ((row << 16) | col));
}

/* A reset zeroes what is live in GRF_B, so it belongs before the first access of the
 * accumulation its write-back drains, which is dispatches earlier when a tile boundary falls
 * inside one accumulator's life. Per lane: opened by the first read after a write. */
static int g_open_lane = -1, g_open_valid = 0, g_open_closed = 0;
static uint64_t g_open_disp = 0;
static uint32_t g_open_rnd = 0;

static void dcc_round_note(int is_write) {
  int lane = __pim_get_bank_id();
  if (lane != g_rnd_lane || cur_dispatch != g_rnd_disp) {
    g_rnd_lane = lane; g_rnd_disp = cur_dispatch; g_rnd = 0; g_rnd_closed = 0;
  }
  if (lane != g_open_lane) { g_open_lane = lane; g_open_valid = 0; g_open_closed = 0; }
  if (is_write) g_rnd_closed = 1;
  else if (g_rnd_closed) { g_rnd++; g_rnd_closed = 0; }
  if (is_write) g_open_closed = 1;
  else if (!g_open_valid || g_open_closed) {
    g_open_valid = 1; g_open_closed = 0; g_open_disp = cur_dispatch; g_open_rnd = g_rnd;
  }
}

/* Replay coverage, marked as each access enters the tracer and before coalescing, GRF
 * residency or the collapse can drop it. It shows every input element was loaded and every
 * output stored, which correct:true misses when the reference equals the initial buffer. It
 * does not show that an access reached the trace. */
static uint8_t *g_cov[MAX_TENSORS][2];
static int g_cov_failures = -1;

static void cov_mark(int tid, const tensor_info_t *t, uint64_t base, uint64_t n, int w) {
  if (tid < 0 || tid >= MAX_TENSORS || t->num_elements <= 0) return;
  if (!g_cov[tid][w]) g_cov[tid][w] = (uint8_t *)calloc(((size_t)t->num_elements + 7) / 8, 1);
  if (!g_cov[tid][w]) { fprintf(stderr, "[pim-runtime] ERROR: coverage alloc\n"); exit(1); }
  uint64_t es = (uint64_t)(t->elem_size > 0 ? t->elem_size : 1);
  uint64_t e0 = (base - (uint64_t)(uintptr_t)t->base_addr) / es;
  for (uint64_t e = e0; e < e0 + n && e < (uint64_t)t->num_elements; e++)
    g_cov[tid][w][e >> 3] |= (uint8_t)(1u << (e & 7));
}

static int cov_count(const uint8_t *m, int n) {
  int c = 0;
  if (m) for (int e = 0; e < n; e++) c += (m[e >> 3] >> (e & 7)) & 1;
  return c;
}

static void cov_report(void) {
  g_cov_failures = 0;
  for (int i = 0; i < num_tensors && i < MAX_TENSORS; i++) {
    const tensor_info_t *t = &tensors[i];
    int n = t->num_elements;
    if (n <= 0) continue;
    int rd = cov_count(g_cov[i][0], n), wr = cov_count(g_cov[i][1], n);
    int acc = t->role == PIM_ROLE_ACCUMULATOR;
    int ok = acc ? wr == n : rd == n;
    fprintf(stderr, "[pim-runtime] coverage tensor %d %s: %d/%d read, %d/%d written%s\n",
            i, acc ? "accumulator" : "input", rd, n, wr, n, ok ? "" : "  NOT COVERED");
    g_cov_failures += !ok;
    free(g_cov[i][0]); free(g_cov[i][1]); g_cov[i][0] = g_cov[i][1] = NULL;
  }
  if (g_pack_overflow) {
    fprintf(stderr, "[pim-runtime] row-pack: %llu accesses beyond a lane's share, placed "
                    "on its last column  NOT COVERED\n", (unsigned long long)g_pack_overflow);
    g_cov_failures++;
  }
  if (g_emit_drops) {
    fprintf(stderr, "[pim-runtime] %llu compute accesses reached no record (outside every "
                    "tensor, out of range, or a store to an input)  NOT COVERED\n",
            (unsigned long long)g_emit_drops);
    g_cov_failures++;
  }
}

/* A run that never reached finalize must not leave its bits for the next one. */
static void cov_reset(void) {
  for (int i = 0; i < MAX_TENSORS; i++)
    for (int w = 0; w < 2; w++) { free(g_cov[i][w]); g_cov[i][w] = NULL; }
  g_cov_failures = -1;
}

/* -1 before any DCC finalize, else the number of tensors the run left uncovered. */
int pim_dcc_coverage_failures(void) { return g_cov_failures; }

/* Everything after staging is recorded and flushed sorted. Pushed in lane-replay order,
 * 32 consecutive stores land on one bank and serialise, where their generator spreads the
 * same stores over 32 banks. */
typedef struct { uint64_t dispatch, addr, seq; uint32_t round; int phase; const char *op; int tid; } dcc_rec_t;
static const tensor_info_t *t_emit = NULL;  /* the tensor the current emit belongs to */
static dcc_rec_t *g_recs = NULL;
static size_t g_rec_n = 0, g_rec_cap = 0;

static uint64_t g_last_mac = 0;       /* a fold step issues at the open row of the last MAC */
static int32_t g_cur_op = 0;          /* im-relu-opcode: 1 while a ReLU-only load issues */
static uint64_t stat_fold_cmds = 0;  /* fold commands emitted after the collapse */

static void dp_record(int phase, const char *op, uint64_t addr) {
  if (g_rec_n == g_rec_cap) {
    g_rec_cap = g_rec_cap ? g_rec_cap * 2 : 1u << 14;
    g_recs = (dcc_rec_t *)realloc(g_recs, g_rec_cap * sizeof *g_recs);
    if (!g_recs) { fprintf(stderr, "[pim-runtime] ERROR: command record alloc\n"); exit(1); }
  }
  g_recs[g_rec_n] = (dcc_rec_t){cur_dispatch, addr, g_rec_n, g_rnd, phase, op,
                                t_emit ? (int)(t_emit - tensors) : -1};
  g_rec_n++;
}

/* GRF-operand staging is recorded, not pushed as text, because its ORDER is DCC's and
 * not the replay's. Each lane replays the program in turn, so pushing at emission gives
 * eight consecutive commands to one bank, serialized; their generator nests block, then
 * column, then bank, so consecutive commands hit 32 different banks. Same commands, 4%
 * apart in cycles. Flushed sorted into the stage and operand-load positions. */
typedef struct { uint64_t dispatch, addr, round; int stage; uint32_t blk; } grf_rec_t;
static grf_rec_t *g_grf_recs = NULL;
static size_t g_grf_n = 0, g_grf_cap = 0;

static void grf_record(uint64_t addr, int stage) {
  if (g_grf_n == g_grf_cap) {
    g_grf_cap = g_grf_cap ? g_grf_cap * 2 : 4096;
    g_grf_recs = (grf_rec_t *)realloc(g_grf_recs, g_grf_cap * sizeof *g_grf_recs);
    if (!g_grf_recs) { fprintf(stderr, "[pim-runtime] ERROR: grf record alloc\n"); exit(1); }
  }
  g_grf_recs[g_grf_n].dispatch = cur_dispatch;
  g_grf_recs[g_grf_n].round = 0;
  g_grf_recs[g_grf_n].stage = stage;
  g_grf_recs[g_grf_n].blk = g_rnd;
  g_grf_recs[g_grf_n++].addr = addr;
}

/* Their nesting is block, column, bank. Within one dispatch a reloaded column (GRF_A
 * evicted it) is a later ROUND, not a neighbour of its first load, so the round sorts
 * ahead of the column. Rounds are numbered after a first sort brings equal addresses
 * together; on the DCC shapes nothing reloads within a dispatch and every round is 0. */
static int grf_rec_cmp(const void *a, const void *b) {
  const grf_rec_t *x = (const grf_rec_t *)a, *y = (const grf_rec_t *)b;
  uint64_t g_ba = (uint64_t)DCC_ROWS_PER_BANK * DCC_COLS_PER_ROW * DCC_COL_BYTES;
  if (x->dispatch != y->dispatch) return x->dispatch < y->dispatch ? -1 : 1;
  if (x->round != y->round) return x->round < y->round ? -1 : 1;
  uint64_t xo = x->addr % g_ba, yo = y->addr % g_ba;      /* column within the bank */
  if (xo != yo) return xo < yo ? -1 : 1;
  return x->addr < y->addr ? -1 : (x->addr > y->addr);    /* then bank */
}

static void grf_sort(void) {
  for (size_t i = 0; i < g_grf_n; i++) g_grf_recs[i].round = 0;
  qsort(g_grf_recs, g_grf_n, sizeof *g_grf_recs, grf_rec_cmp);   /* equal addrs adjacent */
  for (size_t i = 1; i < g_grf_n; i++)
    if (g_grf_recs[i].addr == g_grf_recs[i - 1].addr &&
        g_grf_recs[i].dispatch == g_grf_recs[i - 1].dispatch)
      g_grf_recs[i].round = g_grf_recs[i - 1].round + 1;
  qsort(g_grf_recs, g_grf_n, sizeof *g_grf_recs, grf_rec_cmp);
}

/* Host staging of an activation, written in DCC's host order at flush: per tensor, then
 * position, then bank (gen_trace_HBMPIM_VA.py:111-117). Issued in first-touch order, a
 * kernel that interleaves two inputs made the host staging thrash rows, a cost of the
 * replay order and not of anything the host or the compiler does. */
typedef struct { int tid; uint64_t pos; int gb; uint64_t addr; } stage_rec_t;
static stage_rec_t *g_stage = NULL;
static size_t g_stage_n = 0, g_stage_cap = 0;

static void stage_record(int tid, uint64_t pos, int gb, uint64_t addr) {
  if (g_stage_n == g_stage_cap) {
    g_stage_cap = g_stage_cap ? g_stage_cap * 2 : 1u << 12;
    g_stage = (stage_rec_t *)realloc(g_stage, g_stage_cap * sizeof *g_stage);
    if (!g_stage) { fprintf(stderr, "[pim-runtime] ERROR: staging record alloc\n"); exit(1); }
  }
  g_stage[g_stage_n++] = (stage_rec_t){tid, pos, gb, addr};
}

static int stage_cmp(const void *pa, const void *pb) {
  const stage_rec_t *a = pa, *b = pb;
  if (a->tid != b->tid) return a->tid < b->tid ? -1 : 1;
  if (a->pos != b->pos) return a->pos < b->pos ? -1 : 1;
  return (a->gb > b->gb) - (a->gb < b->gb);
}

/* Their GEMV stages its vector as a host read of every column, position then bank, then
 * the writes bank by bank, a barrier after each half, in rounds of 16 positions
 * (gen_trace_HBMPIM_GEMV.py:165-182). Adopted for the GRF operand so both sides pay it. */
static uint64_t stage_bank_bytes(void) {
  return (uint64_t)DCC_ROWS_PER_BANK * DCC_COLS_PER_ROW * DCC_COL_BYTES;
}
static int stage_pos_cmp(const void *pa, const void *pb) {
  uint64_t a = *(const uint64_t *)pa, b = *(const uint64_t *)pb, g = stage_bank_bytes();
  if (a % g != b % g) return a % g < b % g ? -1 : 1;
  return (a > b) - (a < b);
}
static int stage_bank_cmp(const void *pa, const void *pb) {
  uint64_t a = *(const uint64_t *)pa, b = *(const uint64_t *)pb, g = stage_bank_bytes();
  if (a / g != b / g) return a / g < b / g ? -1 : 1;
  return (a > b) - (a < b);
}

/* DRAM state and register state have different lifetimes, so they are emitted on
 * different conditions. The ST puts the value into the bank and it STAYS there, so it is
 * charged once per address for the whole kernel. The PIM_LD_OP1 pulls it bank -> GRF_A
 * and the register file is reloaded per dispatch, so it is charged on every miss.
 * Coupling them, which this did until 2026-09-17, re-staged x once per output block:
 * 768 surplus ST worth 1,512 cycles at 1x32x128x512, charged against us on the parity
 * rung and credited to us on the lever rung. DCC separates them the same way -- their
 * 256 LD + 256 ST is a one-time host broadcast, their 1,024 PIM_LD_OP1 is per block. */
/* Replicated operands (the W path) stage through the same convention, once per column. */
static uint64_t *g_op_stage = NULL;
static size_t g_op_stage_n = 0, g_op_stage_cap = 0;

static void op_stage_record(uint64_t addr) {
  if (g_op_stage_n == g_op_stage_cap) {
    g_op_stage_cap = g_op_stage_cap ? g_op_stage_cap * 2 : 1u << 12;
    g_op_stage = (uint64_t *)realloc(g_op_stage, g_op_stage_cap * sizeof *g_op_stage);
    if (!g_op_stage) { fprintf(stderr, "[pim-runtime] ERROR: staging alloc\n"); exit(1); }
  }
  g_op_stage[g_op_stage_n++] = addr;
}

static void grf_stage_flush(void) {
  uint64_t *s = (uint64_t *)malloc((g_grf_n + g_op_stage_n + 1) * sizeof *s);
  if (!s) { fprintf(stderr, "[pim-runtime] ERROR: staging alloc\n"); exit(1); }
  size_t n = 0;
  for (size_t i = 0; i < g_grf_n; i++)
    if (g_grf_recs[i].stage) s[n++] = g_grf_recs[i].addr;
  for (size_t i = 0; i < g_op_stage_n; i++) s[n++] = g_op_stage[i];
  qsort(s, n, sizeof *s, stage_pos_cmp);
  uint64_t g = stage_bank_bytes();
  for (size_t i = 0; i < n;) {
    size_t j = i;
    int npos = 0;
    uint64_t last = UINT64_MAX;
    for (; j < n; j++) {
      if (s[j] % g != last) {
        if (npos == 16) break;
        npos++;
        last = s[j] % g;
      }
    }
    for (size_t k = i; k < j; k++)
      fprintf(trace_fp, "LD 0x%08llx\n", (unsigned long long)s[k]);
    fprintf(trace_fp, "PIM_BARRIER 0x00000000\n");
    qsort(s + i, j - i, sizeof *s, stage_bank_cmp);
    for (size_t k = i; k < j; k++)
      fprintf(trace_fp, "ST 0x%08llx\n", (unsigned long long)s[k]);
    fprintf(trace_fp, "PIM_BARRIER 0x00000000\n");
    i = j;
  }
  free(s);
}

/* Rounds in (dispatch, round) order and, within one, reset, operand load, MAC, write-back.
 * Outputs come after every round, position outer and bank inner. A MAC is one command
 * per lockstep group and a GRF load arrives already in grf_sort's order, so both keep
 * the order they were recorded in. */
static int rec_cmp(const void *a, const void *b) {
  const dcc_rec_t *x = (const dcc_rec_t *)a, *y = (const dcc_rec_t *)b;
  uint64_t g_ba = (uint64_t)DCC_ROWS_PER_BANK * DCC_COLS_PER_ROW * DCC_COL_BYTES;
  int xout = x->phase == DP_OUT, yout = y->phase == DP_OUT;
  if (xout != yout) return xout - yout;
  if (!xout) {
    if (x->dispatch != y->dispatch) return x->dispatch < y->dispatch ? -1 : 1;
    if (x->round != y->round) return x->round < y->round ? -1 : 1;
    if (x->phase != y->phase) return x->phase < y->phase ? -1 : 1;
    if (x->phase == DP_MAC || x->phase == DP_LDOP)
      return x->seq < y->seq ? -1 : (x->seq > y->seq);
  }
  uint64_t xo = x->addr % g_ba, yo = y->addr % g_ba;
  if (xo != yo) return xo < yo ? -1 : 1;
  if (x->addr != y->addr) return x->addr < y->addr ? -1 : 1;
  return x->seq < y->seq ? -1 : (x->seq > y->seq);
}

/* DCC's machine, natively. Their generators address compute to the 16 banks at
 * bank-in-group 0 and 1 and run each program as an even-bank wave, then an odd-bank wave,
 * with their own accumulator addressing. PIM_DCC_NATIVE selects what is emitted that way:
 * FAN issues each collapsed compute command per addressed bank, ACC applies the accumulator
 * convention the compiler stated per tensor (__pim_dcc_acc), WAVE splits each program into
 * its two waves. ADDR issues each kept MAC at the address DCC's generator gives its tile,
 * from the geometry the compiler stated, their pos_mat stride included (a convention of
 * their trace). RET emits a tensor's return stage as wide as the input the compiler named,
 * as their RED sizes it from the input, walking away from the input in allocation order,
 * which mirrors their addresses, under row-pack placement only. A convention the runtime
 * cannot apply is counted rather than fatal, so an in-process harness survives and the
 * reading refuses the stream. NAT_ALL leaves out bit 8, the retired HOIST bit, so a mask
 * carrying it is refused. */
enum { NAT_FAN = 1, NAT_ACC = 2, NAT_WAVE = 4, NAT_TILE = 16, NAT_ADDR = 32, NAT_RET = 64,
       NAT_ALL = 119 };
static int g_dcc_native = 0;
static uint64_t stat_native_off_bank0 = 0, g_dcc_waves = 0, stat_tile_absorbed = 0;
static uint64_t stat_addr_order_mismatch = 0, stat_addr_unmapped = 0, stat_addr_mapped = 0;
static uint64_t stat_ret_refused = 0, stat_acc_unstated = 0;
static int g_emit_elem = -1;  /* logical element index of the access being emitted */

/* ADDR: their generator's address for the tile the kept read at element e opens. With A as
 * mat[head][k][out], their MAC for head-in-bank itr, k tile i and column group c sits at
 * itr*K*M + i*K + c*n_mac elements from the matrix base (gen_trace_HBMPIM_GEMV.py:139,
 * the k tile stepping by dhead where the row is seq long), here from our slab's base. */
static void map_element(const tensor_info_t *t, int elem_idx, int *ch, int *pch,
                        int *bg, int *bank, int *sa, int *row, int *col);

static uint64_t dcc_mac_addr(tensor_info_t *t, int e, int ch, int pch, int bg, int bank) {
  int64_t n = t->tile_mac_reads, K = t->addr_kext, M = t->addr_k;
  int64_t h = e / t->addr_head, k = (e % t->addr_head) / M, o = e % M;
  int64_t itr = h / (__pim_lanes > 0 ? __pim_lanes : 1);
  int64_t off = itr * K * M + (k / n) * K + (o / n) * n;
  /* A read off a tile corner has no tile of theirs. It keeps our address and is counted,
   * and a reading at their addresses refuses the stream. */
  int vpc = t->values_per_col > 0 ? t->values_per_col : 16;
  int vpr = t->values_per_row > 0 ? t->values_per_row : 512;
  if (e < 0 || k % n || o % n || t->elem_size != 2 || n != vpc || off % vpc ||
      (t->pack_base_col >= 0 && off / vpc >= t->pack_cols)) {
    stat_addr_unmapped++;
    return 0;
  }
  /* Their order is column group outer, k tile inner, through every block of a head, so the
   * key runs on across the epoch's dispatches. Another order would read their aliasing
   * through our schedule, so it is counted and the reading refuses it. */
  int64_t key = ((o / n) * (K / n) + k / n);
  uint64_t ep = (__pim_persistent ? cur_dispatch >> PIM_TILE_BITS : cur_dispatch) + 1;
  if (t->addr_disp == ep && key != t->addr_last + 1) stat_addr_order_mismatch++;
  t->addr_disp = ep;
  t->addr_last = key;
  stat_addr_mapped++;
  /* The matrix base is the slab's first column under row-pack, else where the tensor's
   * first element lands in this bank's row map. */
  long cpr = vpr / vpc, base = t->pack_base_col;
  if (base < 0) {
    int c0, p0, g0, b0, s0, r0, col0;
    map_element(t, 0, &c0, &p0, &g0, &b0, &s0, &r0, &col0);
    base = ((long)s0 * cfg_num_rows + r0) * cpr + col0;
  }
  long lc = base + off / vpc;
  int linear_row = (int)(lc / cpr);
  return dcc_addr(ch, pch, bg, bank, linear_row / cfg_num_rows, linear_row % cfg_num_rows,
                  (int)(lc % cpr));
}

/* TILE: a stamped tensor issues one MAC per tile of surviving column reads. The reads it
 * absorbs still resolve, place, stage and count for coverage, so no later address moves.
 * A dispatch that ends mid-tile is refused: a DCC MAC cannot span one. */
static int dcc_tile_issue(tensor_info_t *t) {
  if (!(g_dcc_native & NAT_TILE) || !t || t->tile_mac_reads <= 0) return 1;
  if (t->tile_disp != cur_dispatch + 1) {
    if (t->tile_disp && t->tile_ord % (uint64_t)t->tile_mac_reads) {
      fprintf(stderr, "[pim-runtime] ERROR: a dispatch ended mid-tile (%llu reads, tiles of "
                      "%d)\n", (unsigned long long)t->tile_ord, t->tile_mac_reads);
      exit(1);
    }
    t->tile_disp = cur_dispatch + 1;
    t->tile_ord = 0;
  }
  if (t->tile_ord++ % (uint64_t)t->tile_mac_reads == 0) return 1;
  stat_tile_absorbed++;
  return 0;
}

static int dcc_addressed(int gb) { return gb % 4 < 2; }
static uint64_t dcc_epoch(const dcc_rec_t *r) {
  return __pim_persistent ? r->dispatch >> PIM_TILE_BITS : r->dispatch;
}

static void nat_print(const dcc_rec_t *r, int wave, uint64_t g_ba) {
  int gb = (int)(r->addr / g_ba);
  if (r->phase == DP_MAC && (g_dcc_native & NAT_FAN)) {
    if (gb != 0 && wave <= 0) stat_native_off_bank0++;  /* once per record, not per wave */
    uint64_t off = r->addr % g_ba;
    for (int b = 0; b < 32; b++)
      if (dcc_addressed(b) && (wave < 0 || (b & 1) == wave))
        fprintf(trace_fp, "%s 0x%08llx\n", r->op,
                (unsigned long long)(off + (uint64_t)b * g_ba));
    return;
  }
  if ((g_dcc_native & NAT_ACC) && (r->phase == DP_RESET || r->phase == DP_WB)) {
    const tensor_info_t *t = r->tid >= 0 && r->tid < num_tensors ? &tensors[r->tid] : NULL;
    if (!t || t->acc_wb < 0) {  /* no convention stated: kept, counted, and refused */
      stat_acc_unstated++;
      if (wave >= 0 && (gb & 1) != wave) return;
      fprintf(trace_fp, "%s 0x%08llx\n", r->op, (unsigned long long)r->addr);
      return;
    }
    if (r->phase == DP_RESET && t->acc_reset == 0) return;
    if (r->phase == DP_WB && t->acc_wb == 2 && !dcc_addressed(gb)) return;
  }
  if (wave >= 0 && (gb & 1) != wave) return;
  fprintf(trace_fp, "%s 0x%08llx\n", r->op, (unsigned long long)r->addr);
}

/* Records [0, nc) are the compute block in rec_cmp order. */
static void dcc_native_emit(size_t nc) {
  uint64_t g_ba = (uint64_t)DCC_ROWS_PER_BANK * DCC_COLS_PER_ROW * DCC_COL_BYTES;
  if (g_dcc_native & NAT_TILE) {
    if (stat_fold_cmds) {
      fprintf(stderr, "[pim-runtime] ERROR: TILE with lane folds, which no DCC MAC covers\n");
      exit(1);
    }
    for (int k = 0; k < num_tensors; k++)
      if (tensors[k].tile_mac_reads > 0 &&
          tensors[k].tile_ord % (uint64_t)tensors[k].tile_mac_reads) {
        fprintf(stderr, "[pim-runtime] ERROR: tensor %d ended mid-tile\n", k);
        exit(1);
      }
  }
  if (cfg_num_channels != 1 || cfg_num_banks != 4 ||
      cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks != 32) {
    fprintf(stderr, "[pim-runtime] ERROR: PIM_DCC_NATIVE models DCC's one-channel, 32-bank "
                    "machine only\n");
    exit(1);
  }
  if (!(g_dcc_native & NAT_WAVE)) {
    for (size_t k = 0; k < nc; k++) nat_print(&g_recs[k], -1, g_ba);
    return;
  }
  for (size_t i = 0; i < nc;) {
    uint64_t ep = dcc_epoch(&g_recs[i]);
    size_t j = i;
    int compute = 0;
    for (; j < nc && dcc_epoch(&g_recs[j]) == ep; j++)
      if (g_recs[j].phase == DP_MAC) compute = 1;
    for (int w = 0; w < 2; w++)
      for (size_t k = i; k < j; k++) nat_print(&g_recs[k], w, g_ba);
    g_dcc_waves += compute;
    i = j;
  }
}

uint64_t pim_dcc_addr_order_mismatch(void) { return stat_addr_order_mismatch; }
uint64_t pim_dcc_addr_unmapped(void) { return stat_addr_unmapped; }
uint64_t pim_dcc_addr_mapped(void) { return stat_addr_mapped; }
uint64_t pim_dcc_ret_refused(void) { return stat_ret_refused; }
uint64_t pim_dcc_acc_unstated(void) { return stat_acc_unstated; }
uint64_t pim_dcc_tile_absorbed(void) { return stat_tile_absorbed; }
uint64_t pim_dcc_waves(void) { return g_dcc_waves; }
uint64_t pim_dcc_native_off_bank0(void) { return stat_native_off_bank0; }

/* RET: the positions past a tensor's own columns, position outer and bank inner over the
 * banks it stored to, each at the next column away from the input. Not coverage: no kernel
 * access touched these. */
static void dcc_return_stage(void) {
  uint64_t g_ba = (uint64_t)DCC_ROWS_PER_BANK * DCC_COLS_PER_ROW * DCC_COL_BYTES;
  for (int t = 0; t < num_tensors; t++) {
    tensor_info_t *T = &tensors[t];
    if (T->ret_src < 0 || T->ret_cols <= T->pack_cols) continue;
    int stored[MAX_BANKS] = {0};
    for (size_t k = 0; k < g_rec_n; k++)
      if (g_recs[k].phase == DP_OUT && g_recs[k].tid == t) {
        uint64_t gb = g_recs[k].addr / g_ba;
        if (gb < MAX_BANKS) stored[gb] = 1;
      }
    int vpc = T->values_per_col > 0 ? T->values_per_col : 16;
    int vpr = T->values_per_row > 0 ? T->values_per_row : 512;
    long cpr = vpr / vpc;
    for (int p = T->pack_cols; p < T->ret_cols; p++) {
      long lc = T->pack_base_col + T->pack_cols - 1 - p;
      int linear_row = (int)(lc / cpr);
      for (int gb = 0; gb < MAX_BANKS; gb++) {
        if (!stored[gb]) continue;
        int ch, pch, bg, bank;
        decompose_global_bank(gb, &ch, &pch, &bg, &bank);
        fprintf(trace_fp, "ST 0x%08llx\n", (unsigned long long)dcc_addr(
                    ch, pch, bg, bank, linear_row / cfg_num_rows, linear_row % cfg_num_rows,
                    (int)(lc % cpr)));
      }
    }
  }
}

/* A map's results sit in GRF_B until its stores, one entry per output column per lane, so a
 * round may store at most GRF_B_ENTRIES columns per bank. Flagged like grf_b_check and never
 * re-blocked, since the strip is the program's. */
static int map_rec_cmp(const void *a, const void *b) {
  const dcc_rec_t *x = *(const dcc_rec_t *const *)a, *y = *(const dcc_rec_t *const *)b;
  uint64_t g_ba = (uint64_t)DCC_ROWS_PER_BANK * DCC_COLS_PER_ROW * DCC_COL_BYTES;
  if (x->dispatch != y->dispatch) return x->dispatch < y->dispatch ? -1 : 1;
  if (x->round != y->round) return x->round < y->round ? -1 : 1;
  if (x->addr / g_ba != y->addr / g_ba) return x->addr / g_ba < y->addr / g_ba ? -1 : 1;
  return (x->addr > y->addr) - (x->addr < y->addr);
}

static void map_grf_b_check(void) {
  if (!(__pim_layout_count > 0 && __pim_acc_cells_per_lane == 0) || !g_rec_n) return;
  uint64_t g_ba = (uint64_t)DCC_ROWS_PER_BANK * DCC_COLS_PER_ROW * DCC_COL_BYTES;
  const dcc_rec_t **v = (const dcc_rec_t **)malloc(g_rec_n * sizeof *v);
  if (!v) { fprintf(stderr, "[pim-runtime] ERROR: map GRF_B check alloc\n"); exit(1); }
  size_t n = 0;
  for (size_t q = 0; q < g_rec_n; q++)
    if (g_recs[q].phase == DP_OUT) v[n++] = &g_recs[q];
  qsort(v, n, sizeof *v, map_rec_cmp);
  int most = 0, cur = 0;
  for (size_t q = 0; q < n; q++) {
    if (!q || v[q]->dispatch != v[q - 1]->dispatch || v[q]->round != v[q - 1]->round ||
        v[q]->addr / g_ba != v[q - 1]->addr / g_ba) cur = 1;
    else if (v[q]->addr != v[q - 1]->addr) cur++;
    if (cur > most) most = cur;
  }
  free(v);
  if (most > GRF_B_ENTRIES) {
    g_grf_b_overflow = most - GRF_B_ENTRIES;
    fprintf(stderr, "[pim-runtime] ERROR: a map stores %d result columns per bank in one "
                    "round, and GRF_B holds %d. Strip it in the kernel. Cycles from this run "
                    "are NOT physical.\n", most, GRF_B_ENTRIES);
  }
}

static void dp_flush(void) {
  if (!trace_fp) return;
  if (g_grf_n) grf_sort();
  /* Staging, once for the kernel, in their host order: activations per tensor with a
   * barrier after each (gen_trace_HBMPIM_VA.py:118-131), then the GRF operand. */
  if (g_stage_n) qsort(g_stage, g_stage_n, sizeof *g_stage, stage_cmp);
  for (size_t k = 0; k < g_stage_n; k++) {
    fprintf(trace_fp, "ST 0x%08llx\n", (unsigned long long)g_stage[k].addr);
    if (k + 1 == g_stage_n || g_stage[k + 1].tid != g_stage[k].tid)
      fprintf(trace_fp, "PIM_BARRIER 0x00000000\n");
  }
  if (dp_len[DP_STAGE]) {
    fwrite(dp_buf[DP_STAGE], 1, dp_len[DP_STAGE], trace_fp);
    fprintf(trace_fp, "PIM_BARRIER 0x00000000\n");
  }
  if (g_grf_n || g_op_stage_n) grf_stage_flush();
  /* GRF loads join the round they were issued in. Their seq sits above every emitted
   * command's, so grf_sort's order survives the sort below. */
  for (size_t i = 0; i < g_grf_n; i++) {
    if (g_rec_n == g_rec_cap) {
      g_rec_cap = g_rec_cap ? g_rec_cap * 2 : 1u << 14;
      g_recs = (dcc_rec_t *)realloc(g_recs, g_rec_cap * sizeof *g_recs);
      if (!g_recs) { fprintf(stderr, "[pim-runtime] ERROR: command record alloc\n"); exit(1); }
    }
    g_recs[g_rec_n++] = (dcc_rec_t){g_grf_recs[i].dispatch, g_grf_recs[i].addr,
                                    ((uint64_t)1 << 62) + i, g_grf_recs[i].blk, DP_LDOP,
                                    "PIM_LD_OP1", -1};
  }
  map_grf_b_check();
  if (g_rec_n) qsort(g_recs, g_rec_n, sizeof *g_recs, rec_cmp);
  size_t i = 0;
  while (i < g_rec_n && g_recs[i].phase != DP_OUT) i++;
  if (g_dcc_native)
    dcc_native_emit(i);
  else
    for (size_t k = 0; k < i; k++)
      fprintf(trace_fp, "%s 0x%08llx\n", g_recs[k].op, (unsigned long long)g_recs[k].addr);
  if (i) fprintf(trace_fp, "PIM_BARRIER 0x00000000\n");
  for (; i < g_rec_n; i++)
    fprintf(trace_fp, "%s 0x%08llx\n", g_recs[i].op, (unsigned long long)g_recs[i].addr);
  if (g_dcc_native & NAT_RET) dcc_return_stage();
  for (int p = 0; p < DP_N; p++) {
    free(dp_buf[p]); dp_buf[p] = NULL; dp_len[p] = dp_cap[p] = 0;
  }
  free(g_recs); g_recs = NULL; g_rec_n = g_rec_cap = 0;
  free(g_stage); g_stage = NULL; g_stage_n = g_stage_cap = 0;
  free(g_op_stage); g_op_stage = NULL; g_op_stage_n = g_op_stage_cap = 0;
  free(g_grf_recs); g_grf_recs = NULL; g_grf_n = g_grf_cap = 0;
  g_rnd_lane = -1; g_rnd_disp = UINT64_MAX; g_rnd = 0; g_rnd_closed = 0;
  g_open_lane = -1; g_open_valid = 0; g_open_closed = 0;
}

/* 1 = miss, stage and load; 0 = resident. Most recent at slot 0. dispatch+1 so a
 * zeroed struct never matches dispatch 0. */
static int grf_a_touch(int gb, uint64_t key) {
  grf_a_t *g = &g_grf_a[gb];
  if (g->dispatch != cur_dispatch + 1) { g->dispatch = cur_dispatch + 1; g->n = 0; }
  for (int i = 0; i < g->n; i++)
    if (g->key[i] == key) {
      for (int j = i; j > 0; j--) g->key[j] = g->key[j - 1];
      g->key[0] = key;
      return 0;
    }
  int n = g->n < GRF_A_ENTRIES ? g->n + 1 : GRF_A_ENTRIES;
  for (int j = n - 1; j > 0; j--) g->key[j] = g->key[j - 1];
  g->key[0] = key;
  g->n = (uint8_t)n;
  return 1;
}

static void grf_b_check(void) {
  /* Cells, so the cap is entries x values-per-entry and depends on the ACCUMULATOR's
   * dtype: 16 fp16 or 8 int32 per 256-bit entry. Reading it off the registered tensor
   * rather than cfg_data_width_bits keeps an int32 accumulator from being judged as fp16. */
  int esz = cfg_data_width_bits > 0 ? cfg_data_width_bits / 8 : 2;
  for (int i = 0; i < num_tensors; i++)
    if (tensors[i].role == PIM_ROLE_ACCUMULATOR && tensors[i].elem_size > 0) {
      esz = tensors[i].elem_size;
      break;
    }
  if (esz < 1) esz = 2;
  int cap = GRF_B_ENTRIES * ((cfg_dq_bits / 8) / esz);
  if ((int)__pim_acc_cells_per_lane > cap) {
    g_grf_b_overflow = (int)__pim_acc_cells_per_lane - cap;
    fprintf(stderr, "[pim-runtime] ERROR: accumulator tile is %d cells per lane; GRF_B "
                    "holds %d. Not realizable as written, block the output in the "
                    "kernel. Cycles from this run are NOT physical.\n",
            (int)__pim_acc_cells_per_lane, cap);
  }
}

/* Read back by the harness over ctypes, so a number can carry whether it is physical. */
int pim_dcc_grf_b_overflow(void) { return g_grf_b_overflow; }
uint64_t pim_dcc_grf_a_refetches(void) { return stat_grf_a_refetch; }

/* op is BR, W, BW or R, the role-level command, emitted in DCC's vocabulary. */
static void emit_trace(const char *op, int ch, int pch, int bg, int bank,
                       int sa, int row, int col) {
  uint64_t a = dcc_addr(ch, pch, bg, bank, sa, row, col);
  if (!strcmp(op, "BR")) {                       /* streamed column read + MAC */
    int tid = (int)(t_emit - tensors);
    if (tid >= 0 && tid < MAX_TENSORS && g_dcc_activation[tid]) {
      /* PER BANK, not per surviving command. The read itself is one group-level
       * command covering every bank, but staging is host-side data movement and
       * DCC charges it once per bank -- their LD is 64 for two M-vectors over 32
       * banks. Deduped on (tensor, row, col) so each column stages exactly once. */
      if (!g_dcc_staged)
        g_dcc_staged = addr_dedup_create(4096);
      if (addr_dedup_check_and_mark(g_dcc_staged, (uint64_t)tid,
                                    (uint64_t)sa * (uint64_t)cfg_num_rows + row,
                                    (uint64_t)col)) {
        int nb = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
        uint64_t pos = ((uint64_t)sa * (uint64_t)cfg_num_rows + (uint64_t)row) *
                           (uint64_t)DCC_COLS_PER_ROW + (uint64_t)col;
        for (int gb = 0; gb < nb; gb++) {
          int c2, p2, g2, b2;
          decompose_global_bank(gb, &c2, &p2, &g2, &b2);
          stage_record(tid, pos, gb, dcc_addr(c2, p2, g2, b2, sa, row, col));
          stat_dcc_staged++;
        }
      }
    }
    if (dcc_tile_issue((tensor_info_t *)t_emit)) {
      if ((g_dcc_native & NAT_ADDR) && t_emit->addr_head > 0) {
        uint64_t their = dcc_mac_addr((tensor_info_t *)t_emit, g_emit_elem, ch, pch, bg,
                                      bank);
        if (their) a = their;
      }
      dp_record(DP_MAC, g_cur_op == 1 ? "PIM_RELU" : "PIM_MAC_OP1", a);
    }
    g_last_mac = a;
  } else if (!strcmp(op, "W")) {                 /* operand staged once, then into the GRF */
    if (!g_dcc_grf_staged) g_dcc_grf_staged = addr_dedup_create(4096);
    if (addr_dedup_check_and_mark(g_dcc_grf_staged,
                                  perbank_ns(t_emit, ch, pch, bg, bank, 0),
                                  (uint64_t)sa * (uint64_t)cfg_num_rows + (uint64_t)row,
                                  (uint64_t)col))
      op_stage_record(a);
    dp_record(DP_LDOP, "PIM_LD_OP1", a);
  } else if (!strcmp(op, "BW")) {                /* their entire output path */
    /* A map has no accumulator to reset or drain, and DCC's own VA and RELU emit
     * neither command. Gate on layout_count too: the weak-extern fallback makes both
     * globals 0, so an artifact carrying no table must not read as elementwise. */
    /* DCC's result-register path writes a map's output back too (WB_RES), so under
     * the ACC bit a tensor the compiler gave a write-back keeps it. */
    if ((g_dcc_native & NAT_ACC) && __pim_dcc_decided && (!t_emit || t_emit->acc_wb < 0))
      stat_acc_unstated++;  /* a stored tensor the compiler stated nothing for */
    if (__pim_layout_count > 0 && __pim_acc_cells_per_lane == 0 &&
        !((g_dcc_native & NAT_ACC) && t_emit && t_emit->acc_wb > 0)) {
      stat_dcc_free_acc_ops += 2;
    } else {
      dp_record(DP_RESET, "PIM_ACC_RESET", a);
      if (g_open_valid && g_open_lane == __pim_get_bank_id() &&
          (g_open_disp != cur_dispatch || g_open_rnd != g_rnd)) {
        g_recs[g_rec_n - 1].dispatch = g_open_disp;
        g_recs[g_rec_n - 1].round = g_open_rnd;
      }
      dp_record(DP_WB, "PIM_WB_ACC", a);
    }
    dp_record(DP_OUT, "ST", a);
  } else if (!strcmp(op, "R")) {
    dp_push(DP_STAGE, "LD", a);
  }
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
  if (g_lockstep_collapse) {
    addr_dedup_destroy(g_lockstep_collapse);
    g_lockstep_collapse = NULL;
  }
}

static void advance_program_epoch_if_needed(void) {
  /* Just record which dispatch we are in; the id goes into the lockstep key. See
   * cur_dispatch for why keying beats resetting. The epoch itself is signalled by the
   * host through the program-id setters rather than inferred from the ids, so the
   * cadence does not depend on the order the replay walks the grid. */
  {
    uint64_t tile = __pim_tile_index();
    uint64_t epoch = __pim_program_epoch;
    uint64_t packed = __pim_persistent ? ((epoch << PIM_TILE_BITS) | tile) : epoch;
    if (!g_warned_dispatch_overflow && __pim_persistent &&
        tile >= (1ULL << PIM_TILE_BITS)) {
      g_warned_dispatch_overflow = 1;
      fprintf(stderr,
              "[pim-runtime] WARN: tile index %llu does not fit %d bits (epoch "
              "%llu). Ids alias and distinct dispatches collapse.\n",
              (unsigned long long)tile, PIM_TILE_BITS, (unsigned long long)epoch);
    }
    /* Only a persistent kernel spends bits on the tile index. A gridded one has
     * none, and shifting its epoch left by 10 overflowed the field at 128
     * instances. The artifact says which shape this is. */
    cur_dispatch = packed;
  }
}

static int compute_global_bank(int ch, int pch, int bg, int bank) {
  int banks_per_pch = cfg_num_bg * cfg_num_banks;
  int banks_per_ch = cfg_num_pch * banks_per_pch;
  return ch * banks_per_ch + pch * banks_per_pch + bg * cfg_num_banks + bank;
}

/* Inverse of compute_global_bank: flat replay bank id -> (ch,pch,bg,bank). */
static void decompose_global_bank(int g, int *ch, int *pch, int *bg, int *bank) {
  int nb = cfg_num_banks > 0 ? cfg_num_banks : 1;
  int nbg = cfg_num_bg > 0 ? cfg_num_bg : 1;
  int npch = cfg_num_pch > 0 ? cfg_num_pch : 1;
  if (g < 0) g = 0;
  *bank = g % nb;  g /= nb;
  *bg   = g % nbg; g /= nbg;
  *pch  = g % npch; g /= npch;
  *ch   = g;
}

/* Bit-interleaved (scheme8-like) element-to-physical mapping, LSB to MSB:
 *
 *   [ within-DQ | col | bank-within-BG | BG | pch | chan | linear_row ]
 *
 * Every field is log2 of a power-of-two geometry value. On this machine that is bits
 * 0-3 within-DQ (16), 4-8 col (32), 9-10 bank (4), 11-12 BG (4), 13 pch, then the row. */
static void map_element(const tensor_info_t *t, int elem_idx, int *ch, int *pch,
                        int *bg, int *bank, int *sa, int *row, int *col) {
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

/* Banks are threads here: lane b is bank b's PE and may only touch bank b. The
 * interleaved map changes bank every 512 elements while a lane's slab is 1024, so
 * lane 0's data sat in banks 0 and 1. Put the element in the executing lane's bank at
 * the next slot of that lane's slab (OptiPIM's alloc_col++ made explicit); the slot is
 * remembered so a re-read lands on the same address. A replicated tensor gets the same
 * treatment: its copy in bank b is a full slab, not the source word's (sa,row,col)
 * with the bank bits dropped, which aliased 4-8 words onto one address. */
static void place_in_lane_slab(tensor_info_t *t, pim_phys_loc_t *loc) {
  static int warned = 0;
  int all_banks = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
  int lane = __pim_get_bank_id();
  if (lane < 0 || lane >= all_banks || lane >= MAX_BANKS) {
    if (!warned++)
      fprintf(stderr, "[pim-runtime] WARN replay lane %d outside the %d banks; kept "
                      "the interleaved home.\n", lane, all_banks);
    return;
  }
  /* Every lane loads a replicated tensor in the same order, so one numbering serves
   * all 32 copies: key on lane 0 and the map stays the size of the tensor, not 32x. */
  /* >= 1, not == 1: the compiler word carries the RECEIVER COUNT now (0 partitioned,
   * 1 every bank, >1 exactly that many), so an equality test read a replicated tensor
   * as partitioned the moment the count stopped being a flag. */
  int key_lane = (t->bank_replicated >= 1) ? 0 : lane;
  int inserted = 0;
  int32_t slot = slot_map_get_or_put(t->lane_slot, key_lane, loc->elem_idx,
                                     t->lane_ord[key_lane], &inserted);
  if (slot < 0) {
    if (!warned++)
      fprintf(stderr, "[pim-runtime] WARN slab slot map allocation failed; kept the "
                      "interleaved home.\n");
    return;
  }
  if (inserted)
    t->lane_ord[key_lane]++;
  int vpc = t->values_per_col > 0 ? t->values_per_col : 16;
  int vpr = t->values_per_row > 0 ? t->values_per_row : 512;
  if (t->pack_base_col >= 0) {
    long c = slot / vpc, cpr = vpr / vpc;
    if (c >= t->pack_cols) {
      g_pack_overflow++;
      c = t->pack_cols - 1;
    }
    long lc = t->pack_base_col + c;
    int linear_row = (int)(lc / cpr);
    decompose_global_bank(lane, &loc->ch, &loc->pch, &loc->bg, &loc->bank);
    loc->sa = linear_row / cfg_num_rows;
    loc->row = linear_row % cfg_num_rows;
    loc->col = (int)(lc % cpr);
    stat_lane_placed++;
    return;
  }
  int r = slot / vpr;
  if (r >= t->lane_row_count) {
    if (!warned++)
      fprintf(stderr, "[pim-runtime] WARN tensor lane %d overflowed its %d-row slab; "
                      "rows now alias.\n", lane, t->lane_row_count);
    r = t->lane_row_count - 1;
  }
  int linear_row = t->lane_row_base + r;
  decompose_global_bank(lane, &loc->ch, &loc->pch, &loc->bg, &loc->bank);
  loc->sa = linear_row / cfg_num_rows;
  loc->row = linear_row % cfg_num_rows;
  loc->col = (slot % vpr) / vpc;
  stat_lane_placed++;
}

/* An access that reaches no record. Only COMPUTE gets here, so it is a silent drop and
 * coverage fails. */
static void note_ignored(void) {
  stat_ignored++;
  g_emit_drops++;
}

/* Resolve one logical element access to its physical HBM tuple.
 *
 * Returns 1 on success and fills `loc`. Returns 0 when the address falls outside the
 * tensor, already counted by note_ignored. */
static int resolve_access_location(tensor_info_t *t, uint64_t addr,
                                   pim_phys_loc_t *loc) {
  int elem_idx = (int)((addr - (uint64_t)t->base_addr) / t->elem_size);
  if (elem_idx < 0 || elem_idx >= t->num_elements) {
    note_ignored();
    return 0;
  }

  loc->elem_idx = elem_idx;
  map_element(t, elem_idx, &loc->ch, &loc->pch, &loc->bg, &loc->bank, &loc->sa,
              &loc->row, &loc->col);
  if (t->bank_replicated >= 0 && t->lane_slot && t->lane_row_base >= 0)
    place_in_lane_slab(t, loc);
  return 1;
}

/* Emit the trace op implied by the access direction and tensor role for an
 * already-resolved physical tuple. */
static void emit_access_by_role_phase(tensor_info_t *t,
                                      const pim_phys_loc_t *loc, int is_write) {
  t_emit = t;
  g_emit_elem = loc->elem_idx;
  if (!is_write) {
    switch (t->role) {
    case PIM_ROLE_STREAMED:
      emit_trace("BR", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa,
                 loc->row, loc->col);
      stat_bank_reads++;
      t->emitted_br++;
      break;
    case PIM_ROLE_OPERAND: {
      /* An operand load is a bus-mediated write into the RECEIVING PE's register
       * file, so address it at the replay bank rather than at the source
       * element's home bank. Otherwise all N fanout writes pile onto one bank and
       * serialize, which over-charges us relative to OptiPIM, whose codegen writes
       * to global_bank_id + i (fimdram.cpp:63-76) and parallelizes.
       *
       * There is deliberately NO broadcast here. docs/path1-bank-model-scope.md
       * forbade costing an all-bank op as ONE dispatch: it drops below OptiPIM's
       * own single_bank_opt floor, and OptiPIM has no all-bank command to match it
       * (its broadcast shortcut is commented out). */
      int dch, dpch, dbg, dbank;
      decompose_global_bank(__pim_get_bank_id(), &dch, &dpch, &dbg, &dbank);
      emit_trace("W", dch, dpch, dbg, dbank, loc->sa, loc->row, loc->col);
      stat_writes++;
      t->emitted_w++;
      /* The W already expands to ST + PIM_LD_OP1 in emit_trace, so no second command.
       * Our W count equals DCC's PIM_LD_OP1 count exactly. */
      break;
    }
    case PIM_ROLE_ACCUMULATOR:
      emit_trace("R", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa,
                 loc->row, loc->col);
      stat_reads++;
      t->emitted_r++;
      break;
    }
  } else {
    if (t->role == PIM_ROLE_ACCUMULATOR) {
      /* ACCUMULATOR here means "target of a tt.store", not "reduction target", so a
       * map lands in this branch too. Whether its reset and writeback are emitted is
       * decided in emit_trace's BW arm from the compiler's acc_cells_per_lane.
       *
       * This is confined to this tree. The default charges an accumulator writeback
       * as the DRAM write it is. */
      emit_trace("BW", loc->ch, loc->pch, loc->bg, loc->bank, loc->sa,
                 loc->row, loc->col);
      stat_bank_writes++;
      t->emitted_bw++;
    }
    /* Stores to STREAMED/OPERAND in COMPUTE phase are ignored (read-only) */
    else
      g_emit_drops++;
  }
}

/* ================================================================
 *  Public API
 * ================================================================ */

void pim_init(const char *trace_file) {
  const char *fmt = getenv("PIM_TRACE_FORMAT");
  if (!fmt || strcmp(fmt, "dcc")) {
    fprintf(stderr, "[pim-runtime] ERROR: this is the DCC-parity tracer and emits only "
                    "PIM_TRACE_FORMAT=dcc (got %s). The OptiPIM-matched tracer is "
                    "third_party/ramulator2.\n", fmt ? fmt : "unset");
    exit(1);
  }
  /* If already initialized, close the existing trace and reinitialize. */
  if (initialized && trace_fp) {
    fclose(trace_fp);
    trace_fp = NULL;
  }
  destroy_dedup_state();
  cov_reset();

  if (!trace_file)
    trace_file = "pim_trace.txt";

  trace_fp = fopen(trace_file, "w");
  if (!trace_fp) {
    fprintf(stderr, "[pim-runtime] ERROR: cannot open %s\n", trace_file);
    exit(1);
  }

  check_compiler_dq_bits(cfg_dq_bits);

  /* This tracer does not implement bank-group interleave, so refuse rather than place
   * the kernel differently from what the artifact states. */
  if (__pim_bg_interleave) {
    fprintf(stderr, "[pim-runtime] ERROR: the compiler asked for bank-group interleave, "
                    "which this tracer does not implement\n");
    exit(1);
  }

  cfg_place_align = __pim_placement_align == 3 ? PIM_ALIGN_ROW_PACK : PIM_ALIGN_DQ;
  fprintf(stderr, "[pim-runtime] placement: interleaved, align=%s (compiler said %d, "
                  "0 = nothing)\n",
          cfg_place_align == PIM_ALIGN_ROW_PACK ? "row-pack" : "dq",
          (int)__pim_placement_align);
  g_interleaved_next_linear = 0;
  g_lane_row_top = -1;
  g_pack_col_top = -1;
  g_pack_overflow = 0;
  g_emit_drops = 0;
  stat_lane_placed = 0;

  num_tensors = 0;
  g_dcc_grf_decided = 0;
  memset(g_dcc_grf_operand, 0, sizeof(g_dcc_grf_operand));
  memset(g_grf_a, 0, sizeof(g_grf_a));
  g_grf_b_overflow = 0;
  stat_grf_a_loads = stat_grf_a_refetch = stat_grf_a_lru_hits = 0;
  if (g_dcc_grf_staged) { addr_dedup_destroy(g_dcc_grf_staged); g_dcc_grf_staged = NULL; }
  cur_phase = PIM_PHASE_IDLE;
  stat_bank_reads = stat_bank_writes = stat_reads = stat_writes = stat_ignored =
      0;
  advance_program_epoch_if_needed();

  fprintf(stderr, "[pim-runtime] DCC-PARITY tracer: group-level commands span the\n                  CHANNEL. Not comparable to OptiPIM numbers.\n");
  fprintf(stderr, "[pim-runtime] emitting DCC's opcodes at DCC's addresses; run this "
                  "trace through THEIR Ramulator, not ours.\n");
  g_dcc_native = 0;
  stat_native_off_bank0 = g_dcc_waves = stat_tile_absorbed = stat_addr_order_mismatch = 0;
  stat_addr_unmapped = stat_addr_mapped = stat_ret_refused = 0;
  stat_acc_unstated = 0;
  {
    const char *nat = getenv("PIM_DCC_NATIVE");
    if (nat && *nat) {
      g_dcc_native = atoi(nat);
      if ((g_dcc_native & ~NAT_ALL) ||
          ((g_dcc_native & NAT_WAVE) && !(g_dcc_native & NAT_FAN)) ||
          ((g_dcc_native & NAT_ADDR) && !(g_dcc_native & NAT_TILE))) {
        fprintf(stderr, "[pim-runtime] ERROR: PIM_DCC_NATIVE=%s is not a valid mask for this "
                        "trace format\n", nat);
        exit(1);
      }
    }
    if ((g_dcc_native & NAT_ACC) && !__pim_dcc_decided) {
      fprintf(stderr, "[pim-runtime] ERROR: the ACC bit needs the compiler's accumulator "
                      "convention (im_dcc_acc_grf)\n");
      exit(1);
    }
    if (getenv("PIM_DCC_FANOUT")) {
      fprintf(stderr, "[pim-runtime] ERROR: PIM_DCC_FANOUT is retired, the compiler states "
                      "the accumulator convention\n");
      exit(1);
    }
    if (g_dcc_native)
      fprintf(stderr, "[pim-runtime] dcc native mask=%d, %d accumulator conventions from the "
                      "compiler\n", g_dcc_native, (int)__pim_dcc_acc_count);
  }

  if (__pim_lanes > 0 &&
      (int)__pim_lanes != cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks)
    fprintf(stderr, "[pim-runtime] WARN kernel compiled for %d lanes, machine has %d "
                    "banks; lane placement puts lane b in bank b and assumes they agree.\n",
            (int)__pim_lanes, cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks);
  stat_lockstep_skips = 0;
  stat_fold_cmds = 0;
  g_last_mac = 0;
  g_cur_op = 0;
  if (g_dcc_staged) { addr_dedup_destroy(g_dcc_staged); g_dcc_staged = NULL; }
  stat_dcc_staged = stat_dcc_free_acc_ops = 0;
  stat_coalesce_overflow = 0;

  /* 136 entries per bank is only the starting size. The table grows on demand, so it
   * bounds nothing. */
  g_lockstep_collapse =
      addr_dedup_create(cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks * 136);

  initialized = 1;
  finalized = 0;

  fprintf(stderr, "[pim-runtime] Initialized. Trace: %s\n", trace_file);
  fprintf(stderr,
          "[pim-runtime] HBM config: %d ch, %d pch, %d bg, %d banks/bg, "
          "%d sa, %d rows, %d cols, %d-bit DQ\n",
          cfg_num_channels, cfg_num_pch, cfg_num_bg, cfg_num_banks, cfg_num_sa,
          cfg_num_rows, cfg_num_cols, cfg_dq_bits);
}

/* ================================================================
 *  Compiler-emitted residency descriptor
 * ================================================================
 * emitPimLayoutTable (TritonIMToLLVM.cpp) puts these globals in the KERNEL object, and
 * kernel and runtime link into one dylib. The record layout is in pim_layout_table.h.
 * operand_arg == tensor id, because pointer args are registered first and in signature
 * order. Only bank_replicated and is_store are read, the footprint words are skipped. */

static int stat_layout_from_compiler = 0;

/* The compiler bakes a bus width into every vector width it picks. Ours must match,
 * or the artifact was tuned for a machine this run is not modelling. */
static void check_compiler_dq_bits(int cfg_bits) {
  if (__pim_dq_bits && __pim_dq_bits != cfg_bits)
    fprintf(stderr,
            "[%s] WARN: bus-width disagreement. The compiler chose vector widths for "
            "a %d-bit bus; this run models %d-bit. Vector accesses are priced against "
            "a width the artifact never assumed.\n",
            "pim-runtime", (int)__pim_dq_bits, cfg_bits);
}

/* Refuse a descriptor whose record width is not the one we stride by. The compiler's
 * kRecWords and this runtime's PIM_LAYOUT_REC_WORDS are in separate submodules, so a
 * change to one alone misreads every record past the first, silently and plausibly. */
static void check_layout_rec_words(void) {
  static int checked = 0;
  if (checked)
    return;
  checked = 1;
  int emitted = (int)__pim_layout_rec_words;
  if (emitted > 0 && emitted != PIM_LAYOUT_REC_WORDS) {
    fprintf(stderr,
            "[pim-runtime] FATAL: layout record width mismatch. The compiler emitted "
            "%d words per record, this runtime strides by %d. The Triton and "
            "ramulator2 submodules have drifted; every record past the first would be "
            "misread. Rebuild both.\n",
            emitted, PIM_LAYOUT_REC_WORDS);
    abort();
  }
}

static void apply_dcc_tile_mac(int tensor_id) {
  tensor_info_t *t = &tensors[tensor_id];
  t->tile_mac_reads = 0;
  t->tile_ord = t->tile_disp = 0;
  int n = (int)__pim_dcc_tile_mac_count;
  for (int i = 0; i < n && i < PIM_MAX_TILE_MAC; i++)
    if (__pim_dcc_tile_mac[2 * i] == tensor_id)
      t->tile_mac_reads = __pim_dcc_tile_mac[2 * i + 1];
  t->addr_head = t->addr_k = t->addr_kext = t->addr_last = 0;
  t->addr_disp = 0;
  n = (int)__pim_dcc_mac_addr_count;
  for (int i = 0; i < n && i < PIM_MAX_TILE_MAC; i++)
    if (__pim_dcc_mac_addr[4 * i] == tensor_id) {
      t->addr_head = __pim_dcc_mac_addr[4 * i + 1];
      t->addr_k = __pim_dcc_mac_addr[4 * i + 2];
      t->addr_kext = __pim_dcc_mac_addr[4 * i + 3];
    }
}

static void apply_dcc_acc(int tensor_id) {
  tensor_info_t *t = &tensors[tensor_id];
  t->acc_reset = t->acc_wb = -1;
  int n = __pim_dcc_decided ? (int)__pim_dcc_acc_count : 0;
  for (int i = 0; i < n && i < PIM_MAX_DCC_ACC; i++)
    if (__pim_dcc_acc[3 * i] == tensor_id) {
      t->acc_reset = __pim_dcc_acc[3 * i + 1];
      t->acc_wb = __pim_dcc_acc[3 * i + 2];
    }
}

int pim_dcc_acc_decided(void) { return __pim_dcc_decided; }

static void apply_dcc_return_from(int tensor_id) {
  tensor_info_t *t = &tensors[tensor_id];
  t->ret_src = -1;
  t->ret_cols = 0;
  int n = (int)__pim_dcc_return_from_count;
  for (int i = 0; i < n && i < PIM_MAX_TILE_MAC; i++)
    if (__pim_dcc_return_from[2 * i] == tensor_id) t->ret_src = __pim_dcc_return_from[2 * i + 1];
  if ((g_dcc_native & NAT_RET) && t->ret_src >= 0 && cfg_place_align != PIM_ALIGN_ROW_PACK) {
    stat_ret_refused++;  /* their layout packs the input beside the output, row-pack only */
    t->ret_src = -1;
  }
}

static void apply_compiler_layout(int tensor_id) {
  apply_dcc_tile_mac(tensor_id);
  apply_dcc_acc(tensor_id);
  apply_dcc_return_from(tensor_id);
  check_layout_rec_words();
  int n = (int)__pim_layout_count;
  if (n <= 0) {
    /* The tensor stays on the interleaved map, but a placement made by default rather
     * than by the compiler must still be audible. */
    static int warned = 0;
    if (!warned++)
      fprintf(stderr, "[pim-runtime] WARN kernel carries no layout table; every tensor "
                      "stays on the interleaved map and the host roles stand.\n");
    return;
  }
  const int32_t *table = __pim_layout_table;
  for (int i = 0; i < n; i++) {
    const int32_t *rec = table + (size_t)i * PIM_LAYOUT_REC_WORDS;
    if (rec[PIM_LW_OPERAND_ARG] != tensor_id)
      continue;
    if (stat_layout_from_compiler == 0)
      fprintf(
          stderr,
          "[pim-runtime] compiler-emitted layout table found in the kernel "
          "artifact (%d entries); host ctypes descriptor push is not used\n",
          n);
    /* COMPILER-DERIVED ROLE. Whether a tensor is replicated across banks
     * is a property of the LAYOUT, so the pass reads it off the encoding and we
     * honor it over the role the host asserted at registration. The host string
     * was written for one particular layout, so any layout change silently
     * inverted the charge: a transposed matmul accumulator measured 31x too fast
     * because both operands ended up in the free role (2026-09-06). The
     * accumulator is exempt -- a store's role is decided by what it is, not by
     * how it is spread. */
    tensor_info_t *_t = &tensors[tensor_id];
    /* Index through PIM_LW_*, never a literal: these shifted when the record lost
     * its reduction_col_axis word on 2026-09-10 and a raw rec[3] silently read
     * num_axes instead, turning every operand into a streamed tensor and dropping
     * the entire W charge (matmul 64x64x32 went 32,786 -> 246 cycles). */
    _t->bank_replicated = (int)rec[PIM_LW_BANK_REPLICATED];
    if (_t->bank_replicated >= 0) {
      /* Reserve a per-lane slab, the same rows in every bank. Partitioned: a lane can
       * touch far more than its even share when lanes overlap (a conv halo lane touched
       * 576 of a 4,096-element input), so a small tensor gets room for every element
       * per lane and only a large one falls back to twice the share. Replicated: every
       * lane holds the whole tensor, so the slab IS the tensor; the copy in bank b
       * used to keep the source word's (sa,row,col) and drop its bank bits, which
       * aliased 4-8 distinct words onto one address (matmul A: 8,192 writes, 2,048
       * addresses). When the rows do not fit, keep the interleaved home and say so. */
      int all_banks = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
      int vpr = _t->values_per_row > 0 ? _t->values_per_row : 512;
      int share2 = 2 * ((_t->num_elements + all_banks - 1) / all_banks);
      if (cfg_place_align == PIM_ALIGN_ROW_PACK) {
        /* The exact share, not the halo allowance below: packing needs the extent up
         * front, so a lane that touches more than its share is refused at placement. */
        int vpc = _t->values_per_col > 0 ? _t->values_per_col : 16;
        long cpr = vpr / vpc;
        int share = _t->bank_replicated >= 1 ? _t->num_elements
                    : (_t->num_elements + all_banks - 1) / all_banks;
        int cols = (share + vpc - 1) / vpc;
        if (g_pack_col_top < 0)
          g_pack_col_top = (long)cfg_num_sa * cfg_num_rows * cpr;
        if (g_pack_col_top - cols < 0) {
          fprintf(stderr, "[pim-runtime] ERROR tensor %d: row-pack share of %d columns does "
                          "not fit the %ld columns left per bank\n",
                  tensor_id, cols, g_pack_col_top);
          exit(1);
        }
        g_pack_col_top -= cols;
        _t->pack_base_col = g_pack_col_top;
        _t->pack_cols = cols;
        if ((g_dcc_native & NAT_RET) && _t->ret_src >= 0) {
          /* Their Ret_addr = Vec_addr + Vec_size: the input sits right beside the output,
           * and the return stage takes the input's width. Anything else emits no return
           * stage and is counted, and the padded reading refuses it. */
          tensor_info_t *src = _t->ret_src < num_tensors ? &tensors[_t->ret_src] : NULL;
          int extra = src ? src->pack_cols - _t->pack_cols : 0;
          if (!src || src->pack_base_col != _t->pack_base_col + _t->pack_cols ||
              g_pack_col_top - (extra > 0 ? extra : 0) < 0) {
            stat_ret_refused++;
          } else {
            _t->ret_cols = src->pack_cols;
            if (extra > 0) g_pack_col_top -= extra;
          }
        }
        _t->lane_row_base = (int)(g_pack_col_top / cpr);
        _t->lane_row_count = (int)((g_pack_col_top + cols + cpr - 1) / cpr) - _t->lane_row_base;
        g_lane_row_top = _t->lane_row_base;
        _t->lane_slot = slot_map_create((size_t)share / 2 + 1);
        if (!_t->lane_slot) {
          fprintf(stderr, "[pim-runtime] ERROR tensor %d: slot map allocation failed\n",
                  tensor_id);
          exit(1);
        }
      } else {
        int per_lane = _t->bank_replicated >= 1 ? _t->num_elements
                       : share2 > 65536 ? share2
                       : (_t->num_elements < 65536 ? _t->num_elements : 65536);
        int rows = (per_lane + vpr - 1) / vpr;
        if (rows < 1) rows = 1;
        if (g_lane_row_top < 0)
          g_lane_row_top = cfg_num_sa * cfg_num_rows;
        if (rows > g_lane_row_top) {
          fprintf(stderr, "[pim-runtime] WARN tensor %d: lane slab of %d rows does not fit "
                          "the %d rows left; placement stays on the interleaved map.\n",
                  tensor_id, rows, g_lane_row_top);
        } else {
          g_lane_row_top -= rows;
          _t->lane_row_base = g_lane_row_top;
          _t->lane_row_count = rows;
          _t->lane_slot = slot_map_create((size_t)per_lane / 2 + 1);
          if (!_t->lane_slot) {
            _t->lane_row_base = -1;
            fprintf(stderr, "[pim-runtime] WARN tensor %d: slot map allocation failed; "
                            "placement stays on the interleaved map.\n", tensor_id);
          }
        }
      }
    }
    /* THE ROLE IS THE COMPILER'S. is_store marks the output (sticky in the emitter, so a
     * tensor both loaded and stored is the output), bank_replicated says whether every
     * bank sees the same elements. The host string used to be the role and a layout
     * change silently inverted the charge (31x on a transposed accumulator); now it is
     * an input to check, and a disagreement fails invariant 6 instead of costing a day. */
    if (rec[PIM_LW_IS_STORE] == 1 || rec[PIM_LW_BANK_REPLICATED] >= 0) {
      int derived = rec[PIM_LW_IS_STORE] == 1 ? PIM_ROLE_ACCUMULATOR
                    : rec[PIM_LW_BANK_REPLICATED] ? PIM_ROLE_OPERAND : PIM_ROLE_STREAMED;
      if (derived != _t->role) {
        static const char *names[] = {"STREAMED", "OPERAND", "ACCUMULATOR"};
        fprintf(stderr, "[pim-runtime] WARN tensor %d: host role %s, compiler layout says "
                        "%s; taking the compiler's.\n", tensor_id,
                (_t->role >= 0 && _t->role < 3) ? names[_t->role] : "?",
                names[derived]);
        _t->role = derived;
      }
    }
    if (_t->bank_replicated < 0 && _t->role != PIM_ROLE_OPERAND)
      fprintf(stderr, "[pim-runtime] WARN tensor %d (%s): compiler stated no partition "
                      "bit; placement stays on the interleaved map.\n", tensor_id,
              _t->role == PIM_ROLE_ACCUMULATOR ? "ACCUMULATOR" : "STREAMED");
    stat_layout_from_compiler++;
    return;
  }
  /* A table exists but has no record for this tensor: same gap, same warning. */
  if (tensors[tensor_id].role != PIM_ROLE_OPERAND)
    fprintf(stderr, "[pim-runtime] WARN tensor %d: no layout record; placement stays "
                    "on the interleaved map.\n", tensor_id);
}

/* Assign the id, then apply the compiler's descriptor to it. */
static int pim_finish_register(void) {
  int tid = num_tensors++;
  apply_compiler_layout(tid);
  return tid;
}

int pim_register_tensor(void *ptr, const int *dims, int ndims, int elem_size,
                        pim_role_t role) {
  if (num_tensors >= MAX_TENSORS) {
    fprintf(stderr, "[pim-runtime] ERROR: too many tensors (max %d)\n",
            MAX_TENSORS);
    return -1;
  }

  /* im.noalias_args and im-load-cluster assume distinct arguments never alias. */
  size_t bytes = (size_t)elem_size;
  for (int i = 0; i < ndims && i < 4; i++) bytes *= (size_t)dims[i];
  for (int i = 0; i < num_tensors; i++) {
    const char *a = (const char *)tensors[i].base_addr, *b = (const char *)ptr;
    if (b < a + tensors[i].total_bytes && a < b + bytes) {
      fprintf(stderr, "[pim-runtime] ERROR: tensor %d overlaps tensor %d; kernel "
                      "arguments must not alias\n", num_tensors, i);
      return -1;
    }
  }

  tensor_info_t *t = &tensors[num_tensors];
  t->base_addr = ptr;
  t->elem_size = elem_size;
  t->role = role;
  t->bank_replicated = -1;
  t->lane_row_base = -1;
  t->lane_row_count = 0;
  t->pack_base_col = -1;
  t->pack_cols = 0;
  memset(t->lane_ord, 0, sizeof(t->lane_ord));
  if (t->lane_slot) { slot_map_destroy(t->lane_slot); t->lane_slot = NULL; }

  int total = 1;
  for (int i = 0; i < ndims && i < 4; i++)
    total *= dims[i];
  t->num_elements = total;
  t->total_bytes = (size_t)total * elem_size;

  /* Column packing follows the MODELLED data width, not the host array's dtype.
   * elem_size stays in use for address decoding (addr -> element index), which is
   * genuinely a property of how the host stores the array. */
  t->values_per_col = cfg_dq_bits / cfg_data_width_bits;
  t->values_per_row = cfg_num_cols * t->values_per_col;

  int total_flat_banks =
      cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
  int rows_capacity = cfg_num_sa * cfg_num_rows;
  uint64_t elems_per_global_row =
      (uint64_t)t->values_per_row * (uint64_t)total_flat_banks;

  /* A contiguous span of the global element space, aligned to the DQ word so it does
   * not share a packed column slot with the previous tensor. */
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
  return pim_finish_register();
}

void pim_set_phase(pim_phase_t phase) {
  if (phase != PIM_PHASE_IDLE && phase != PIM_PHASE_COMPUTE) {
    fprintf(stderr, "[pim-runtime] ERROR: phase %d is not IDLE or COMPUTE, the only "
                    "phases this tracer emits\n", (int)phase);
    exit(1);
  }
  fprintf(stderr, "[pim-runtime] Phase -> %s\n",
          phase == PIM_PHASE_COMPUTE ? "COMPUTE" : "IDLE");
  cur_phase = phase;
}

void pim_finalize(void) {
  if (finalized)
    return;

  finalized = 1;

  /* The check is lazy, on the first compute-phase load of a non-accumulator tensor. A
   * kernel with no such load would otherwise never be judged, so a run can never end
   * with the bound unexamined. */
  if (!g_dcc_grf_decided) {
    g_dcc_grf_decided = 1;
    dcc_pick_grf_operand();
    grf_b_check();
  }
  cov_report();
  dp_flush();

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
  fprintf(stderr,
          "[pim-runtime]   DCC activation staging commands : %" PRIu64 "\n"
          "[pim-runtime]   DCC zero-cost acc ops omitted   : %" PRIu64 "\n"
          "[pim-runtime]   Lockstep collapse skips         : %" PRIu64 "\n",
          stat_dcc_staged, stat_dcc_free_acc_ops, stat_lockstep_skips);
  fprintf(stderr,
          "[pim-runtime]   GRF_A operand loads / refetches : %" PRIu64 " / %" PRIu64 "\n"
          "[pim-runtime]   GRF_A reloads within one dispatch: %" PRIu64 " (charged)\n"
          "[pim-runtime]   GRF_B accumulator overflow      : %d cells beyond capacity\n",
          stat_grf_a_loads, stat_grf_a_refetch, stat_grf_a_lru_hits, g_grf_b_overflow);
  fprintf(stderr, "[pim-runtime]   Lane-placed accesses            : %" PRIu64 "\n",
          stat_lane_placed);
  /* Slab occupancy per lane: how many values each lane placed in its own bank. Uneven
   * rows are masked or straddling lanes, not a bug; a zero row for an active lane is. */
  for (int i = 0; i < num_tensors; i++) {
    tensor_info_t *t = &tensors[i];
    if (t->bank_replicated < 0 || t->lane_row_base < 0)
      continue;
    int all_banks = cfg_num_channels * cfg_num_pch * cfg_num_bg * cfg_num_banks;
    fprintf(stderr, "[pim-runtime]   tensor %d slab slots per lane:", i);
    for (int l = 0; l < all_banks && l < MAX_BANKS; l++)
      fprintf(stderr, " %d", t->lane_ord[l]);
    fprintf(stderr, "  (rows %d..%d)\n", t->lane_row_base,
            t->lane_row_base + t->lane_row_count - 1);
  }
  {
    uint64_t sat = addr_dedup_saturations(g_lockstep_collapse);
    if (sat)
      fprintf(stderr,
              "[pim-runtime]   WARN dedup table saturated       : %" PRIu64
              " (collapse stopped; trace over-emits)\n", sat);
  }
  fprintf(stderr,
          "[pim-runtime]   Load coalesce (intra-vector)    : %" PRIu64 "\n",
          stat_load_coalesced);
  if (stat_coalesce_overflow)
    fprintf(stderr,
            "[pim-runtime]   WARN coalesce buffer overflow    : %" PRIu64
            " (raise PIM_COALESCE_MAX)\n",
            stat_coalesce_overflow);

  /* Per-tensor breakdown — Stage-0 diagnostics. Useful for figuring out
   * which tensor's accesses dominate the trace and where dedup is biting. */
  fprintf(stderr,
          "\n[pim-runtime] === Per-tensor breakdown ===\n"
          "[pim-runtime]   tid  role         range_calls   BR        BW        "
          "R         W         dedup_skips\n");
  for (int i = 0; i < num_tensors; i++) {
    tensor_info_t *t = &tensors[i];
    const char *role_str = (t->role == PIM_ROLE_STREAMED)  ? "STREAMED"
                           : (t->role == PIM_ROLE_OPERAND) ? "OPERAND "
                                                           : "ACCUMUL.";
    fprintf(stderr,
            "[pim-runtime]   %3d  %-9s    %10" PRIu64 "  %8" PRIu64
            "  %8" PRIu64 "  %8" PRIu64 "  %8" PRIu64 "  %12" PRIu64 "\n",
            i, role_str, t->range_calls, t->emitted_br, t->emitted_bw,
            t->emitted_r, t->emitted_w, t->dedup_skips);
  }

  destroy_dedup_state();
  initialized = 0;
}

/* ================================================================
 *  LLVM pass hooks
 * ================================================================ */

/*
 * Translate a single memory access into PIM trace operations.
 *
 * COMPUTE phase (PIM PE active):
 *   STREAMED load   -> BR (bank-read: data streams through PE)
 *   OPERAND load    -> W  (write: data to PE register via bus)
 *   ACCUMULATOR load  -> R  (read: partial sum from PE)
 *   ACCUMULATOR store -> BW (bank-write: result to bank)
 *
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

  /* Lockstep dedup: collapse bank-replicated events at the same
   * (tensor_id, sa, row, col, access ordinal) within a program-id. Models 1 SIMD
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
  /* EVERY operand load bypasses the collapse and pays its per-bank write, broadcasts
   * included. Real HBM-PIM does have an all-bank GRF broadcast (Lee et al. ISCA'21
   * III-A/III-B), so this is a SYMMETRY choice, not a hardware limit: OptiPIM prices
   * input loading per PU inside its MILP objective, and taking the broadcast without
   * re-solving their optimizer would charge us for a dispatch the baseline never
   * gets. Both stacks now charge one write per receiving bank. */
  int is_operand_load = (!is_write && t->role == PIM_ROLE_OPERAND);
  int is_acc_store = (is_write && t->role == PIM_ROLE_ACCUMULATOR);
  /* GRF-SIDE OPERAND. A per-bank event decided by that bank's GRF_A, so
   * it must not enter the group-level collapse below: the collapse folds a later re-read
   * of the same column into the first as if it were a lane replica, which granted
   * residency upstream of any bound (K=256 loaded each column once where a bounded
   * GRF_A reloads per block). It also emitted a group-level MAC_OP1 for the read, which
   * DCC does not: their x read IS the LD_OP1. */
  if (!is_write && t->role != PIM_ROLE_ACCUMULATOR) {
    if (!g_dcc_grf_decided) {
      g_dcc_grf_decided = 1;
      dcc_pick_grf_operand();
      grf_b_check();
    }
    int tid = (int)(t - tensors);
    if (tid >= 0 && tid < MAX_TENSORS && g_dcc_grf_operand[tid]) {
      int lane = __pim_get_bank_id();
      if (lane < 0 || lane >= MAX_BANKS)
        lane = 0;
      uint64_t lrow = (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
      uint64_t key = ((uint64_t)tid << 40) | (lrow << 8) | (uint64_t)loc.col;
      if (!grf_a_touch(lane, key))
        stat_grf_a_lru_hits++;
      /* Keyed WITHOUT the dispatch: has this value ever been put in this bank. */
      if (!g_dcc_grf_staged) g_dcc_grf_staged = addr_dedup_create(4096);
      int first_stage = addr_dedup_check_and_mark(
          g_dcc_grf_staged, perbank_ns(t, loc.ch, loc.pch, loc.bg, loc.bank, 0),
          lrow, (uint64_t)loc.col);
      if (first_stage)
        stat_grf_a_loads++;
      else
        stat_grf_a_refetch++;
      grf_record(dcc_addr(loc.ch, loc.pch, loc.bg, loc.bank, loc.sa, loc.row, loc.col),
                 first_stage);
      /* Counted as W: an operand delivered into a bank's register file, the same ST +
       * PIM_LD_OP1 pair the W branch emits. */
      stat_writes++;
      t->emitted_w++;
      return;
    }
  }
  if (g_lockstep_collapse && !is_operand_load) {
    /* Compute and streamed reads are group-level and span the CHANNEL (DCC's group).
     * The accumulator is bank-level in DCC's model, so it keeps the bank in its key --
     * still packed by column, just not folded across banks. */
    uint64_t tensor_key =
        is_acc_store ? perbank_ns(t, loc.ch, loc.pch, loc.bg, loc.bank, is_write)
                     : channel_ns(t, loc.ch, is_write);
    /* Keyed on the lane's access ordinal too, so only the bank replicas of one
     * instruction fold and a lane re-reading its own column pays. The fields exceed
     * the packed key, so it is a 64-bit hash split back into k1/k2/k3. */
    uint64_t key_row = (uint64_t)loc.sa * (uint64_t)cfg_num_rows + (uint64_t)loc.row;
    uint64_t h = lockstep_key(tensor_key, cur_dispatch, g_access_ord, key_row,
                              (uint64_t)loc.col);
    if (!addr_dedup_check_and_mark(g_lockstep_collapse, h, h >> 16, h >> 48)) {
      stat_lockstep_skips++;
      t->dedup_skips++;
      return;
    }
  }

  emit_access_by_role_phase(t, &loc, is_write);
}

/* Cap on distinct DRAM columns one instrumented access can coalesce over. */
#define PIM_COALESCE_MAX 64

/* Process a vector or scalar access spanning [base_addr, base_addr+size).
 * MemTracePass calls this once per IR-level load/store; we expand to one
 * pim_trace_access_one per logical element so multi-bank/multi-row vector
 * loads are correctly accounted for, and the dedup primitive then collapses
 * intra-tile spatial redundancy. */
static void pim_trace_access_range(uint64_t base_addr, uint64_t size,
                                   int is_write, uint64_t compiler_lanes) {
  if (cur_phase == PIM_PHASE_IDLE) {
    stat_ignored++;
    return;
  }

  int tidx = find_tensor(base_addr);
  if (tidx < 0) {
    note_ignored();
    return;
  }
  tensor_info_t *t = &tensors[tidx];
  t->range_calls++;

  /* Track program-id boundaries so the dedup table is reset per tile. */
  advance_program_epoch_if_needed();
  access_ord_next();

  int elem_size = t->elem_size > 0 ? t->elem_size : 1;
  uint64_t n_elements = size / (uint64_t)elem_size;
  if (n_elements == 0)
    n_elements = 1;

  /* The compiler told us how many lanes this access moves. We derived the same
   * number from the HOST-registered element size. They are the same physical
   * quantity decided in two places, so say so when they differ instead of pricing
   * a width the artifact never emitted. Disagreement means the registered dtype
   * and the IR element type have parted company. */
  if (compiler_lanes && compiler_lanes != n_elements) {
    static int warned = 0;
    if (!warned) {
      warned = 1;
      fprintf(stderr,
              "[pim-runtime] WARN: lane-count disagreement. The compiler emitted "
              "%llu lanes for a %llu-byte access; elem_size=%d makes that %llu. "
              "Pricing the runtime's count.\n",
              (unsigned long long)compiler_lanes, (unsigned long long)size,
              elem_size, (unsigned long long)n_elements);
    }
  }

  dcc_round_note(is_write);
  cov_mark(tidx, t, base_addr, n_elements, is_write);

  /* WITHIN-CALL DQ-WORD COALESCING for loads. One __mem_trace_load call is ONE
   * machine instruction. A vector load of 8 fp16 values is one 128-bit bus
   * transaction, not eight, so it must cost one command per distinct DRAM column it
   * touches.
   *
   * LOADS ONLY. The lanes of one store call share an access ordinal, so the collapse
   * already folds every same-column store lane. An OPERAND load bypasses the collapse,
   * which is what leaves this copy load-bearing.
   *
   * STRICTLY within the call, and that is the whole point. Two SEPARATE scalar loads
   * of the same word, issued far apart, still pay twice, because by then the word is
   * gone. That is the k-outer penalty and it is real physics: the k-outer matmul
   * measures 0 memo fires against the k-packed kernel's 7/8, an 8x difference that
   * the compiler earns by consuming word-mates together. Grouping across calls would
   * hand k-outer a discount it never earned.
   *
   * Inert for scalar loads: n_elements == 1 takes the fast path below untouched, so
   * this changes nothing until a kernel actually emits vector loads. */
  if (!is_write && n_elements > 1) {
    uint64_t seen[PIM_COALESCE_MAX];
    int n_seen = 0;
    for (uint64_t i = 0; i < n_elements; i++) {
      uint64_t elem_addr = base_addr + i * (uint64_t)elem_size;
      if (elem_addr >= (uint64_t)t->base_addr + t->total_bytes) {
        int ntidx = find_tensor(elem_addr);
        if (ntidx < 0) {
          note_ignored();
          continue;
        }
        t = &tensors[ntidx];
      }
      int elem_idx =
          (int)((elem_addr - (uint64_t)t->base_addr) / t->elem_size);
      if (elem_idx < 0 || elem_idx >= t->num_elements) {
        note_ignored();
        continue;
      }
      /* Keyed by the element's physical tuple, not by the receiving bank where an
       * OPERAND load's bus write actually lands. Re-keying was tried 2026-09-10 and
       * reverted as inert, but the premise has since changed: the conv weight load
       * used to arrive one element per call and skip this path entirely, and it now
       * vectorizes. Re-keying is untested under that shape. */
      /* Key on the PLACED tuple. Keying on the interleaved home dropped every
       * word-mate before it reached place_in_lane_slab, so a partitioned tensor's
       * slab held one slot per word instead of one per element and came out 2-8x
       * too few columns (matmul B read 16 BR where its 16 words per lane owe 32). */
      pim_phys_loc_t pl;
      pl.elem_idx = elem_idx;
      map_element(t, elem_idx, &pl.ch, &pl.pch, &pl.bg, &pl.bank, &pl.sa, &pl.row,
                  &pl.col);
      if (t->bank_replicated >= 0 && t->lane_slot && t->lane_row_base >= 0)
        place_in_lane_slab(t, &pl);
      int gb = compute_global_bank(pl.ch, pl.pch, pl.bg, pl.bank);
      uint64_t key = ((uint64_t)gb & 0xFFFFULL) |
                     (((uint64_t)pl.sa & 0xFFULL) << 16) |
                     (((uint64_t)pl.row & 0xFFFFFFULL) << 24) |
                     (((uint64_t)pl.col & 0xFFFFULL) << 48);
      int dup = 0;
      for (int j = 0; j < n_seen; j++)
        if (seen[j] == key) { dup = 1; break; }
      if (dup) {
        stat_load_coalesced++;
        t->dedup_skips++;
        continue;
      }
      if (n_seen < PIM_COALESCE_MAX)
        seen[n_seen++] = key;
      else
        stat_coalesce_overflow++;
      pim_trace_access_one(t, elem_addr, is_write);
    }
    return;
  }

  for (uint64_t i = 0; i < n_elements; i++) {
    uint64_t elem_addr = base_addr + i * (uint64_t)elem_size;
    /* Re-resolve tensor if the vector spans tensor boundaries. Common case
     * (single-tensor vector) keeps the cached pointer. */
    if (elem_addr >= (uint64_t)t->base_addr + t->total_bytes) {
      int ntidx = find_tensor(elem_addr);
      if (ntidx < 0) {
        note_ignored();
        continue;
      }
      t = &tensors[ntidx];
    }
    pim_trace_access_one(t, elem_addr, is_write);
  }
}

/* im-lane-fold: a reduce just folded the lanes of one DRAM column, `outputs` times in
 * this lane, each a tree of `steps`. One instruction per lane, so it collapses across
 * bank replicas like an access. DCC prices each step as a PIM_MAC_OP1 in the bank. */
void __pim_trace_fold(int64_t steps, int64_t outputs) {
  if (!trace_fp || cur_phase != PIM_PHASE_COMPUTE || steps <= 0 || outputs <= 0)
    return;
  advance_program_epoch_if_needed();
  access_ord_next();
  int fch, fpch, fbg, fbank;
  decompose_global_bank(__pim_get_bank_id(), &fch, &fpch, &fbg, &fbank);
  for (int64_t o = 0; o < outputs; o++)
    for (int64_t k = 0; k < steps; k++) {
      if (g_lockstep_collapse) {
        /* Per channel, as the MACs it follows are. */
        uint64_t h = lockstep_key(0xF01DULL ^ ((uint64_t)fch << 16), cur_dispatch,
                                  g_access_ord, (uint64_t)o, (uint64_t)k);
        if (!addr_dedup_check_and_mark(g_lockstep_collapse, h, h >> 16, h >> 48))
          continue;
      }
      dp_record(DP_MAC, "PIM_MAC_OP1", g_last_mac);
      stat_fold_cmds++;
    }
}

/* im-relu-opcode brackets a load whose only use is a ReLU. DCC issues that read as
 * PIM_RELU, which their simulator does not space on the command bus. */
void __pim_trace_op(int32_t op) { g_cur_op = op; }

/* Fold commands emitted, so a harness can say which kernels owe one. */
uint64_t pim_dcc_fold_cmds(void) { return stat_fold_cmds; }

void __mem_trace_load(void *addr, uint64_t size, uint64_t lanes) {
  if (!trace_fp)
    return;
  pim_trace_access_range((uint64_t)addr, size, 0, lanes);
}

void __mem_trace_store(void *addr, uint64_t size, uint64_t lanes) {
  if (!trace_fp)
    return;
  pim_trace_access_range((uint64_t)addr, size, 1, lanes);
}

