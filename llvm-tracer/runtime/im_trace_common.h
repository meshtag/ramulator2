/* im_trace_common.h: what the two trace runtimes share, pim_runtime.c for HBM-PIM and
 * simdram_runtime.c for SIMDRAM. It decodes the compiler's records and checks them against
 * the runtime, finds the tensor an access falls in, writes trace lines and holds the
 * MemTracePass hooks.
 *
 * Each runtime includes it once, after defining trace_fp, tensors, num_tensors and its
 * cfg_num_pch, cfg_num_bg and cfg_num_banks, which the helpers read. Each also defines
 * trace_access_range, which the hooks call. The two runtimes are never linked together, so
 * every library holds one copy of the hooks. */
#ifndef IM_TRACE_COMMON_H
#define IM_TRACE_COMMON_H

#include "pim_layout_table.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Roles as both runtimes number them, and as pipeline._PIM_ROLE passes them. */
enum { IM_ROLE_STREAMED, IM_ROLE_OPERAND, IM_ROLE_ACCUMULATOR };

static inline const char *role_name(int role) {
  static const char *const names[] = {"STREAMED", "OPERAND", "ACCUMULATOR"};
  return (role >= 0 && role < 3) ? names[role] : "?";
}

static inline void emit_trace(const char *op, int ch, int pch, int bg, int bank, int sa,
                              int row, int col) {
  fprintf(trace_fp, "%s %d,%d,%d,%d,%d,%d,%d\n", op, ch, pch, bg, bank, sa, row, col);
}

/* The registered tensor that holds addr, or -1. */
static inline int find_tensor(uint64_t addr) {
  for (int i = 0; i < num_tensors; i++) {
    uint64_t base = (uint64_t)tensors[i].base_addr;
    if (addr >= base && addr < base + tensors[i].total_bytes)
      return i;
  }
  return -1;
}

/* Flat bank id -> (ch, pch, bg, bank), bank innermost. */
static inline void decompose_global_bank(int g, int *ch, int *pch, int *bg, int *bank) {
  if (g < 0)
    g = 0;
  *bank = g % cfg_num_banks; g /= cfg_num_banks;
  *bg   = g % cfg_num_bg;    g /= cfg_num_bg;
  *pch  = g % cfg_num_pch;   g /= cfg_num_pch;
  *ch   = g;
}

/* The compiler bakes a bus width into every vector width it picks, so a run that models
 * another width is refused. */
static inline void check_compiler_dq_bits(const char *who, int cfg_bits) {
  if (__pim_dq_bits && __pim_dq_bits != cfg_bits) {
    fprintf(stderr,
            "[%s] ERROR: bus-width disagreement. The compiler chose vector widths for a "
            "%d-bit bus and this run models %d-bit, so every word count would be priced "
            "against a width the artifact never assumed.\n",
            who, (int)__pim_dq_bits, cfg_bits);
    exit(1);
  }
}

/* The compiler's kRecWords and PIM_LAYOUT_REC_WORDS live in separate submodules, and a
 * change to one alone would misread every record past the first. */
static inline void check_layout_rec_words(const char *who) {
  int emitted = (int)__pim_layout_rec_words;
  if (emitted > 0 && emitted != PIM_LAYOUT_REC_WORDS) {
    fprintf(stderr,
            "[%s] FATAL: layout record width mismatch. The compiler emitted %d words per "
            "record and this runtime strides by %d. The Triton and ramulator2 submodules "
            "have drifted, so every record past the first would be misread. Rebuild both.\n",
            who, emitted, PIM_LAYOUT_REC_WORDS);
    abort();
  }
}

/* The compiler's record for tensor_id, or NULL when the kernel carries no table or no
 * record for it. Pointer arguments register first and in signature order, so a record's
 * operand_arg is its tensor id. */
static inline const int32_t *layout_record(int tensor_id) {
  for (int i = 0; i < (int)__pim_layout_count; i++) {
    const int32_t *rec = __pim_layout_table + (size_t)i * PIM_LAYOUT_REC_WORDS;
    if (rec[PIM_LW_OPERAND_ARG] == tensor_id)
      return rec;
  }
  return NULL;
}

/* The role is the compiler's. is_store marks the output, and bank_replicated says whether
 * every bank sees the same elements. The host's role stands only where the record states
 * neither, and a disagreement is reported. */
static inline int role_from_record(const char *who, int tensor_id, const int32_t *rec,
                                   int host_role) {
  int role = host_role;
  if (rec[PIM_LW_IS_STORE] == 1)
    role = IM_ROLE_ACCUMULATOR;
  else if (rec[PIM_LW_BANK_REPLICATED] >= 0)
    role = rec[PIM_LW_BANK_REPLICATED] ? IM_ROLE_OPERAND : IM_ROLE_STREAMED;
  if (role != host_role)
    fprintf(stderr, "[%s] WARN tensor %d: host role %s, compiler layout says %s; taking the "
                    "compiler's.\n", who, tensor_id, role_name(host_role), role_name(role));
  return role;
}

/* MemTracePass passes the vector width the compiler emitted. The runtime prices the count it
 * derives from the registered element size and says once when the two differ, which means
 * the registered dtype and the IR element type have parted. */
static inline void check_access_lanes(const char *who, int *warned, uint64_t compiler_lanes,
                                      uint64_t size, int elem_size, uint64_t n) {
  if (compiler_lanes && compiler_lanes != n && !(*warned)++)
    fprintf(stderr, "[%s] WARN lane-count disagreement: the compiler emitted %llu lanes for a "
                    "%llu-byte access, elem_size=%d makes that %llu. Pricing the runtime's "
                    "count.\n", who, (unsigned long long)compiler_lanes,
            (unsigned long long)size, elem_size, (unsigned long long)n);
}

/* The MemTracePass hooks, called once per IR-level load or store. */
static void trace_access_range(uint64_t addr, uint64_t size, int is_write, uint64_t lanes);

void __mem_trace_load(void *addr, uint64_t size, uint64_t lanes) {
  if (trace_fp)
    trace_access_range((uint64_t)addr, size, 0, lanes);
}

void __mem_trace_store(void *addr, uint64_t size, uint64_t lanes) {
  if (trace_fp)
    trace_access_range((uint64_t)addr, size, 1, lanes);
}

#endif /* IM_TRACE_COMMON_H */
