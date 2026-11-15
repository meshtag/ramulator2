/* pim_layout_table.h -- the compiler-emitted residency descriptor.
 *
 * emitPimLayoutTable (TritonIMToLLVM.cpp) puts these globals in the KERNEL object.
 * Kernel and runtime link into one dylib, so a runtime just reads them.
 *
 * Record: [operand_arg, bank_replicated, num_axes, is_store,
 * footprint...]. bank_replicated is the compiler-DERIVED role: 1 when every bank sees
 * the same elements (deliver to each PE), 0 when the tensor is bank-partitioned, -1
 * when the pass said nothing and the host's role stands. operand_arg == tensor id,
 * because pointer args are registered first and in signature order.
 *
 * TRAP: the widths below must equal kRecWords in TritonIMToLLVM.cpp, which lives in a
 * DIFFERENT submodule. Change one without the other and every record past the first is
 * misread, silently. This header exists so at least the two runtimes cannot drift from
 * each other as well; the cross-submodule half still has no compile-time check.
 */
#ifndef PIM_LAYOUT_TABLE_H
#define PIM_LAYOUT_TABLE_H

#include <stdint.h>

#define PIM_MAX_FP_AXES 3
#define PIM_MAX_STRIDE_ARGS 3
#define PIM_FP_AXIS_WORDS (2 + PIM_MAX_STRIDE_ARGS)
/* Four head words, the footprint axes, then two tail words. Two words left on
 * 2026-09-10 (reduction_col_axis and layout_kind). is_store arrived 2026-09-10 because nothing
 * else says which tensor is the output once reuse_class stopped travelling, and the
 * SIMDRAM occupancy charge needs the OUTPUT footprint specifically. The drift check
 * below catches a half-rebuilt pair loudly. */
#define PIM_LAYOUT_REC_WORDS (6 + PIM_MAX_FP_AXES * PIM_FP_AXIS_WORDS)

/* Word offsets within one record. Index through these, never a literal: a bare rec[3]
 * survived a width change once and silently read num_axes as bank_replicated. */
#define PIM_LW_OPERAND_ARG 0
#define PIM_LW_BANK_REPLICATED 1
#define PIM_LW_NUM_AXES 2
#define PIM_LW_IS_STORE 3
#define PIM_LW_AXES_BASE 4
/* TAIL word, after the padded axis block, so every index above keeps its value.
 * The factor a rank-collapsing reduce hid from the store's footprint. It MULTIPLIES
 * the derived occupancy, and is 1 unless a reduce sits between the loop-carried
 * accumulator and the store, so it is the identity on every kernel but split-K. */
#define PIM_LW_LIVE_SPLIT (4 + PIM_MAX_FP_AXES * PIM_FP_AXIS_WORDS)
/* SECOND TAIL word. Output cells in ONE LANE that consume a single loaded value, so a
 * bit-serial machine has to write the value into that many columns. 1 on every tensor
 * whose value feeds exactly one cell, which is every matvec and every accumulator. */
#define PIM_LW_CELL_FANOUT (5 + PIM_MAX_FP_AXES * PIM_FP_AXIS_WORDS)

/* Weak DEFINITIONS, not weak references. A weak reference does not link on Mach-O when
 * nothing defines the symbol, which is the normal case whenever
 * im-operand-residency-layout is skipped. So count == 0, not a null pointer, is the
 * "no compiler decision" signal. The split decl/def keeps external linkage under C++
 * and dodges -Wextern-initializer. Define PIM_LAYOUT_TABLE_DEFINE in exactly one
 * translation unit per runtime. */
__attribute__((weak)) extern const int32_t __pim_layout_table[PIM_LAYOUT_REC_WORDS];
__attribute__((weak)) extern const int32_t __pim_layout_count;
/* Record width as the COMPILER emitted it. 0 means this kernel predates the check or
 * carries no table; any other value must equal PIM_LAYOUT_REC_WORDS or the two
 * submodules have drifted and every record past the first is misread. */
__attribute__((weak)) extern const int32_t __pim_layout_rec_words;
/* Placement the compiler stated, 0 when the kernel said nothing. This runtime places
 * interleaved (__pim_layout_scheme 2) at the DQ word (__pim_placement_align 1) and
 * refuses any other stamp. */
__attribute__((weak)) extern const int32_t __pim_layout_scheme;
__attribute__((weak)) extern const int32_t __pim_placement_align;
/* 1 when the kernel loops over tiles inside one program instance. The trace
 * runtime needs it to pick how it packs its dispatch id: a persistent kernel has
 * few instances and many tiles, a gridded one has many instances and no tiles, and
 * reserving tile bits unconditionally overflowed the field on a 128-instance grid. */
__attribute__((weak)) extern const int32_t __pim_persistent;

/* Data-bus width in bits AS THE COMPILER ASSUMED IT when it chose vector widths.
 * The runtime holds the same quantity as cfg_dq_bits. They were decided
 * independently until 2026-09-10, in two languages with no shared symbol; this
 * makes a disagreement sayable. 0 means the kernel predates the emission. */
__attribute__((weak)) extern const int32_t __pim_dq_bits;

/* Lanes the kernel was compiled for (threadsPerWarp = num_banks). The SIMDRAM occupancy
 * divisor and the leader gate assume one lane per bank; if this disagrees with the
 * configured bank count the charge shifts by their ratio. 0 = the kernel did not say. */
__attribute__((weak)) extern const int32_t __pim_lanes;

#ifdef PIM_LAYOUT_TABLE_DEFINE
const int32_t __pim_layout_table[PIM_LAYOUT_REC_WORDS] = {0};
const int32_t __pim_layout_count = 0;
const int32_t __pim_layout_rec_words = 0;
const int32_t __pim_dq_bits = 0;
const int32_t __pim_lanes = 0;
const int32_t __pim_persistent = 0;
const int32_t __pim_layout_scheme = 0;
const int32_t __pim_placement_align = 0;
#endif

#endif /* PIM_LAYOUT_TABLE_H */
