/* pim_layout_table.h -- the compiler-emitted residency descriptor.
 *
 * emitPimLayoutTable (IMArtifact.cpp) puts these globals in the KERNEL object.
 * Kernel and runtime link into one dylib, so a runtime just reads them.
 *
 * Record: [operand_arg, bank_replicated, num_axes, is_store,
 * footprint...]. bank_replicated is the compiler-derived receive count: 0 when the
 * tensor is bank-partitioned, otherwise how many banks receive a copy, 1 meaning every
 * bank, and -1 when the pass said nothing and the host's role stands. operand_arg ==
 * tensor id, because pointer args are registered first and in signature order.
 *
 * TRAP: the widths below must equal kRecWords in IMArtifact.cpp, which lives in a
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
/* Four head words, the footprint axes, then two tail words. is_store is the one word that
 * names the output, whose footprint the SIMDRAM occupancy charge reads. */
#define PIM_LAYOUT_REC_WORDS (6 + PIM_MAX_FP_AXES * PIM_FP_AXIS_WORDS)

/* Word offsets within one record. Index through these and never a literal, so that a
 * change of record width moves every reader with it. */
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
/* Record width as the compiler emitted it, 0 when the kernel does not state it. Any other
 * value must equal PIM_LAYOUT_REC_WORDS, or the two submodules have drifted and every
 * record past the first is misread. */
__attribute__((weak)) extern const int32_t __pim_layout_rec_words;
/* Placement the compiler stated, 0 when the kernel said nothing. This runtime places
 * interleaved (__pim_layout_scheme 2) at the DQ word (__pim_placement_align 1) and
 * refuses any other stamp. */
__attribute__((weak)) extern const int32_t __pim_layout_scheme;
__attribute__((weak)) extern const int32_t __pim_placement_align;
/* 1 when the kernel loops over tiles inside one program instance. pim_runtime gives tile
 * bits of its dispatch id to such a kernel only, since a gridded kernel has many instances
 * and no tiles. */
__attribute__((weak)) extern const int32_t __pim_persistent;

/* Data-bus width in bits as the compiler assumed it when it chose vector widths, 0 when
 * the kernel does not state it. A runtime whose cfg_dq_bits differs refuses the kernel. */
__attribute__((weak)) extern const int32_t __pim_dq_bits;

/* Lanes the kernel was compiled for (threadsPerWarp = num_banks), 0 when the kernel does
 * not say. pim_runtime refuses any count but half its banks, one PE per bank pair. The
 * SIMDRAM occupancy divisor and leader gate follow the count, and a count that disagrees
 * with the configured banks shifts the charge by their ratio. */
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
