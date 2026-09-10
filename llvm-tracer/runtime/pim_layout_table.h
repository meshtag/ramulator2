/* pim_layout_table.h -- the compiler-emitted residency descriptor.
 *
 * emitPimLayoutTable (TritonIMToLLVM.cpp) puts these globals in the KERNEL object.
 * Kernel and runtime link into one dylib, so a runtime just reads them.
 *
 * Record: [operand_arg, bank_replicated, num_axes,
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
/* 3 fixed words then the footprint axes. Two words left on 2026-09-10:
 * reduction_col_axis with the reduction-to-column lever, and layout_kind, whose only
 * consumer (the ROW_DUP broadcast collapse) had been reverted on fairness grounds.
 * The drift check below catches a half-rebuilt pair loudly. */
#define PIM_LAYOUT_REC_WORDS (3 + PIM_MAX_FP_AXES * PIM_FP_AXIS_WORDS)

/* Word offsets within one record. */
#define PIM_LW_OPERAND_ARG 0
#define PIM_LW_BANK_REPLICATED 1
#define PIM_LW_NUM_AXES 2

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
/* Whole-kernel bank-group interleave decision, 0 when the compiler did not ask. */
__attribute__((weak)) extern const int32_t __pim_bg_interleave;
/* PLACEMENT POLICY, stated by the compiler so the choice lives in the artifact
 * instead of in a runtime default plus an env var. 0 means the kernel said nothing
 * and the runtime's default stands; the env vars then become overrides that warn.
 *
 *   __pim_layout_scheme    1 = striped (legacy compact/row-stripe), 2 = interleaved
 *   __pim_placement_align  1 = align each tensor to the DQ word only, so consecutive
 *                              tensors share rows but land on different banks
 *                          2 = align to a global-row boundary, so every tensor starts
 *                              at bank 0
 *
 * The alignment rule is not cosmetic: aligning to a global row hot-spots every tensor
 * on bank 0 and measured 3.2x worse on conv2d_1x8x16x16x3x3. That number used to
 * justify a hardcoded constant in the allocator. Naming it makes it a lever the
 * compiler picks and an ablation can vary. */
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

#ifdef PIM_LAYOUT_TABLE_DEFINE
const int32_t __pim_layout_table[PIM_LAYOUT_REC_WORDS] = {0};
const int32_t __pim_layout_count = 0;
const int32_t __pim_layout_rec_words = 0;
const int32_t __pim_bg_interleave = 0;
const int32_t __pim_dq_bits = 0;
const int32_t __pim_persistent = 0;
const int32_t __pim_layout_scheme = 0;
const int32_t __pim_placement_align = 0;
#endif

#endif /* PIM_LAYOUT_TABLE_H */
