/* pim_layout_table.h -- the compiler-emitted residency descriptor.
 *
 * emitPimLayoutTable (TritonIMToLLVM.cpp) puts these globals in the KERNEL object.
 * Kernel and runtime link into one dylib, so a runtime just reads them.
 *
 * Record: [operand_arg, layout_kind, reduction_col_axis, bank_replicated, num_axes,
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
#define PIM_LAYOUT_REC_WORDS (5 + PIM_MAX_FP_AXES * PIM_FP_AXIS_WORDS)

/* Word offsets within one record. */
#define PIM_LW_OPERAND_ARG 0
#define PIM_LW_KIND 1
#define PIM_LW_REDCOL 2
#define PIM_LW_BANK_REPLICATED 3
#define PIM_LW_NUM_AXES 4

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

#ifdef PIM_LAYOUT_TABLE_DEFINE
const int32_t __pim_layout_table[PIM_LAYOUT_REC_WORDS] = {0};
const int32_t __pim_layout_count = 0;
const int32_t __pim_layout_rec_words = 0;
const int32_t __pim_bg_interleave = 0;
#endif

#endif /* PIM_LAYOUT_TABLE_H */
