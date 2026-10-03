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
 * misread, silently. Only the runtime's check_layout_rec_words catches it.
 */
#ifndef PIM_LAYOUT_TABLE_H
#define PIM_LAYOUT_TABLE_H

#include <stdint.h>

#define PIM_MAX_FP_AXES 3
#define PIM_MAX_STRIDE_ARGS 3
#define PIM_FP_AXIS_WORDS (2 + PIM_MAX_STRIDE_ARGS)
/* 4 fixed words, the footprint axes, then two tail words (live split, cell fanout)
 * this runtime does not read. */
#define PIM_LAYOUT_REC_WORDS (6 + PIM_MAX_FP_AXES * PIM_FP_AXIS_WORDS)

/* Word offsets within one record. Index through these, never a literal: a bare rec[3]
 * survived a width change once and silently read num_axes as bank_replicated. */
#define PIM_LW_OPERAND_ARG 0
#define PIM_LW_BANK_REPLICATED 1
#define PIM_LW_NUM_AXES 2
#define PIM_LW_IS_STORE 3
#define PIM_LW_AXES_BASE 4

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
/* Whole-kernel bank-group interleave decision, 0 when the compiler did not ask. This
 * runtime refuses anything but 0. */
__attribute__((weak)) extern const int32_t __pim_bg_interleave;
/* Placement alignment, stated by the compiler: 1 aligns each tensor to the DQ word only,
 * 3 row-packs the lane slabs, 0 means the kernel said nothing. */
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

/* Lanes the kernel was compiled for (threadsPerWarp = num_banks). Lane placement puts
 * lane b in bank b, so this must equal the bank count. 0 = the kernel did not say. */
__attribute__((weak)) extern const int32_t __pim_lanes;

/* Cells ONE LANE keeps live across a reduction: the widest loop-carried accumulator, as
 * the compiler saw it. The register file has to hold this, so the bound compares it
 * against GRF_B. 0 = no loop-carried store, which is every elementwise kernel. */
__attribute__((weak)) extern const int32_t __pim_acc_cells_per_lane;

/* DCC's tile MAC (im-dcc-tile-mac), [operand_arg, column reads per MAC] pairs. Emitted only
 * when the kernel selected it, so count 0 is the default. */
#define PIM_MAX_TILE_MAC 16
__attribute__((weak)) extern const int32_t __pim_dcc_tile_mac[2 * PIM_MAX_TILE_MAC];
__attribute__((weak)) extern const int32_t __pim_dcc_tile_mac_count;

/* DCC's accumulator convention per stored tensor (im_dcc_acc_grf), [operand_arg, reset,
 * wb] triples. reset 0 none, 1 every lane. wb 1 every lane, 2 the addressed bank of each
 * PCU pair. __pim_dcc_grf_a lists the operands the compiler elected for GRF_A. Both count
 * only when __pim_dcc_decided is 1, which the compiler sets whenever it made the call. */
#define PIM_MAX_DCC_ACC 16
__attribute__((weak)) extern const int32_t __pim_dcc_acc[3 * PIM_MAX_DCC_ACC];
__attribute__((weak)) extern const int32_t __pim_dcc_acc_count;
__attribute__((weak)) extern const int32_t __pim_dcc_grf_a[PIM_MAX_DCC_ACC];
__attribute__((weak)) extern const int32_t __pim_dcc_grf_a_count;
__attribute__((weak)) extern const int32_t __pim_dcc_decided;

/* DCC's MAC addressing (im_dcc_mac_addressing), [operand_arg, head stride, reduce stride,
 * reduce extent] per tensor in elements: the geometry their generator addresses a tile by. */
__attribute__((weak)) extern const int32_t __pim_dcc_mac_addr[4 * PIM_MAX_TILE_MAC];
__attribute__((weak)) extern const int32_t __pim_dcc_mac_addr_count;

/* DCC's return stage sized from the input (im_dcc_return_from_input), [store_arg,
 * source_arg] pairs: the stored partials' return stage is as wide per bank as the input,
 * as their RED sizes it (gen_trace_HBMPIM_RED.py:150-151). */
__attribute__((weak)) extern const int32_t __pim_dcc_return_from[2 * PIM_MAX_TILE_MAC];
__attribute__((weak)) extern const int32_t __pim_dcc_return_from_count;

#ifdef PIM_LAYOUT_TABLE_DEFINE
const int32_t __pim_layout_table[PIM_LAYOUT_REC_WORDS] = {0};
const int32_t __pim_layout_count = 0;
const int32_t __pim_layout_rec_words = 0;
const int32_t __pim_bg_interleave = 0;
const int32_t __pim_dq_bits = 0;
const int32_t __pim_lanes = 0;
const int32_t __pim_acc_cells_per_lane = 0;
const int32_t __pim_dcc_tile_mac[2 * PIM_MAX_TILE_MAC] = {0};
const int32_t __pim_dcc_tile_mac_count = 0;
const int32_t __pim_dcc_acc[3 * PIM_MAX_DCC_ACC] = {0};
const int32_t __pim_dcc_acc_count = 0;
const int32_t __pim_dcc_grf_a[PIM_MAX_DCC_ACC] = {0};
const int32_t __pim_dcc_grf_a_count = 0;
const int32_t __pim_dcc_decided = 0;
const int32_t __pim_dcc_mac_addr[4 * PIM_MAX_TILE_MAC] = {0};
const int32_t __pim_dcc_mac_addr_count = 0;
const int32_t __pim_dcc_return_from[2 * PIM_MAX_TILE_MAC] = {0};
const int32_t __pim_dcc_return_from_count = 0;
const int32_t __pim_persistent = 0;
const int32_t __pim_placement_align = 0;
#endif

#endif /* PIM_LAYOUT_TABLE_H */
