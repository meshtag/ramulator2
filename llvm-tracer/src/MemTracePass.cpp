/*
 * MemTracePass: instruments every LLVM IR load/store with a call into the trace
 * runtime (__mem_trace_load / __mem_trace_store, carrying the pointer and the
 * access size in bytes), and brackets main() with __mem_trace_init/_fini.
 *
 * Loads and stores in the scratchpad address space are on-chip reuse, not DRAM,
 * and are not instrumented.
 *
 * The call carries the LANE COUNT as well as the byte size, so the runtime can
 * check its own idea of how many values one access moves against the width the
 * compiler actually emitted, instead of inferring it and never finding out.
 *
 * The pid-invariance classifier that used to route some loads to a separate
 * __pim_load_persistent entry point was deleted 2026-09-10: it was runtime-
 * inferred cross-pid residency, retired 2026-06-22 and byte-identical inert since.
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {

// Scratchpad (on-chip PE register file / SRAM) address space. Memory ops in
// this space are NOT DRAM: they model operand reuse the im-operand-residency-
// layout lowering has staged into on-chip scratchpad (loaded once from global,
// then reused). MemTracePass therefore emits NO DRAM trace event for them, so
// the trace faithfully contains only real DRAM traffic — reuse realized in the
// IR, not by a runtime dedup (Phase B). The IM lowering must place scratchpad
// allocations and their loads/stores in this address space.
static constexpr unsigned IM_SCRATCHPAD_ADDRSPACE = 7;

static bool isScratchpadAccess(const Value *Ptr) {
  Type *T = Ptr->getType();
  return T->isPointerTy() &&
         T->getPointerAddressSpace() == IM_SCRATCHPAD_ADDRSPACE;
}

/// Lane count of one access: the vector width, or 1 for a scalar.
static uint64_t laneCount(Type *T) {
  if (auto *VT = dyn_cast<FixedVectorType>(T))
    return VT->getNumElements();
  return 1;
}

struct MemTracePass : public PassInfoMixin<MemTracePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    LLVMContext &Ctx = M.getContext();

    // (ptr, bytes, lanes). The lane count is the vector width the COMPILER chose;
    // without it the runtime reconstructs lanes as bytes / host-registered element
    // size and can never notice it meant something else. The two disagree whenever
    // the modelled element width differs from the array dtype, which is every
    // non-16-bit tensor today. Passing it lets the runtime say so.
    FunctionType *TraceFnTy = FunctionType::get(
        Type::getVoidTy(Ctx),
        {PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx),
         Type::getInt64Ty(Ctx)},
        false);

    FunctionCallee LoadFn =
        M.getOrInsertFunction("__mem_trace_load", TraceFnTy);
    FunctionCallee StoreFn =
        M.getOrInsertFunction("__mem_trace_store", TraceFnTy);

    FunctionType *VoidFnTy =
        FunctionType::get(Type::getVoidTy(Ctx), false);
    FunctionCallee InitFn =
        M.getOrInsertFunction("__mem_trace_init", VoidFnTy);
    FunctionCallee FiniFn =
        M.getOrInsertFunction("__mem_trace_fini", VoidFnTy);

    bool Modified = false;

    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      // Skip our own runtime helpers.
      if (F.getName().starts_with("__mem_trace") ||
          F.getName().starts_with("__pim_"))
        continue;

      // __mem_trace_init / fini at main's boundaries.
      if (F.getName() == "main") {
        IRBuilder<> EntryBuilder(&*F.getEntryBlock().getFirstInsertionPt());
        EntryBuilder.CreateCall(InitFn);
        for (auto &BB : F) {
          for (auto &I : BB) {
            if (auto *RI = dyn_cast<ReturnInst>(&I)) {
              IRBuilder<> RetBuilder(RI);
              RetBuilder.CreateCall(FiniFn);
            }
          }
        }
        Modified = true;
      }

      // Collect first to avoid iterator invalidation.
      SmallVector<LoadInst *, 64> Loads;
      SmallVector<StoreInst *, 64> Stores;
      for (auto &BB : F) {
        for (auto &I : BB) {
          // Scratchpad accesses are on-chip (reuse), not DRAM — do not trace.
          if (auto *LI = dyn_cast<LoadInst>(&I)) {
            if (!isScratchpadAccess(LI->getPointerOperand()))
              Loads.push_back(LI);
          } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
            if (!isScratchpadAccess(SI->getPointerOperand()))
              Stores.push_back(SI);
          }
        }
      }

      for (auto *LI : Loads) {
        IRBuilder<> Builder(LI);
        Value *Ptr = LI->getPointerOperand();
        uint64_t Size = M.getDataLayout().getTypeStoreSize(LI->getType());
        Builder.CreateCall(
            LoadFn, {Ptr, ConstantInt::get(Type::getInt64Ty(Ctx), Size),
                     ConstantInt::get(Type::getInt64Ty(Ctx),
                                      laneCount(LI->getType()))});
        Modified = true;
      }

      for (auto *SI : Stores) {
        IRBuilder<> Builder(SI);
        Value *Ptr = SI->getPointerOperand();
        Type *VT = SI->getValueOperand()->getType();
        uint64_t Size = M.getDataLayout().getTypeStoreSize(VT);
        Builder.CreateCall(
            StoreFn, {Ptr, ConstantInt::get(Type::getInt64Ty(Ctx), Size),
                      ConstantInt::get(Type::getInt64Ty(Ctx), laneCount(VT))});
        Modified = true;
      }
    }

    return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

} // anonymous namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "MemTracePass", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "mem-trace") {
                    MPM.addPass(MemTracePass());
                    return true;
                  }
                  return false;
                });
          }};
}
