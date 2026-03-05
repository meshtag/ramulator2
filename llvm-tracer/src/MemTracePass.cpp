#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct MemTracePass : public PassInfoMixin<MemTracePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    LLVMContext &Ctx = M.getContext();

    // void __mem_trace_load(void* addr, uint64_t size)
    FunctionType *TraceFnTy = FunctionType::get(
        Type::getVoidTy(Ctx),
        {PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx)},
        false);

    FunctionCallee LoadFn =
        M.getOrInsertFunction("__mem_trace_load", TraceFnTy);
    FunctionCallee StoreFn =
        M.getOrInsertFunction("__mem_trace_store", TraceFnTy);

    // void __mem_trace_init(void) / void __mem_trace_fini(void)
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
      // Don't instrument our own runtime functions
      if (F.getName().starts_with("__mem_trace"))
        continue;

      // Insert __mem_trace_init() at the entry of main
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

      // Collect instructions first to avoid iterator invalidation
      SmallVector<LoadInst *, 64> Loads;
      SmallVector<StoreInst *, 64> Stores;

      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *LI = dyn_cast<LoadInst>(&I))
            Loads.push_back(LI);
          else if (auto *SI = dyn_cast<StoreInst>(&I))
            Stores.push_back(SI);
        }
      }

      for (auto *LI : Loads) {
        IRBuilder<> Builder(LI);
        Value *Ptr = LI->getPointerOperand();
        uint64_t Size =
            M.getDataLayout().getTypeStoreSize(LI->getType());
        Builder.CreateCall(LoadFn,
                           {Ptr, ConstantInt::get(Type::getInt64Ty(Ctx), Size)});
        Modified = true;
      }

      for (auto *SI : Stores) {
        IRBuilder<> Builder(SI);
        Value *Ptr = SI->getPointerOperand();
        uint64_t Size = M.getDataLayout().getTypeStoreSize(
            SI->getValueOperand()->getType());
        Builder.CreateCall(StoreFn,
                           {Ptr, ConstantInt::get(Type::getInt64Ty(Ctx), Size)});
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
