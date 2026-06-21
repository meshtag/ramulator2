/*
 * MemTracePass — instruments every LLVM IR load/store with a runtime trace
 * call, classifying each load by which program-id axes its pointer chain
 * depends on so the runtime can apply axis-aware deduplication.
 *
 * Routing table for loads:
 *
 *                                        invariant in   reset when
 *   __pim_load_persistent       (mask 0) x, y, z        phase change only
 *   __pim_load_persistent_yz    (mask 1) y, z           pid_x changes
 *   __pim_load_persistent_xz    (mask 2) x, z           pid_y changes
 *   __pim_load_persistent_xy    (mask 4) x, y           pid_z changes
 *   __mem_trace_load          (mask 3+) (none)          per-program-id reset
 *
 * Stores route to __mem_trace_store unconditionally; the runtime applies
 * its own dedup at write-once granularity (per-program-id, the natural
 * scope under Triton's program-id-as-unit-of-work model).
 *
 * Mask convention: bit 0 = depends on pid_x, bit 1 = pid_y, bit 2 = pid_z.
 *
 * Why the mask is per-axis rather than a single scalar:
 *   In a kernel whose launch grid has multiple program-id axes, a tensor
 *   may be invariant in some axes and dependent on others. Treating "depends
 *   on any pid axis" as a single bit collapses that distinction and forces
 *   conservative per-program-id dedup even when broader scope is correct.
 *   With the per-axis mask, each load gets the widest scope its dataflow
 *   can prove safe, and the runtime resets each scope only when one of the
 *   axes that scope depends on actually changes.
 *
 * Why the analysis lives in LLVM IR rather than MLIR:
 *   The classification is a SEMANTIC property of the kernel — does the load's
 *   pointer chain reach a call to __pim_get_program_id*. Triton's MLIR IM-
 *   backend lowering preserves these calls intact through to LLVM IR, so the
 *   analysis can be done at either layer with the same answer. The walker
 *   logic is a use-def reachability check; it ports unchanged if we later
 *   move it to a TTGIR pass.
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

#include <cstdlib>

using namespace llvm;

namespace {

/* Mask bit assigned to each program-id source.
 *
 * Bank-id (__pim_get_bank_id) is deliberately NOT a source: it's a lane
 * index within a kernel invocation. PE k computes a different physical
 * address than PE k+1 for the same program-id, but PE k on program-id 0
 * and PE k on program-id 1 compute the SAME physical address (provided the
 * pointer is otherwise pid-invariant). That is precisely the pattern
 * persistent dedup is meant to capture, so bank-id dependence must not
 * disqualify a load from the persistent classification. */
static unsigned programIdSourceMask(const Function *F) {
  if (!F)
    return 0;
  StringRef Name = F->getName();
  if (Name == "__pim_get_program_id")
    return 0x1; // bit 0 = pid_x
  if (Name == "__pim_get_program_id_y")
    return 0x2; // bit 1 = pid_y
  if (Name == "__pim_get_program_id_z")
    return 0x4; // bit 2 = pid_z
  return 0;
}

/* Bounded DFS through the SSA use-def chain. Returns the OR of all
 * program-id source bits reachable from V. The bound (`MaxVisits`) caps
 * compile time on pathological IRs; under normal Triton lowering the chain
 * is a few dozen ops. */
static unsigned programIdAxesMask(Value *V, SmallPtrSetImpl<Value *> &Visited,
                                  int MaxVisits = 4096) {
  if (!V)
    return 0;
  if (!Visited.insert(V).second)
    return 0;
  if ((int)Visited.size() > MaxVisits)
    return 0;

  if (auto *CI = dyn_cast<CallInst>(V)) {
    unsigned mask = programIdSourceMask(CI->getCalledFunction());
    for (Use &U : CI->args())
      mask |= programIdAxesMask(U.get(), Visited, MaxVisits);
    return mask;
  }

  if (auto *I = dyn_cast<Instruction>(V)) {
    unsigned mask = 0;
    for (Use &U : I->operands())
      mask |= programIdAxesMask(U.get(), Visited, MaxVisits);
    return mask;
  }

  // Constants, function arguments, globals: not program-id-dependent.
  return 0;
}

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

struct MemTracePass : public PassInfoMixin<MemTracePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    LLVMContext &Ctx = M.getContext();

    // All trace entry points share the same (ptr, i64) signature.
    FunctionType *TraceFnTy = FunctionType::get(
        Type::getVoidTy(Ctx),
        {PointerType::getUnqual(Ctx), Type::getInt64Ty(Ctx)},
        false);

    FunctionCallee LoadFn =
        M.getOrInsertFunction("__mem_trace_load", TraceFnTy);
    FunctionCallee StoreFn =
        M.getOrInsertFunction("__mem_trace_store", TraceFnTy);
    FunctionCallee LoadPersistentFn =
        M.getOrInsertFunction("__pim_load_persistent", TraceFnTy);
    FunctionCallee LoadPersistentYZFn =
        M.getOrInsertFunction("__pim_load_persistent_yz", TraceFnTy);
    FunctionCallee LoadPersistentXZFn =
        M.getOrInsertFunction("__pim_load_persistent_xz", TraceFnTy);
    FunctionCallee LoadPersistentXYFn =
        M.getOrInsertFunction("__pim_load_persistent_xy", TraceFnTy);

    FunctionType *VoidFnTy =
        FunctionType::get(Type::getVoidTy(Ctx), false);
    FunctionCallee InitFn =
        M.getOrInsertFunction("__mem_trace_init", VoidFnTy);
    FunctionCallee FiniFn =
        M.getOrInsertFunction("__mem_trace_fini", VoidFnTy);

    // IM_PERSISTENT=0 disables axis-wise classification entirely: all loads
    // route to __mem_trace_load (per-program-id reset). Useful for ablation.
    const char *PersistEnv = std::getenv("IM_PERSISTENT");
    bool PersistentEnabled = !(PersistEnv && PersistEnv[0] == '0');

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

        // Classify by axis-mask. The walker is conservative — anything we
        // can't prove invariant routes to the per-program-id path.
        FunctionCallee Routed = LoadFn;
        if (PersistentEnabled) {
          SmallPtrSet<Value *, 32> Visited;
          unsigned mask = programIdAxesMask(Ptr, Visited);
          switch (mask) {
          case 0x0:
            Routed = LoadPersistentFn;
            break; // invariant in xyz
          case 0x1:
            Routed = LoadPersistentYZFn;
            break; // depends on x only
          case 0x2:
            Routed = LoadPersistentXZFn;
            break; // depends on y only
          case 0x4:
            Routed = LoadPersistentXYFn;
            break; // depends on z only
          default:
            Routed = LoadFn;
            break; // multi-axis dep
          }
        }
        Builder.CreateCall(
            Routed, {Ptr, ConstantInt::get(Type::getInt64Ty(Ctx), Size)});
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
