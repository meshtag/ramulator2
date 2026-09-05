#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {

/*
 * Arithmetic opcode enum — must match simdram_runtime.h simdram_op_t.
 * Passed as the first argument to __compute_trace().
 */
enum SimdramOp {
  SIMDRAM_OP_ADD = 0,
  SIMDRAM_OP_SUB,
  SIMDRAM_OP_MUL,
  SIMDRAM_OP_DIV,
  SIMDRAM_OP_REM,
  SIMDRAM_OP_AND,
  SIMDRAM_OP_OR,
  SIMDRAM_OP_XOR,
  SIMDRAM_OP_SHL,
  SIMDRAM_OP_LSHR,
  SIMDRAM_OP_ASHR,
  SIMDRAM_OP_FADD,
  SIMDRAM_OP_FSUB,
  SIMDRAM_OP_FMUL,
  SIMDRAM_OP_FDIV,
  SIMDRAM_OP_FREM,
  SIMDRAM_OP_OTHER
};

static int mapOpcode(unsigned LLVMOp) {
  switch (LLVMOp) {
  case Instruction::Add:
    return SIMDRAM_OP_ADD;
  case Instruction::Sub:
    return SIMDRAM_OP_SUB;
  case Instruction::Mul:
    return SIMDRAM_OP_MUL;
  case Instruction::UDiv:
  case Instruction::SDiv:
    return SIMDRAM_OP_DIV;
  case Instruction::URem:
  case Instruction::SRem:
    return SIMDRAM_OP_REM;
  case Instruction::And:
    return SIMDRAM_OP_AND;
  case Instruction::Or:
    return SIMDRAM_OP_OR;
  case Instruction::Xor:
    return SIMDRAM_OP_XOR;
  case Instruction::Shl:
    return SIMDRAM_OP_SHL;
  case Instruction::LShr:
    return SIMDRAM_OP_LSHR;
  case Instruction::AShr:
    return SIMDRAM_OP_ASHR;
  case Instruction::FAdd:
    return SIMDRAM_OP_FADD;
  case Instruction::FSub:
    return SIMDRAM_OP_FSUB;
  case Instruction::FMul:
    return SIMDRAM_OP_FMUL;
  case Instruction::FDiv:
    return SIMDRAM_OP_FDIV;
  case Instruction::FRem:
    return SIMDRAM_OP_FREM;
  default:
    return SIMDRAM_OP_OTHER;
  }
}

/*
 * Forward def-use walker: does a BinaryOperator's result flow into a genuine
 * DATA store? Returns the terminal store instruction (a StoreInst, or an
 * llvm.masked.store / llvm.masked.scatter) reached DFS-first; the caller
 * inserts __compute_trace immediately before it so the destination address
 * dominates the call site in SSA (avoids "instruction does not dominate all
 * uses" verifier failures). Sets DestAddr to the store's pointer operand
 * (null for masked.scatter's vector-of-pointers → runtime resolves via
 * current_acc).
 *
 * W4 FIX (2026-07-01): returns nullptr for BinaryOperators whose value never
 * reaches a data store — i.e. pure ADDRESS/INDEX/LOOP-CONTROL arithmetic, which
 * flows only into GEPs (address computation), CmpInsts + PHIs (loop induction),
 * or int↔ptr casts, and NEVER into a stored data value. Previously such BOs
 * were still charged as PE compute via a null-dest fallback, so the SIMDRAM
 * compute column over-counted host loop-counter/address arithmetic as bit-serial
 * PE work (the reviewer's construct-validity + unroll-sensitivity bug). Charging
 * only value-producing compute that lands in a data store makes the compute count
 * proportional to actual output elements, hence unroll-invariant. Legitimate
 * INTEGER PE compute (bit-serial int8/int4 MAC) is preserved: it reaches a data
 * store, unlike index math which terminates at a GEP.
 *
 * Walks transparently through value-propagating ops (BinaryOperator, PHINode,
 * SelectInst, CastInst, vector shuffles, value-producing intrinsics like
 * fmuladd/vector reductions). Stops at LoadInst, GEPs, int↔ptr conversions, and
 * opaque calls — those don't propagate our value into a stored destination.
 *
 * Imprecision:
 *   - Vector BO: all lanes resolve to the vector store's tile-base pointer.
 *   - DAG fan-out (BO feeds multiple stores): first store reached.
 */
static Instruction *reachesDataStore(BinaryOperator *BO, Value *&DestAddr) {
  SmallVector<Value *, 16> worklist;
  SmallSet<Value *, 32> visited;
  worklist.push_back(BO);

  while (!worklist.empty()) {
    Value *cur = worklist.pop_back_val();
    if (!visited.insert(cur).second)
      continue;

    for (User *U : cur->users()) {
      if (auto *Store = dyn_cast<StoreInst>(U)) {
        if (Store->getValueOperand() == cur) {
          DestAddr = Store->getPointerOperand();
          return Store;
        }
        continue;
      }

      if (isa<GetElementPtrInst>(U) || isa<LoadInst>(U) ||
          isa<PtrToIntInst>(U) || isa<IntToPtrInst>(U))
        continue;

      if (auto *Call = dyn_cast<CallInst>(U)) {
        if (auto *II = dyn_cast<IntrinsicInst>(Call)) {
          switch (II->getIntrinsicID()) {
          // Masked vector stores are genuine data stores (vectorized epilogue).
          case Intrinsic::masked_store:
            if (II->getArgOperand(0) == cur) {   // (value, ptr, align, mask)
              DestAddr = II->getArgOperand(1);
              return II;
            }
            break;
          case Intrinsic::masked_scatter:
            if (II->getArgOperand(0) == cur) {   // (value, <vec ptr>, align, mask)
              DestAddr = nullptr;                 // vector of ptrs → current_acc
              return II;
            }
            break;
          case Intrinsic::fmuladd:
          case Intrinsic::fma:
          case Intrinsic::minnum:
          case Intrinsic::maxnum:
          case Intrinsic::minimum:
          case Intrinsic::maximum:
          case Intrinsic::sqrt:
          case Intrinsic::exp:
          case Intrinsic::exp2:
          case Intrinsic::log:
          case Intrinsic::log2:
          case Intrinsic::pow:
          case Intrinsic::fabs:
          case Intrinsic::vector_reduce_add:
          case Intrinsic::vector_reduce_mul:
          case Intrinsic::vector_reduce_fadd:
          case Intrinsic::vector_reduce_fmul:
          case Intrinsic::vector_reduce_fmax:
          case Intrinsic::vector_reduce_fmin:
            worklist.push_back(II);
            break;
          default:
            break;
          }
        }
        continue;
      }

      if (isa<BinaryOperator>(U) || isa<PHINode>(U) ||
          isa<SelectInst>(U) || isa<CastInst>(U) ||
          isa<InsertElementInst>(U) || isa<ExtractElementInst>(U) ||
          isa<InsertValueInst>(U) || isa<ExtractValueInst>(U) ||
          isa<ShuffleVectorInst>(U) || isa<UnaryOperator>(U) ||
          isa<CmpInst>(U) || isa<FreezeInst>(U)) {
        worklist.push_back(U);
        continue;
      }
    }
  }

  return nullptr;
}

struct ComputeTracePass : public PassInfoMixin<ComputeTracePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    LLVMContext &Ctx = M.getContext();

    // void __compute_trace(int32_t opcode, int32_t bit_width,
    //                       void *dest_addr);
    // dest_addr is the destination address that the BinaryOperator's
    // result eventually flows into (via def-use chains). The runtime
    // uses it to attribute compute to a specific output cell. A null
    // dest_addr means "no destination determined" — the runtime falls
    // back to current_acc-based addressing.
    PointerType *PtrTy = PointerType::get(Ctx, /*AddressSpace=*/0);
    FunctionType *TraceFnTy = FunctionType::get(
        Type::getVoidTy(Ctx),
        {Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), PtrTy},
        false);

    FunctionCallee ComputeFn =
        M.getOrInsertFunction("__compute_trace", TraceFnTy);

    bool Modified = false;

    auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      // Don't instrument our own runtime functions
      if (F.getName().starts_with("__mem_trace") ||
          F.getName().starts_with("__compute_trace"))
        continue;

      // Collect arithmetic instructions to avoid iterator invalidation
      SmallVector<BinaryOperator *, 64> BinOps;

      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *BO = dyn_cast<BinaryOperator>(&I))
            BinOps.push_back(BO);
        }
      }

      Value *NullPtr = ConstantPointerNull::get(PtrTy);
      DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);

      for (auto *BO : BinOps) {
        // W4 FIX: only charge BinaryOperators whose value flows into a genuine
        // DATA store as PE compute. BOs that never reach a data store are pure
        // address/index/loop-control arithmetic (they terminate at GEPs, CmpInsts
        // or int↔ptr casts) and are NOT bit-serial PE work — skipping them removes
        // the host-arithmetic over-count and makes the compute cost unroll-invariant.
        Value *DestAddr = nullptr;
        Instruction *Store = reachesDataStore(BO, DestAddr);
        if (!Store)
          continue;

        int op = mapOpcode(BO->getOpcode());
        unsigned bits = BO->getType()->getScalarSizeInBits();
        Value *opArg = ConstantInt::get(Type::getInt32Ty(Ctx), op);
        Value *bitsArg = ConstantInt::get(Type::getInt32Ty(Ctx), bits);

        // Insert at the BO, so the call fires once per ARITHMETIC EXECUTION.
        // Inserting at the store (the old behavior) put it OUTSIDE the reduction
        // loop, so a K-deep MAC chain was charged like a single MAC and SIMDRAM
        // cost was independent of the arithmetic. dest_addr is passed only when it
        // dominates the BO; otherwise null, and the runtime attributes to
        // current_acc. (void)Store keeps reachesDataStore as the "is this real PE
        // work vs address math" filter.
        (void)Store;
        IRBuilder<> Builder(BO->getNextNode());
        Value *Dest = NullPtr;
        if (DestAddr) {
          auto *DestI = dyn_cast<Instruction>(DestAddr);
          if (!DestI || DT.dominates(DestI, BO)) {
            Dest = DestAddr;
            if (Dest->getType() != PtrTy)
              Dest = Builder.CreateBitOrPointerCast(Dest, PtrTy);
          }
        }
        Builder.CreateCall(ComputeFn, {opArg, bitsArg, Dest});
        Modified = true;
      }
    }

    return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

} // anonymous namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "ComputeTracePass", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "compute-trace") {
                    MPM.addPass(ComputeTracePass());
                    return true;
                  }
                  return false;
                });
          }};
}
