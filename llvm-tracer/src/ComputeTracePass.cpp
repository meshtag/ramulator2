#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

/* simdram_op_t, the opcode numbering __compute_trace takes. */
#include "../runtime/simdram_runtime.h"

using namespace llvm;

namespace {

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
 * Forward def-use walk: does a BinaryOperator's result flow into a data store? Returns the
 * first StoreInst reached and sets DestAddr to its pointer operand. Only a plain store
 * counts, as in MemTracePass, since the IM backend lowers a masked store to a branch
 * around one.
 *
 * Arithmetic that reaches no store is address, index or loop-control math, host work that
 * ends at GEPs, compares, induction PHIs or int/ptr casts, so it is not charged and the
 * compute count stays proportional to the output whatever the unrolling. The walk passes
 * through value-propagating ops and intrinsics and stops at loads, GEPs, int/ptr
 * conversions and opaque calls. A vector BO resolves to its store's tile base, and a BO
 * feeding several stores to the first one reached.
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

/* The kernel pointer argument a destination derives from through GEPs, casts and constant
 * expressions, or nullptr. Under lane placement an argument's base address resolves to the
 * executing lane's bank, so passing it keeps the charge in the right bank when the element
 * pointer cannot be passed. */
static Value *argumentBaseOf(Value *V) {
  SmallSet<Value *, 16> seen;
  while (V && seen.insert(V).second) {
    if (isa<Argument>(V))
      return V;
    if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) { V = GEP->getPointerOperand(); continue; }
    if (auto *CI = dyn_cast<CastInst>(V)) { V = CI->getOperand(0); continue; }
    if (auto *CE = dyn_cast<ConstantExpr>(V)) { V = CE->getOperand(0); continue; }
    return nullptr;
  }
  return nullptr;
}

struct ComputeTracePass : public PassInfoMixin<ComputeTracePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    LLVMContext &Ctx = M.getContext();

    // void __compute_trace(int32_t opcode, int32_t lanes, void *dest_addr)
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
        Value *DestAddr = nullptr;
        Instruction *Store = reachesDataStore(BO, DestAddr);
        if (!Store)
          continue;

        int op = mapOpcode(BO->getOpcode());
        unsigned lanes = 1;
        if (auto *VT = dyn_cast<FixedVectorType>(BO->getType()))
          lanes = VT->getNumElements();
        Value *opArg = ConstantInt::get(Type::getInt32Ty(Ctx), op);
        Value *lanesArg = ConstantInt::get(Type::getInt32Ty(Ctx), lanes);

        // Inserted at the BO, so the call fires once per arithmetic execution and a K-deep
        // MAC chain pays K times. dest_addr is normally the tensor argument, because the
        // accumulator is stored after the loop and its element pointer does not dominate
        // the BO. The element pointer is passed when it dominates, and null when neither
        // names a tensor.
        IRBuilder<> Builder(BO->getNextNode());
        Value *Dest = NullPtr;
        if (DestAddr) {
          auto *DestI = dyn_cast<Instruction>(DestAddr);
          if (!DestI || DT.dominates(DestI, BO))
            Dest = DestAddr;
          else if (Value *Base = argumentBaseOf(DestAddr))
            Dest = Base;
          if (Dest->getType() != PtrTy)
            Dest = Builder.CreateBitOrPointerCast(Dest, PtrTy);
        }
        Builder.CreateCall(ComputeFn, {opArg, lanesArg, Dest});
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
