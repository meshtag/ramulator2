#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
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
 * Forward def-use walker to find the eventual StoreInst that a
 * BinaryOperator's result flows into.
 *
 * Returns the StoreInst (DFS-first reached); the caller takes the
 * store's pointer operand as the destination address. We return the
 * Store rather than the address so the caller can insert the
 * compute_trace call immediately before the Store — that guarantees
 * the address (the Store's pointer operand) dominates the call site
 * in SSA, avoiding "instruction does not dominate all uses" verifier
 * failures that would result from instrumenting at the BO position
 * where the destination address is computed later in program order.
 *
 * Walks transparently through value-propagating ops (BinaryOperator,
 * PHINode, SelectInst, CastInst, vector shuffles, value-producing
 * intrinsics like fmuladd/vector reductions). Stops at LoadInst,
 * GEPs, int↔ptr conversions, and opaque calls — those don't propagate
 * our value into a stored destination.
 *
 * Imprecision:
 *   - Vector BO: all lanes resolve to the vector store's tile-base
 *     pointer. For SIMDRAM's bit-serial layout where dq_bits values
 *     share one col_slot, this resolves to the col_slot containing
 *     the tile's first lane — acceptable granularity.
 *   - DAG fan-out (BO feeds multiple stores): we return the first
 *     store reached. Mild misattribution for fused-with-scratch
 *     kernels.
 *   - BO never stored: returns nullptr; caller falls back to
 *     instrumenting at the BO position with a null dest_addr, which
 *     the runtime resolves via current_acc.
 */
static StoreInst *findEventualStore(BinaryOperator *BO) {
  SmallVector<Value *, 16> worklist;
  SmallSet<Value *, 32> visited;
  worklist.push_back(BO);

  while (!worklist.empty()) {
    Value *cur = worklist.pop_back_val();
    if (!visited.insert(cur).second)
      continue;

    for (User *U : cur->users()) {
      if (auto *Store = dyn_cast<StoreInst>(U)) {
        if (Store->getValueOperand() == cur)
          return Store;
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

      for (auto *BO : BinOps) {
        int op = mapOpcode(BO->getOpcode());
        unsigned bits = BO->getType()->getScalarSizeInBits();
        Value *opArg = ConstantInt::get(Type::getInt32Ty(Ctx), op);
        Value *bitsArg = ConstantInt::get(Type::getInt32Ty(Ctx), bits);

        // Forward def-use to find the BinaryOp's eventual store.
        // Insert __compute_trace at the STORE position (not the BO
        // position) so the dest_addr (the store's pointer operand)
        // dominates the call — instrumenting at the BO would reference
        // a not-yet-defined SSA value and trip the LLVM verifier.
        StoreInst *Store = findEventualStore(BO);

        if (Store) {
          IRBuilder<> Builder(Store);
          Value *DestAddr = Store->getPointerOperand();
          if (DestAddr->getType() != PtrTy)
            DestAddr = Builder.CreateBitOrPointerCast(DestAddr, PtrTy);
          Builder.CreateCall(ComputeFn, {opArg, bitsArg, DestAddr});
        } else {
          // No store found — instrument at the BO position with a
          // null dest_addr; the runtime falls back to current_acc.
          IRBuilder<> Builder(BO->getNextNode() ? BO->getNextNode()
                                                : (Instruction *)BO);
          Builder.CreateCall(ComputeFn, {opArg, bitsArg, NullPtr});
        }
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
