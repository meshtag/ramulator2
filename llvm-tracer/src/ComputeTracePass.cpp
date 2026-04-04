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

struct ComputeTracePass : public PassInfoMixin<ComputeTracePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    LLVMContext &Ctx = M.getContext();

    // void __compute_trace(int32_t opcode, int32_t bit_width)
    FunctionType *TraceFnTy = FunctionType::get(
        Type::getVoidTy(Ctx), {Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx)},
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

      for (auto *BO : BinOps) {
        IRBuilder<> Builder(BO);
        int op = mapOpcode(BO->getOpcode());
        unsigned bits = BO->getType()->getScalarSizeInBits();
        Builder.CreateCall(ComputeFn,
                           {ConstantInt::get(Type::getInt32Ty(Ctx), op),
                            ConstantInt::get(Type::getInt32Ty(Ctx), bits)});
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
