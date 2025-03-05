//===- FinchLoopletPasses.cpp - Finch Looplet passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "Finch/FinchPasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::finch {
#define GEN_PASS_DEF_FINCHLOOPLETRUN
#include "Finch/FinchPasses.h.inc"

namespace {

class FinchLoopletRunRewriter : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    auto loopIndex = forOp.getInductionVar();
    
    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();

    for (auto& accessOp : *forOp.getBody()) {
      if (!isa<mlir::finch::AccessOp>(accessOp)) 
        continue;
      
      Value accessIndex = accessOp.getOperand(1);
      if (accessIndex != loopIndex) 
        continue;
      
      auto runLooplet = dyn_cast<finch::RunOp>(
          accessOp.getOperand(0).getDefiningOp());
      if (!runLooplet) 
        continue;
        
      Value runValue = runLooplet.getOperand(); 
      
      // Replace Access to Run Value
      rewriter.replaceOp(&accessOp, runValue);
      
      return success();
    }
    
    return failure();
  }
};


class FinchLoopletRun
    : public impl::FinchLoopletRunBase<FinchLoopletRun> {
public:
  using impl::FinchLoopletRunBase<
      FinchLoopletRun>::FinchLoopletRunBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FinchLoopletRunRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};


} // namespace
} // namespace mlir::finch
