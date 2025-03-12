//===- FinchSequencePasses.cpp - Finch Sequence pass -----------------*- C++ -*-===//
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
#define GEN_PASS_DEF_FINCHLOOPLETSEQUENCE
#include "Finch/FinchPasses.h.inc"

namespace {

class FinchLoopletSequenceRewriter : public OpRewritePattern<scf::ForOp> {
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
      
      auto seqLooplet = dyn_cast<finch::SequenceOp>(
          accessOp.getOperand(0).getDefiningOp());
      if (!seqLooplet) 
        continue;
         
      rewriter.setInsertionPoint(forOp);
      // Main Sequence Rewrite          
      IRMapping mapper1;
      IRMapping mapper2;
      Operation* newForOp1 = rewriter.clone(*forOp, mapper1);
      Operation* newForOp2 = rewriter.clone(*forOp, mapper2);
      rewriter.moveOpAfter(newForOp1, forOp);
      rewriter.moveOpAfter(newForOp2, newForOp1);

      // Replace Access operand with Sequence bodies
      auto newAccess1 = mapper1.lookupOrDefault(&accessOp);
      auto newAccess2 = mapper2.lookupOrDefault(&accessOp);
      auto bodyLooplet1 = seqLooplet.getOperand(1);
      auto bodyLooplet2 = seqLooplet.getOperand(2);
      auto newBodyLooplet1 = mapper1.lookupOrDefault(bodyLooplet1);
      auto newBodyLooplet2 = mapper2.lookupOrDefault(bodyLooplet2);
      newAccess1->setOperand(0, newBodyLooplet1);
      newAccess2->setOperand(0, newBodyLooplet2);
     
      // scf.for loop bounds
      Value loopLb = forOp.getLowerBound();
      Value loopUb = forOp.getUpperBound();

      // Finch.jl used both closed endpoints [st,en],
      // but Finch.mlir uses [st,en) to align syntax 
      // with scf.for that uses [st,en).
      //
      //       firstBodyUb=secondBodyLb
      //                  v
      // [---firstBody---)[---secondBody---)
      Value firstBodyUb = seqLooplet.getOperand(0);
      Value secondBodyLb = firstBodyUb;
      if (!firstBodyUb.getType().isIndex()) {
        firstBodyUb = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), firstBodyUb);
        secondBodyLb = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), secondBodyLb);
      }         
      
      // intersect
      Value newFirstLoopUb = rewriter.create<arith::MinUIOp>(
          loc, loopUb, firstBodyUb);
      Value newSecondLoopLb = rewriter.create<arith::MaxUIOp>(
          loc, loopLb, secondBodyLb);
      
      // set new loop bound with intersection
      cast<scf::ForOp>(newForOp1).setUpperBound(newFirstLoopUb);
      cast<scf::ForOp>(newForOp2).setLowerBound(newSecondLoopLb);

      rewriter.eraseOp(forOp);
      
      return success();
    }
    return failure();
  }
};

class FinchLoopletSequence
    : public impl::FinchLoopletSequenceBase<FinchLoopletSequence> {
public:
  using impl::FinchLoopletSequenceBase<
      FinchLoopletSequence>::FinchLoopletSequenceBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FinchLoopletSequenceRewriter>(&getContext()); 
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::finch
