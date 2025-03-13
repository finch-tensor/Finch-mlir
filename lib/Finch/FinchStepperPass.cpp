//===- FinchStepperPasses.cpp - Finch Stepper passes -----------------*- C++ -*-===//
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
#define GEN_PASS_DEF_FINCHLOOPLETSTEPPER
#include "Finch/FinchPasses.h.inc"

namespace {

class FinchLoopletStepperRewriter : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    auto indVar = forOp.getInductionVar();
    
    //llvm::outs() << "(0)\n";
    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();

    // Collect all the steppers from accesses
    IRMapping mapper;
    SmallVector<finch::StepperOp, 4> stepperLooplets;
    SmallVector<finch::AccessOp, 4> accessOps;
    for (auto& accessOp : *forOp.getBody()) {
      if (isa<mlir::finch::AccessOp>(accessOp)) {
        Value accessVar = accessOp.getOperand(1);
        if (accessVar == indVar) {
          Operation* looplet = accessOp.getOperand(0).getDefiningOp();
          if (isa<finch::StepperOp>(looplet)) {
            // There can be multiple uses of this Stepper.
            // We don't want to erase original Stepper when lowering
            // because of other use.
            // So everytime we lower Stepper, clone it.
            //llvm::outs() << *looplet << "\n";
            Operation* clonedStepper = rewriter.clone(*looplet);  
            stepperLooplets.push_back(cast<finch::StepperOp>(clonedStepper));
            accessOps.push_back(cast<finch::AccessOp>(accessOp));
          }
        }
      }
    }
    //llvm::outs() << "(0')\n";
    //llvm::outs() << *(forOp->getBlock()->getParentOp()->getBlock()->getParentOp()) << "\n";

    if (stepperLooplets.empty()) {
      return failure();
    }

    // Main Stepper Rewrite        
    Value loopLowerBound = forOp.getLowerBound();
    Value loopUpperBound = forOp.getUpperBound();

    //llvm::outs() << "(1)\n";
    // Call Seek
    SmallVector<Value, 4> seekPositions;
    for (auto& stepperLooplet : stepperLooplets) {
      Block &seekBlock = stepperLooplet.getRegion(0).front();

      Operation* seekReturn = seekBlock.getTerminator();
      rewriter.inlineBlockBefore(&seekBlock, forOp, ValueRange(loopLowerBound));
      Value seekPosition = seekReturn->getOperand(0);
      seekPositions.push_back(seekPosition);
      rewriter.eraseOp(seekReturn); 
    }
 
    // create while Op
    seekPositions.push_back(loopLowerBound);
    unsigned numIterArgs = seekPositions.size();
    ValueRange iterArgs(seekPositions);
    scf::WhileOp whileOp = rewriter.create<scf::WhileOp>(
        loc, iterArgs.getTypes(), iterArgs);


    // fill condition
    SmallVector<Location, 4> locations(numIterArgs, loc);
    Block *before = rewriter.createBlock(&whileOp.getBefore(), {},
                                         iterArgs.getTypes(), locations);
    rewriter.setInsertionPointToEnd(before);
    Value cond = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                before->getArgument(numIterArgs-1), 
                                                loopUpperBound);
    rewriter.create<scf::ConditionOp>(loc, cond, before->getArguments());


    // after region of while op 
    Block *after = rewriter.createBlock(&whileOp.getAfter(), {},
                                        iterArgs.getTypes(), locations);

    rewriter.setInsertionPointToEnd(after);
    rewriter.moveOpBefore(forOp, after, after->end());
          

    //llvm::outs() << "(2)\n";
    // call stop then intersection
    rewriter.setInsertionPoint(forOp);
    SmallVector<Value, 4> stopCoords;
    Value intersectUpperBound = loopUpperBound;
    for (unsigned i = 0; i < stepperLooplets.size(); i++) {
      auto stepperLooplet = stepperLooplets[i];
      Block &stopBlock = stepperLooplet.getRegion(1).front();
      
      // IDK why but this order is important.
      // getTerminator -> inlineBlockBefore -> getOperand -> eraseOp
      Operation* stopReturn = stopBlock.getTerminator();
      rewriter.inlineBlockBefore(&stopBlock, forOp, after->getArgument(i));
      Value stopCoord = stopReturn->getOperand(0);
      rewriter.eraseOp(stopReturn);

      //llvm::outs() << "(2-2)\n";
      intersectUpperBound = rewriter.create<arith::MinUIOp>(
          loc, intersectUpperBound, stopCoord);
      stopCoords.push_back(stopCoord);
    }
    //llvm::outs() << *(forOp->getBlock()->getParentOp()->getBlock()->getParentOp()) << "\n";
    //llvm::outs() << numIterArgs << "\n";
    forOp.setLowerBound(after->getArgument(numIterArgs-1));
    forOp.setUpperBound(intersectUpperBound); 



    //llvm::outs() << "(3)\n";
    //llvm::outs() << *(forOp->getBlock()->getParentOp()->getBlock()->getParentOp()) << "\n";

    // call body and replace access 
    for (unsigned i = 0; i < stepperLooplets.size(); i++) {
      auto stepperLooplet = stepperLooplets[i];      
      Block &bodyBlock = stepperLooplet.getRegion(2).front();
      Operation* bodyReturn = bodyBlock.getTerminator();
      Value bodyLooplet = bodyReturn->getOperand(0);
      rewriter.inlineBlockBefore(&bodyBlock, forOp, after->getArgument(i));
     
      //Operation* loopletOp = stepperLooplet;
      //Operation* accessOp = mapper.lookupOrDefault(loopletOp);
      //accessOp->setOperand(0, bodyLooplet);
      accessOps[i].setOperand(0, bodyLooplet);
      rewriter.eraseOp(bodyReturn);
    }
  
    //// current Upper Bound become next iteration's lower bound 
    rewriter.setInsertionPointToEnd(after);
    Value nextCoord = intersectUpperBound;

    //llvm::outs() << "(4)\n";
    //// call next
    SmallVector<Value,4> nextPositions;
    Type indexType = rewriter.getIndexType();
    for (unsigned i = 0; i < stepperLooplets.size(); i++) {
      auto stepperLooplet = stepperLooplets[i];            
      auto currPos = after->getArgument(i);
      auto stopCoord = stopCoords[i];

      Block &nextBlock = stepperLooplet.getRegion(3).front();
      Operation* nextReturn = nextBlock.getTerminator();
      Value nextPos = nextReturn->getOperand(0);
      
      rewriter.setInsertionPointToEnd(after);
      Value eq = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, stopCoord, intersectUpperBound);

      scf::IfOp ifOp = rewriter.create<scf::IfOp>(loc, indexType, eq, true);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      scf::YieldOp thenYieldOp = rewriter.create<scf::YieldOp>(loc, nextPos);
      rewriter.inlineBlockBefore(&nextBlock, thenYieldOp, currPos);
      
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      scf::YieldOp elseYieldOp = rewriter.create<scf::YieldOp>(loc, currPos);
      
      nextPositions.push_back(ifOp.getResult(0));
      rewriter.eraseOp(nextReturn); 
    }
    nextPositions.push_back(nextCoord);
    rewriter.setInsertionPointToEnd(after);
    rewriter.create<scf::YieldOp>(loc, ValueRange(nextPositions));

    // Todo:Build a chain
    // %0 = tensor.empty()
    // %1 = scf.for $i = $b0 to %b1 step %c1 iter_args(%v = %0) //forOp
    // return %1
    //
    // vvv
    //
    // %0 = tensor.empty()
    // %res:4 = scf.while iter_args(%pos1=%pos1_, %pos2=%pos2_, %idx=%idx_, %tensor=%0) {
    //    %2 = scf.for $i = $b0 to %b1 step %c1 iter_args(%v = %tensor) //newForOp2
    // }
    // return %res#3



    return success();
  }
};

class FinchLoopletStepper
    : public impl::FinchLoopletStepperBase<FinchLoopletStepper> {
public:
  using impl::FinchLoopletStepperBase<
      FinchLoopletStepper>::FinchLoopletStepperBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FinchLoopletStepperRewriter>(&getContext()); 
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};


} // namespace
} // namespace mlir::finch
