//===- FinchBufferPasses.cpp - Finch Buffer passes -----------------*- C++ -*-===//
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "Finch/FinchPasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::finch {
#define GEN_PASS_DEF_FINCHBUFFERCURRENTSIZE
#include "Finch/FinchPasses.h.inc"

namespace {

class FinchBufferCurrentSizeRewriter : public OpRewritePattern<finch::CurrentSizeOp> {
public:
  using OpRewritePattern<finch::CurrentSizeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(finch::CurrentSizeOp currentSizeOp,
                                PatternRewriter &rewriter) const final {
    OpBuilder builder(currentSizeOp);
    Location loc = currentSizeOp.getLoc();

    //  What we see:
    //  %0 = memref.alloc() : memref<3xi32>
    //  %buffer = finch.make_buffer(%0): !finch.buffer
    //  %size = finch.current_size %buffer: (!finch.buffer) -> index

    //  What we want:
    //  %0 = memref.alloc() : memref<3xi32>
    //  %dim0 = arith.constant 0 
    //  %size = memref.dim %0 %dim0: memref<3xi32>

    Value finch_buffer = currentSizeOp.getOperand();
    Operation* makeBufferOp_ptr = finch_buffer.getDefiningOp();
    if (!isa<finch::MakeBufferOp>(makeBufferOp_ptr))
      return failure(); 
    
    auto makeBufferOp = dyn_cast<finch::MakeBufferOp>(makeBufferOp_ptr);
    Value memref_buffer = makeBufferOp.getOperand(); //%0

    auto constantZeroIndexOp= 
      rewriter.create<arith::ConstantIndexOp>(loc, 0);

    auto newOp = 
      rewriter.create<memref::DimOp>(
        loc, 
        memref_buffer, 
        constantZeroIndexOp.getResult()
      );   

    rewriter.replaceOp(currentSizeOp, newOp);
   
    return success();
  }
};


class FinchBufferCurrentSize
    : public impl::FinchBufferCurrentSizeBase<FinchBufferCurrentSize> {
public:
  using impl::FinchBufferCurrentSizeBase<
      FinchBufferCurrentSize>::FinchBufferCurrentSizeBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FinchBufferCurrentSizeRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};


} // namespace
} // namespace mlir::finch
