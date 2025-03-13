//===- FinchPasses.cpp - Finch passes -----------------*- C++ -*-===//
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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "Finch/FinchPasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::finch {
#define GEN_PASS_DEF_FINCHSIMPLIFIER
#define GEN_PASS_DEF_FINCHINSTANTIATE
#define GEN_PASS_DEF_FINCHLOOPLETSEQUENCE
#define GEN_PASS_DEF_FINCHLOOPLETLOOKUP
#include "Finch/FinchPasses.h.inc"

namespace {
class FinchNextLevelRewriter : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    auto indVar = forOp.getInductionVar();
    
    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();

    for (auto& accessOp : *forOp.getBody()) {
      if (isa<mlir::finch::AccessOp>(accessOp)) {
        auto accessVar = accessOp.getOperand(1);
        if (accessVar == indVar) {
          auto nextLevelOp = accessOp.getOperand(0).getDefiningOp<finch::NextLevelOp>();
          if (!nextLevelOp) {
            continue;
          }
          
          Value nextLevelPosition = nextLevelOp.getOperand(); 
          rewriter.replaceOp(&accessOp, nextLevelPosition);

          return success();
        }
      }
    }
    
    return failure();
  }
};

class FinchInstantiateRewriter : public OpRewritePattern<finch::GetLevelOp> {
public:
  using OpRewritePattern<finch::GetLevelOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(finch::GetLevelOp op,
                                PatternRewriter &rewriter) const final {
    Value levelDef = op.getOperand(0);
    Value levelPos = op.getOperand(1);
  
    Operation* lvlDefOp = levelDef.getDefiningOp<finch::DefineLevelOp>();
    if (!lvlDefOp) {
      return failure();
    }

    Operation* lvlPosOp = levelPos.getDefiningOp<finch::AccessOp>();
    if (lvlPosOp) {
      // position is coming from finch::AccessOp,
      // which means looplet passes are not done.
      return failure();
    }

    Operation* clonedLvlDefOp = rewriter.clone(*lvlDefOp);  

    Block &defBlock = clonedLvlDefOp->getRegion(0).front();
    Operation* retLooplet = defBlock.getTerminator();
    Value looplet = retLooplet->getOperand(0);
    
    rewriter.inlineBlockBefore(&defBlock, op, ValueRange(levelPos));
    rewriter.eraseOp(retLooplet);
    rewriter.replaceOp(op, looplet);
    
    return success();
  }
};

class FinchSemiringRewriter : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const final {
    for (auto& bodyOp : *op.getBody()) {
      if (isa<arith::MulFOp>(bodyOp)) {
        auto constOp = bodyOp.getOperand(1).getDefiningOp();
        if (matchPattern(constOp, m_AnyZeroFloat())) {
          rewriter.replaceOp(&bodyOp, constOp);
          return success();
        }
      } else if (isa<arith::AddFOp>(bodyOp)) {
        auto constOp = bodyOp.getOperand(1).getDefiningOp();
        if (matchPattern(constOp, m_AnyZeroFloat())) {
          rewriter.replaceOp(&bodyOp, bodyOp.getOperand(0));
          return success();
        }
      }
    }

    return failure();
  }
};

class FinchLoopInvariantCodeMotion : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const {
    size_t numMoved = moveLoopInvariantCode(op);
    return numMoved > 0 ? success() : failure();
  }
};

class FinchAssignRewriter : public OpRewritePattern<finch::AssignOp> {
public:
  using OpRewritePattern<finch::AssignOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(finch::AssignOp op,
                                PatternRewriter &rewriter) const {
    Value out = op.getOperand(0); 
    Value in = op.getOperand(1); 
 
    // finch.assign %out = %in
    if (in == out) {
      rewriter.eraseOp(op);
      return success();
    }
   
    Operation* loadOp = out.getDefiningOp();
    if (isa<finch::AccessOp>(loadOp)) {
      return failure();
    }

    assert(isa<memref::LoadOp>(loadOp) && "Currently Assign can only convert memref.load after elementlevel");
    assert(loadOp->getNumOperands() == 2 && "Currently only accept non-scalar tensor");

   // Value "in" is dependent to "out"
   // e.g.,
   // %in = arith.addf %out, %1
   // finch.assign %out = %in
    bool isReduction = false;
    for (Operation *user : loadOp->getUsers()) {
        if (in.getDefiningOp() == user) { 
          isReduction = true;
          break;
        }
    }

    auto sourceMemref = loadOp->getOperand(0);
    auto sourcePos = loadOp->getOperand(1);
    auto storeOp = rewriter.replaceOpWithNewOp<memref::StoreOp>(
                  op, in, sourceMemref, sourcePos);


    // seriously consider replaceing this into finch.assign %out += %in
    if (isReduction) {
      rewriter.setInsertionPointToStart(storeOp->getBlock());
      Operation* newLoadOp = rewriter.clone(*loadOp);
      rewriter.replaceOpUsesWithinBlock(loadOp, newLoadOp->getResult(0), storeOp->getBlock());
    }

    return success();
  }
};


class FinchMemrefStoreLoadRewriter : public OpRewritePattern<memref::StoreOp> {
public:
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp op,
                                PatternRewriter &rewriter) const final {
    auto storeValue = op.getOperand(0);
    auto storeMemref = op.getOperand(1);
    
    auto loadOp = storeValue.getDefiningOp<memref::LoadOp>();
    if (!loadOp) {
      return failure();
    }
    auto loadMemref = loadOp.getOperand(0);

    bool isMemrefSame = storeMemref == loadMemref; 
    bool isIndexSame = true;

    // variadic index
    if (op.getNumOperands() > 2) {
      unsigned storeNumIndex = op.getNumOperands() - 2;
      unsigned loadNumIndex = loadOp.getNumOperands() - 1;
    
      if (storeNumIndex != loadNumIndex) {
        isIndexSame = false;
      } else {
        for (unsigned i=0; i<storeNumIndex; i++) {
          auto storeIndex = op.getOperand(2+i);
          auto loadIndex = loadOp.getOperand(1+i);
          isIndexSame = isIndexSame && (storeIndex == loadIndex); 
        }
      }
    }

    if (isMemrefSame && isIndexSame) {
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};


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
      
      // Intersection
      Value loopLb = forOp.getLowerBound();
      Value loopUb = forOp.getUpperBound();

      // Finch.jl used both closed endpoints,
      // but Finch.mlir uses [st,en) for now.
      // This is for aligning syntax with scf.for
      // scf.for uses [st,en).
      // TODO: We need to make finch.for
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
      
      // Main intersect
      Value newFirstLoopUb = rewriter.create<arith::MinUIOp>(
          loc, loopUb, firstBodyUb);
      Value newSecondLoopLb = rewriter.create<arith::MaxUIOp>(
          loc, loopLb, secondBodyLb);
      cast<scf::ForOp>(newForOp1).setUpperBound(newFirstLoopUb);
      cast<scf::ForOp>(newForOp2).setLowerBound(newSecondLoopLb);


      // Build a chain on scf.for
      // %0 = tensor.empty()
      // %1 = scf.for $i = $b0 to %b1 step %c1 iter_args(%v = %0) //forOp
      // %2 = scf.for $i = $b0 to %b2 step %c1 iter_args(%v = %0) //newForOp1
      // %3 = scf.for $i = $b2 to %b1 step %c1 iter_args(%v = %0) //newForOp2
      // return %1
      //
      // vvv
      //
      // %0 = tensor.empty()
      // %1 = scf.for $i = $b0 to %b2 step %c1 iter_args(%v = %0) //newForOp1  
      // %2 = scf.for $i = $b2 to %b1 step %c1 iter_args(%v = %1) //newForOp2
      // return %2

      // First three are lowerbound, upperbound, and step
      int numIterArgs = newForOp2->getNumOperands() - 3;
      if (numIterArgs > 0) {
        rewriter.replaceAllUsesWith(forOp->getResults(), newForOp2->getResults());
        newForOp2->setOperands(3, numIterArgs, newForOp1->getResults());
      }

      rewriter.eraseOp(forOp);
      
      return success();
    }
    return failure();
  }
};


class FinchLoopletLookupRewriter : public OpRewritePattern<scf::ForOp> {
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
      
      auto lookupLooplet = dyn_cast<finch::LookupOp>(
          accessOp.getOperand(0).getDefiningOp());
      
      if (!lookupLooplet) 
        continue;
            
      Operation* lookupLooplet_ = 
        rewriter.clone(*lookupLooplet);  
      Block &bodyBlock = 
        lookupLooplet_->getRegion(0).front();
      Operation* bodyReturn = bodyBlock.getTerminator();
      rewriter.inlineBlockBefore(
          &bodyBlock, 
          &accessOp, 
          forOp.getInductionVar());
      Value bodyLooplet = bodyReturn->getOperand(0);
      
      accessOp.setOperand(0, bodyLooplet);
      rewriter.eraseOp(bodyReturn);
      return success();
    }

    return failure();
  }
};



class FinchInstantiate
    : public impl::FinchInstantiateBase<FinchInstantiate> {
public:
  using impl::FinchInstantiateBase<
      FinchInstantiate>::FinchInstantiateBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FinchNextLevelRewriter>(&getContext());
    patterns.add<FinchInstantiateRewriter>(&getContext());
    patterns.add<FinchLoopInvariantCodeMotion>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};


class FinchSimplifier
    : public impl::FinchSimplifierBase<FinchSimplifier> {
public:
  using impl::FinchSimplifierBase<
      FinchSimplifier>::FinchSimplifierBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FinchMemrefStoreLoadRewriter>(&getContext());
    patterns.add<FinchSemiringRewriter>(&getContext());
    patterns.add<FinchLoopInvariantCodeMotion>(&getContext());
    patterns.add<FinchAssignRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
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

class FinchLoopletLookup
    : public impl::FinchLoopletLookupBase<FinchLoopletLookup> {
public:
  using impl::FinchLoopletLookupBase<
      FinchLoopletLookup>::FinchLoopletLookupBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FinchLoopletLookupRewriter>(&getContext()); 
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};


} // namespace
} // namespace mlir::finch
