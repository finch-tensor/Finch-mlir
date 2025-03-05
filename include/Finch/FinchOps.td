//===- FinchOps.td - Finch dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FINCH_OPS
#define FINCH_OPS

include "Finch/FinchDialect.td"
include "Finch/FinchTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ParallelCombiningOpInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def Finch_FooOp : Finch_Op<"foo", [Pure,SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `finch.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = finch.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}



def Finch_AccessOp : Finch_Op<"access", [Pure]> {
    let summary = "Finch Access op";
    let description = [{
        This operation accesses a Looplet using a loop index.
        It takes two input arguments: (1) a Looplet (of Looplet Type) and (2) a loop index (of Index Type).
        It then outputs an element of the Looplet at the corresponding input index.

        ```mlir
        %0 = arith.constant 3.3 : f32
        %1 = finch.run %0 : (f32) -> (!finch.looplet)

        scf.for %i = %b0 to %b1 step %1 {
            %2 = finch.access %1, %i : f32
        }
        ```
    }];

    let arguments = (ins Finch_LoopletType:$input, Index:$idx); 
    let results = (outs AnyType:$res);

    let assemblyFormat = [{
        $input `,` $idx attr-dict `:` type($res)
    }];
}

def Finch_NextLevelOp : Finch_Op<"nextlevel", [Pure]> {
    let summary = "Finch NextLevel Looplets";
    let description = [{
        ```mlir
        %3 = finch.nextlevel %pos : Index -> looplet
        ```
    }];

    let arguments = (ins Index:$pos); 
    let results = (outs Finch_LoopletType:$res);

    let assemblyFormat = [{
        $pos attr-dict `:` `(` type($pos) `)` `->` `(` type($res) `)`
    }];
}


def Finch_RunOp : Finch_Op<"run", [Pure]> {
    let summary = "Finch Run Looplets";
    let description = [{
        This operation defines the Run Looplet.
        It takes a single argument: an element of runs, which is of a numeric type.

        ```mlir
        %0 = arith.constant 4 : i32
        %1 = finch.run %0 : (i32) -> (!finch.looplet)
        ```
    }];

    let arguments = (ins AnyNumber:$val); 
    let results = (outs Finch_LoopletType:$res);

    let assemblyFormat = [{
         $val attr-dict `:` `(` type($val) `)` `->` `(` type($res) `)`
    }];
}


def Finch_SequenceOp : Finch_Op<"sequence", [Pure]> {
    let summary = "Finch Sequence Looplets";
    let description = [{
        This operation defines the Sequence Looplet.
        It takes three arguments : 
            (1) Splitting Coordinate (of numeric Type or index type)
            (2) FirstBody (of Looplet Type)
            (3) SecondBody (of Looplet Type)

        `finch.sequence crd, body1, body2` can be thought as a function

        f(x) = body1 (if x < crd)
               body2 (if x >= crd)

        ```mlir
        %0 = arith.constant 0 : f32
        %1 = arith.constant 1 : f32
        %2 = finch.run %0 : (f32) -> (!finch.looplet)
        %3 = finch.run %1 : (f32) -> (!finch.looplet)
        %4 = arith.constant 4 : index
        %5 = finch.sequence %4, %2, %3 : (index, !finch.looplet, !finch.looplet) -> !finch.looplet
        ```
    }];

    let arguments = (ins AnyNumberOrIndex:$split, 
                         Finch_LoopletType:$val1,
                         Finch_LoopletType:$val2); 
    let results = (outs Finch_LoopletType:$res);

    let assemblyFormat = [{
        $split `,` $val1 `,` $val2 attr-dict `:` `(` type($split) `,` type($val1) `,` type($val2) `)` `->` `(` type($res) `)`
    }];
}

def Finch_LookupOp : Finch_Op<"lookup",[Pure, NoTerminator]> {
    let summary = "Finch Lookup Looplets";
    let description = [{
        ```mlir
        %3 = finch.stepper %lb %ub : 
        ```
    }];

    //let arguments = () ;
    let regions = (region AnyRegion:$body);
    let results = (outs Finch_LoopletType:$res);

    let assemblyFormat = [{ 
        `\n`
        `body` `=` $body  attr-dict 
    }];
}

def Finch_StepperOp : Finch_Op<"stepper",[Pure, NoTerminator]> {
    let summary = "Finch Stepper Looplets";
    let description = [{
        ```mlir
        %3 = finch.stepper %lb %ub : 
        ```
    }];

    //let arguments = () ;
    let regions = (region AnyRegion:$seek,
                          AnyRegion:$stop,
                          AnyRegion:$body,
                          AnyRegion:$next);
    let results = (outs Finch_LoopletType:$res);

    let assemblyFormat = [{ 
        `\n`
        `seek` `=` $seek `\n` 
        `stop` `=` $stop `\n` 
        `body` `=` $body `\n` 
        `next` `=` $next attr-dict 
    }];
}

def Finch_ReturnOp : Finch_Op<"return", [Pure, Terminator, ReturnLike]> {
    let summary = "Finch Return Looplets";
    let description = [{
        ```mlir
        finch.return %9 : !finch.looplet 
        ```
    }];

    let arguments = (ins AnyType:$result);
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
    let assemblyFormat = [{
       $result attr-dict  `:` type($result)
    }];
}

def Finch_DefineLevelOp : Finch_Op<"definelevel", [Pure]> {
    let summary = "Finch Define Level op";
    let description = [{
        ```mlir
        %2 = finch.definelevel 
        ```
    }];

    let regions = (region AnyRegion:$def);
    let results = (outs Finch_LoopletType:$res);

    let assemblyFormat = [{ 
        $def attr-dict  
    }];
}


def Finch_GetLevelOp : Finch_Op<"getlevel", [Pure]> {
    let summary = "Finch getlevel op";
    let description = [{
        ```mlir
        %2 = finch.definelevel 
        ```
    }];

    let arguments = (ins Finch_LoopletType:$level,
                         Index:$pos); 
    let results = (outs Finch_LoopletType:$res);

    let assemblyFormat = [{ 
        $level `,` $pos attr-dict `:` `(` type($level) `,` type($pos) `)` `->` `(` type($res) `)`
    }];
}

def Finch_AssignOp : Finch_Op<"assign", [MemRefsNormalizable, AllTypesMatch<["in","out"]>]> {
    let summary = "Finch getlevel op";
    let description = [{
        ```mlir
        %2 = finch.assign 
        ```
    }];

    let arguments = (ins AnyType:$out,AnyType:$in ); 

    let assemblyFormat = [{ 
        $out `=` $in attr-dict `:` type($in)
    }];
}


#endif // FINCH_OPS
