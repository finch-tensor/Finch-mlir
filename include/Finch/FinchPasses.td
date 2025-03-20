//===- FinchPsss.td - Finch dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FINCH_PASS
#define FINCH_PASS

include "mlir/Pass/PassBase.td"

def FinchInstantiate: Pass<"finch-instantiate"> {
  let summary = "Instantiate Looplet";
  let description = [{
    Compiler Pass for handling all looplets
    ```
    to be filled.

    ```
  }];
}


def FinchSimplifier: Pass<"finch-simplifier"> {
  let summary = "finch simplifier";
  let description = [{
    Switches the name of a FuncOp named `bar` to `foo` and folds.
    ```
    func.func @bar() {
      return
    }
    // Gets transformed to:
    func.func @foo() {
      return
    }
    ```
  }];
}

def FinchLoopletRun: Pass<"finch-looplet-run"> {
  let summary = "Compiler Pass for Run Looplets";
  let description = [{
    Switches the name of a FuncOp named `bar` to `foo` and folds.
    ```
    affine.for %i = %lb to %ub ... {
       %0 = finch.run %lb2 %ub2 %val
       %1 = finch.access %0 %i
       ... 
    }

    to

    %lb_1 = arith.min %lb %lb2
   
    %newlb = arith.max %lb %lb2
    %newub = arith.min %ub %ub2
    
    %ub_2 = arith.max %ub %ub2

    affine.for %i = %lb_1 to %newlb ... {
      replace %1 to %undefined
    }

    affine.for %i = %newlb to %newub ... {
      replace %1 to %val
    }

    affine.for %i = %newlb to %newub ... {
      replace %1 to %val
    }

    ```
  }];
}


def FinchLoopletSequence: Pass<"finch-looplet-sequence"> {
  let summary = "Compiler Pass for Sequence Looplets";
  let description = [{
    Compiler Pass for handling sequence looplets
    ```
    to be filled.

    ```
  }];
}

def FinchLoopletStepper: Pass<"finch-looplet-stepper"> {
  let summary = "Compiler Pass for Stepper Looplets";
  let description = [{
    Compiler Pass for handling stepper looplets
    ```
    to be filled.

    ```
  }];
}

def FinchLoopletLookup: Pass<"finch-looplet-lookup"> {
  let summary = "Compiler Pass for Lookup Looplets";
  let description = [{
    Compiler Pass for handling lookup looplets
    ```
    to be filled.

    ```
  }];
}

def FinchLoopletPass: Pass<"finch-looplet-pass"> {
  let summary = "Compiler Pass for Looplets";
  let description = [{
    Compiler Pass for handling all looplets
    ```
    to be filled.

    ```
  }];
}

def FinchBufferLoadPass: Pass<"finch-buffer-load-pass"> {
  let summary = "Compiler Pass for Buffer Loads";
  let description = [{
      ```mlir
      %0 = memref.alloc() : memref<3xi32>
      %buffer = finch.make_buffer(%0): !finch.buffer
      %value = finch.buffer_load %buffer, %index: (!finch.buffer, index) -> i32
      ```

      What we want:
      ```mlir
      %0 = memref.alloc() : memref<3xi32>
      %value = memref.load %0, %index: memref<3xi32>
      ```

  }];
}


#endif // FINCH_PASS
