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
    to be filled 
  }];
}

def FinchLoopletRun: Pass<"finch-looplet-run"> {
  let summary = "Compiler Pass for Run Looplets";
  let description = [{
    Lowers finch.access of finch.run inside scf.for.
    Specifically, it replaces finch.access with the argument of finch.run.

    ```
    %0 = finch.run %val : (f32) -> (!finch.looplet)
    scf.for %i = %lb to %ub ... {
       %1 = finch.access %0 %i : f32
    }
    ```

    to

    ```
    scf.for %i = %lb to %ub ... {
       %1 = %val 
    }
    ```
  }];
}


def FinchLoopletSequence: Pass<"finch-looplet-sequence"> {
  let summary = "Compiler Pass for Sequence Looplets";
  let description = [{
    Lowers finch.access of finch.sequence inside scf.for
    
    `finch.sequence crd, body1, body2` can be thought as a function

    f(x) = body1 (if x < crd)
           body2 (if x >= crd)

    if we iterate f(x) over [st, en),
    f(x) = body1 (if x in [-inf,crd) ∩ [st, en))
           body2 (if x in [crd,+inf) ∩ [st, en))

    ```mlir
    %0 = finch.run %cst_0 : (f32) -> (!finch.looplet)
    %1 = finch.run %cst_1 : (f32) -> (!finch.looplet)
    %c3 = arith.constant 3 : index
    %2 = finch.sequence %c3, %0, %1 : (index, !finch.looplet, !finch.looplet) -> (!finch.looplet)
    scf.for %idx = %st to %en step %c1 {
        %3 = finch.access %2, %idx : f32
    }
    ```
    to

    ```mlir 
    %0 = finch.run %cst_0 : (f32) -> (!finch.looplet)
    %1 = finch.run %cst_1 : (f32) -> (!finch.looplet)
    %c3 = arith.constant 3 : index
    %2 = arith.minui %en, %c3 : index
    %3 = arith.maxui %st, %c3 : index
    scf.for %idx = %st to %2 step %c1 {
      %5 = finch.access %0, %idx : f32
    }
    scf.for %arg2 = %3 to %en step %c1 {
      %5 = finch.access %1, %idx : f32
    }

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
//def FinchBufferLoad: Pass<"finch-buffer-load-pass"> {
//  let summary = "Compiler Pass for Buffer Loads";
//  let description = [{
//      ```mlir
//      %0 = memref.alloc() : memref<3xi32>
//      %buffer = finch.make_buffer(%0): !finch.buffer
//      %value = finch.buffer_load %buffer, %index: (!finch.buffer, index) -> i32
//      ```

//      What we want:
//      ```mlir
//      %0 = memref.alloc() : memref<3xi32>
//      %value = memref.load %0, %index: memref<3xi32>
//      ```

//  }];
//}

#endif // FINCH_PASS
