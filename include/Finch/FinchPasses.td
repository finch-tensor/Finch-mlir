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

#endif // FINCH_PASS
