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
    Lowers finch.access of finch.stepper inside scf.for
    
    `finch.stepper seek(crd):pos, stop(pos):crd, body(pos):looplet, next(pos):pos` can be thought as a function
    p = initpos
    f(x) = body(p) (if x < stop(p))
           body(next(p)) (if stop(p) <= x < stop(next(p)))
           body(next(next(p))) (if stop(next(p)) <= x < stop(next(next(p))))
           ...

    if we iterate f(x) over [st, en),
    p = seek(st)
    f(x) = body(p) (if x in [st, min(en, stop(p))) )
           body(next(p)) (if x in [min(en, stop(p)), min(en, stop(next(p)))))
           ...

    A pseudocode of lowering a single stepper looks like :

    ```mlir

    %0 = finch.stepper
      seek = {
        ^bb(%crd:index):
          ...
      }
      stop = {
        ^bb(%pos:index):
          ...
      }
      body = {
        ^bb(%pos:index):
          ...
      }
      next = {
        ^bb(%pos:index):
          ...
      }
    scf.for %idx = %st to %en step %c1 {
        %1 = finch.access %0, %idx : f32
    }
    ```
    to 

    ```mlir 
    %initpos = stepper.seek(%st)
    %1:2 = scf.while (%pos=%initpos, %idx=%st) : (index,index) -> (index,index) {
      %cmp = arith.cmpi ult, %idx, %en : index
      scf.condition(%cmp) %pos, %idx : index, index
    } do {
      ^bb(%pos: index, %idx: index):
        %stop = stepper.stop(%pos)
        %body = stepper.body(%pos)
        %end = arith.minui %2, %en : index
        scf.for %idx2 = %idx to %end step %c1 {
          %1 = finch.access %body, %idx2 : f32
        }
        
        %nextpos = stepper.next(%pos)
        scf.yield %nextpos, %end
    }
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
