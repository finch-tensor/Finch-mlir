//===- FinchTypes.td - Finch dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FINCH_TYPES
#define FINCH_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "Finch/FinchDialect.td"

//===----------------------------------------------------------------------===//
// Finch type definitions
//===----------------------------------------------------------------------===//

class Finch_Type<string name, string typeMnemonic, list<Trait> traits = [],
                      string baseCppClass="::mlir::Type">
    : TypeDef<Finch_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def Finch_CustomType : Finch_Type<"Custom", "custom"> {
    let summary = "Finch custom type";
    let description = "Custom type in finch dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}



def Finch_LoopletType : TypeDef<Finch_Dialect, "Looplet", []> {
    let summary = "Finch Looplets type";
    let description = "Looplets type in finch dialect";
    let mnemonic = "looplet";
    //let parameters = (ins "int":$lb, "int":$ub);
    //let assemblyFormat = "`<` $lb `,` $ub `>`";
}

def AnyNumber : AnyTypeOf<[AnyInteger, AnyFloat]>;
def AnyNumberOrIndex : AnyTypeOf<[AnyInteger, AnyFloat, Index]>;
def Looplet : AnyTypeOf<[Finch_LoopletType]>;
def LoopletOrNumber : AnyTypeOf<[Finch_LoopletType, AnyNumber]>;



def Finch_BufferType: Finch_Type<"Buffer", "buffer"> {
    let summary = "Finch buffer type";
    let description = "Buffer type in finch dialect";
    let mnemonic = "buffer";
    //let parameters = (ins AnyMemRef:$memref);
    //let assemblyFormat = "`(` $memref `)`"; 
}

#endif // FINCH_TYPES
