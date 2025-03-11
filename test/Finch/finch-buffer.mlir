// RUN: finch-opt %s | finch-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = memref.alloc() : memref<3xi32>
       
        %1 = finch.current_size %0 : memref<3xi32> -> index
 
        return
    }
}
