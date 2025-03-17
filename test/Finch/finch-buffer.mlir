// RUN: finch-opt %s | finch-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = memref.alloc() : memref<3xi32>
        %buffer = finch.make_buffer(%0):  memref<3xi32> -> !finch.buffer
       
        %1 = finch.current_size %buffer : !finch.buffer -> index
 
        return
    }
}
