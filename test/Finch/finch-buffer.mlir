// RUN: finch-opt %s | finch-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() -> index {
        %0 = memref.alloc() : memref<3xi32>
        %buffer = finch.make_buffer(%0):  memref<3xi32> -> !finch.buffer
       
        %1 = finch.current_size %buffer : !finch.buffer -> index


        %c5 = arith.constant 5 : index
        %bufferBig = finch.resize_if_smaller(%buffer, %c5): (!finch.buffer, index) -> !finch.buffer



        %start = arith.constant 0 : index
        %end = arith.constant 3 : index
        %value = arith.constant 1 : i32
        %bufferFilled = finch.fill_range(%buffer, %value, %start, %end): (!finch.buffer, i32, index, index) -> !finch.buffer


        %index = arith.constant 0 : index
        %2 = finch.buffer_load %bufferFilled, %index: (!finch.buffer, index) -> i32

        %c2 = arith.constant 2 : i32
        %idx = arith.constant 1 : index
        finch.buffer_store %c2, %buffer, %idx: (i32, !finch.buffer, index)

        %3 = finch.buffer_load %buffer, %idx: (!finch.buffer, index) -> i32

        %buffer2 = finch.make_buffer(%buffer):  !finch.buffer -> !finch.buffer
        return %1 : index
    }
}
