add_mlir_dialect(FinchOps finch)
add_mlir_doc(FinchDialect FinchDialect Finch/ -gen-dialect-doc)
add_mlir_doc(FinchOps FinchOps Finch/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS FinchPasses.td)
mlir_tablegen(FinchPasses.h.inc --gen-pass-decls)
# mlir_tablegen(FinchOpsTypes.h.inc -gen-typedef-decls -typedefs-dialect=finch)
# mlir_tablegen(FinchOpsTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=finch)
add_public_tablegen_target(MLIRFinchPassesIncGen)

set(LLVM_TARGET_DEFINITIONS FinchTypes.td)
mlir_tablegen(FinchTypes.h.inc -gen-typedef-decls -typedefs-dialect=finch)
mlir_tablegen(FinchTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=finch)
add_public_tablegen_target(MLIRFinchTypesIncGen)
