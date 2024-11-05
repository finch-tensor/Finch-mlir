add_llvm_library(FinchPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        finch-plugin.cpp

        DEPENDS
        MLIRFinch
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRFinch
        )
target_include_directories(
        FinchPlugin
        PRIVATE
        "${FINCH_BINARY_DIR}/include"
)
