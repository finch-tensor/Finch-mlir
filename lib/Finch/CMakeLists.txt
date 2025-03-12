add_mlir_dialect_library(MLIRFinch
        FinchTypes.cpp
        FinchDialect.cpp
        FinchOps.cpp
        FinchRunPass.cpp
        FinchSequencePass.cpp
        FinchPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Finch

        DEPENDS
        MLIRFinchOpsIncGen
        MLIRFinchPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
