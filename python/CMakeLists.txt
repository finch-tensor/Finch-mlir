include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_finch`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_finch.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(FinchPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT FinchPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_finch"
  TD_FILE dialects/FinchOps.td
  SOURCES
    dialects/finch.py
  DIALECT_NAME finch)

declare_mlir_python_extension(FinchPythonSources.Extension
  MODULE_NAME _finchDialects
  ADD_TO_PARENT FinchPythonSources
  SOURCES
    FinchExtension.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    FinchCAPI
    # MLIRCAPIConversion
    # MLIRCAPITransforms
    # MLIRCAPIRegisterEverything
)

# EMBED_CAPI_LINK_LIBS
#   MLIRCAPIDebug
#   MLIRCAPIIR
#   MLIRCAPIInterfaces
# declare_mlir_python_extension(FinchPythonSources.RegisterWhatWeNeed
#   MODULE_NAME _finchRegisterWhatWeNeed
#   ADD_TO_PARENT FinchPythonSources
#   SOURCES
#     RegisterEverything.cpp
#   PRIVATE_LINK_LIBS
#     LLVMSupport
#   EMBED_CAPI_LINK_LIBS
#     MLIRCAPIConversion
#     MLIRCAPITransforms
#     MLIRCAPIRegisterEverything
# )

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(FPCAPI
  INSTALL_COMPONENT FinchPythonModules
  INSTALL_DESTINATION mlir_finch/_ms
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/mlir_finch/_ms"
  RELATIVE_INSTALL_ROOT "../.."
  DECLARED_SOURCES
    FinchPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.ExecutionEngine
    #MLIRPythonExtension.RegisterEverything
    #MLIRPythonSources.ExecutionEngine
    #MLIRPythonSources.Dialects
    MLIRPythonSources #.Core
    #LLVMSupport
  #EMBED_CAPI_LINK_LIBS
    #MLIRCAPIConversion
    #MLIRCAPITransforms
    #MLIRCAPIRegisterEverything
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(FinchPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/mlir_finch"
  INSTALL_PREFIX "mlir_finch"
  DECLARED_SOURCES
    FinchPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.Core
    MLIRPythonExtension.ExecutionEngine
    MLIRPythonSources.ExecutionEngine
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    FPCAPI
    MLIRCAPIRegisterEverything
  )
