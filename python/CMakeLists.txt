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
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(FinchPythonCAPI
  INSTALL_COMPONENT FinchPythonModules
  INSTALL_DESTINATION python_packages/finch/mlir_finch/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/finch/mlir_finch/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    FinchPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(FinchPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/finch/mlir_finch"
  INSTALL_PREFIX "python_packages/finch/mlir_finch"
  DECLARED_SOURCES
    FinchPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonExtension.ExecutionEngine
    MLIRPythonSources.ExecutionEngine
  COMMON_CAPI_LINK_LIBS
    FinchPythonCAPI
  )
