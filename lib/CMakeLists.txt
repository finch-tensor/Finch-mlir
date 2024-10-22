add_subdirectory(CAPI)
add_subdirectory(Finch)

install(IMPORTED_RUNTIME_ARTIFACTS mlir_c_runner_utils
  LIBRARY DESTINATION lib
)
install(IMPORTED_RUNTIME_ARTIFACTS mlir_float16_utils
  LIBRARY DESTINATION lib
)
