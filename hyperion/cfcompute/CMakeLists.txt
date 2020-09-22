if (hyperion_USE_CASACORE
    AND hyperion_USE_KOKKOS
    AND MAX_DIM GREATER_EQUAL "8")
  add_executable(cfcompute cfcompute.cc)
  set_host_target_properties(cfcompute)
  target_link_libraries(cfcompute hyperion)
  install(TARGETS cfcompute)
endif()
