if (hyperion_USE_CASACORE
    AND hyperion_USE_KOKKOS
    AND ${hyperion_MAX_DIM} GREATER_EQUAL 8)
  add_executable(cfcompute cfcompute.cc)
  target_link_libraries(cfcompute hyperion)
  install(TARGETS cfcompute)
endif()
