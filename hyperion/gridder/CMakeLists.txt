if (hyperion_USE_HDF5 AND hyperion_USE_CASACORE
    AND hyperion_USE_YAML AND MAX_DIM GREATER_EQUAL "7")
  add_executable(gridder gridder.cc args.h args.cc)
  set_host_target_properties(gridder)
  target_link_libraries(gridder hyperion yaml-cpp)
  install(TARGETS gridder
    RUNTIME DESTINATION bin)
endif()
