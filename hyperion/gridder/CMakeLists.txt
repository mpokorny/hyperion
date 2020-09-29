if (hyperion_USE_HDF5
    AND hyperion_USE_CASACORE
    AND hyperion_USE_YAML
    AND ${hyperion_MAX_DIM} GREATER_EQUAL 7)
  add_executable(gridder gridder.cc args.h args.cc)
  target_link_libraries(gridder hyperion yaml-cpp)
  install(TARGETS gridder)
endif()
