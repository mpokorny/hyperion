option(gridder_USE_YAML "Use YAML" ON)

if (gridder_USE_YAML)
  find_package(yaml-cpp REQUIRED)
endif()

if (hyperion_USE_HDF5 AND hyperion_USE_CASACORE)
  if (MAX_DIM GREATER_EQUAL "7" AND gridder_USE_YAML)
    add_executable(gridder gridder.cc args.h args.cc)
    set_host_target_properties(gridder)
    target_link_libraries(gridder hyperion_synthesis hyperion yaml-cpp)
    install(TARGETS gridder
      RUNTIME DESTINATION bin)
  endif ()
endif()
