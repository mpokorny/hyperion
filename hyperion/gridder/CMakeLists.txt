if (USE_HDF5 AND USE_CASACORE)
  if (MAX_DIM GREATER_EQUAL "7" AND USE_YAML)
    add_executable(gridder gridder.cc args.h args.cc)
    add_dependencies(gridder hyperion)
    set_target_properties(gridder PROPERTIES
      CXX_STANDARD 17
      COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
    target_link_libraries(gridder hyperion yaml-cpp)
    target_include_directories(gridder
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../legion/runtime
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..
      PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../legion/runtime
      PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../..
      PUBLIC ${YAML_CPP_INCLUDE_DIR}
      PUBLIC ${CASACORE_INCLUDE_DIR}
      PUBLIC ${HDF5_INCLUDE_DIRS})
    install(TARGETS gridder
      RUNTIME DESTINATION bin)
  endif ()
endif()
