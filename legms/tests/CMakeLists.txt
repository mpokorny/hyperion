include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${Legion_INSTALL_DIR}/include)
if (legms_USE_CASACORE)
  include_directories(${casacore_INCLUDE_DIR})
endif(legms_USE_CASACORE)
if (legms_USE_HDF5)
  include_directories(${HDF5_INCLUDE_DIRS})
  link_libraries(${HDF5_LIBRARIES})
endif (legms_USE_HDF5)
link_libraries(legms_testing)

add_executable(utIndexColumnTask utIndexColumnTask.cc)
set_target_properties(utIndexColumnTask PROPERTIES
  CXX_STANDARD 17
  COMPILE_DEFINITIONS "${legms_COMPILE_DEFINITIONS}")
add_test(
  NAME IndexColumnTaskUnitTest
  COMMAND ../testing/TestRunner.py ./utIndexColumnTask)

add_executable(utReindexColumnTask utReindexColumnTask.cc)
set_target_properties(utReindexColumnTask PROPERTIES
  CXX_STANDARD 17
  COMPILE_DEFINITIONS "${legms_COMPILE_DEFINITIONS}")
add_test(
  NAME ReindexColumnTaskUnitTest
  COMMAND ../testing/TestRunner.py ./utReindexColumnTask)

if (legms_USE_HDF5)
  add_executable(utHdf5 utHdf5.cc)
  set_target_properties(utHdf5 PROPERTIES
    CXX_STANDARD 17
    COMPILE_DEFINITIONS "${legms_COMPILE_DEFINITIONS}")
  add_test(
    NAME Hdf5UnitTest
    COMMAND ../testing/TestRunner.py ./utHdf5)
endif()

add_subdirectory(data)
