include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_BINARY_DIR}/../..
  ${Legion_INSTALL_DIR}/include)
if (USE_CASACORE)
  include_directories(${casacore_INCLUDE_DIR})
endif(USE_CASACORE)
if (USE_HDF5)
  include_directories(${HDF5_INCLUDE_DIRS})
  link_libraries(${HDF5_LIBRARIES})
endif (USE_HDF5)
link_libraries(hyperion_testing)

add_executable(utIndexColumnTask utIndexColumnTask.cc)
set_target_properties(utIndexColumnTask PROPERTIES
  CXX_STANDARD 17
  COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
add_test(
  NAME IndexColumnTaskUnitTest
  COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
  ./utIndexColumnTask)

add_executable(utReindexColumnTask utReindexColumnTask.cc)
set_target_properties(utReindexColumnTask PROPERTIES
  CXX_STANDARD 17
  COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
add_test(
  NAME ReindexColumnTaskUnitTest
  COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
  ./utReindexColumnTask)

add_executable(utReindexed utReindexed.cc)
set_target_properties(utReindexed PROPERTIES
  CXX_STANDARD 17
  COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
add_test(
  NAME ReindexedUnitTest
  COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
  ./utReindexed)

add_executable(utTable utTable.cc)
set_target_properties(utTable PROPERTIES
  CXX_STANDARD 17
  COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
add_test(
  NAME TableUnitTest
  COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
          ./utTable)

if (USE_HDF5)
  add_executable(utHdf5 utHdf5.cc)
  set_target_properties(utHdf5 PROPERTIES
    CXX_STANDARD 17
    COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
  add_test(
    NAME Hdf5UnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
            ./utHdf5)
endif()

if (USE_CASACORE)
  add_executable(utMS utMS.cc)
  set_target_properties(utMS PROPERTIES
    CXX_STANDARD 17
    COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
  add_test(
    NAME MSUnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
    ./utMS -ll:io 1)
  set_tests_properties(
    MSUnitTest PROPERTIES
    FIXTURES_REQUIRED T0MS)

  add_executable(utMeasRef utMeasRef.cc)
  set_target_properties(utMeasRef PROPERTIES
    CXX_STANDARD 17
    COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
  add_test(
    NAME MeasRefUnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
            ./utMeasRef)

  add_executable(utMeasRefDict utMeasRefDict.cc)
  set_target_properties(utMeasRefDict PROPERTIES
    CXX_STANDARD 17
    COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
  add_test(
    NAME MeasRefDictUnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
            ./utMeasRefDict)
endif()

add_subdirectory(data)
