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

# add_executable(utIndexColumnTask utIndexColumnTask.cc)
# set_target_properties(utIndexColumnTask PROPERTIES
#   CXX_STANDARD 17
#   COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
# add_test(
#   NAME IndexColumnTaskUnitTest
#   COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
#   ./utIndexColumnTask)

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

add_executable(utPartition utPartition.cc)
set_target_properties(utPartition PROPERTIES
  CXX_STANDARD 17
  COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
add_test(
  NAME PartitionUnitTest
  COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
  ./utPartition)

add_executable(utMapper utMapper.cc)
set_target_properties(utMapper PROPERTIES
  CXX_STANDARD 17
  COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
add_test(
  NAME DefaultMapperUnitTest
  COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
  ./utMapper)

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

  add_executable(utMSAntennaTable utMSAntennaTable.cc)
  set_target_properties(utMSAntennaTable PROPERTIES
    CXX_STANDARD 17
    COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
  add_test(
    NAME MSAntennaTableUnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
    ./utMSAntennaTable -ll:io 1)
  set_tests_properties(
    MSAntennaTableUnitTest PROPERTIES
    FIXTURES_REQUIRED T0MS)

  add_executable(utMSSpWindowTable utMSSpWindowTable.cc)
  set_target_properties(utMSSpWindowTable PROPERTIES
    CXX_STANDARD 17
    COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
  add_test(
    NAME MSSpWindowTableUnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
    ./utMSSpWindowTable -ll:io 1)
  set_tests_properties(
    MSSpWindowTableUnitTest PROPERTIES
    FIXTURES_REQUIRED T0MS)

  add_executable(utMSFieldTable utMSFieldTable.cc)
  set_target_properties(utMSFieldTable PROPERTIES
    CXX_STANDARD 17
    COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
  add_test(
    NAME MSFieldTableUnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
    ./utMSFieldTable -ll:io 1)
  set_tests_properties(
    MSFieldTableUnitTest PROPERTIES
    FIXTURES_REQUIRED T0MS)

  add_executable(utMSDataDescriptionTable utMSDataDescriptionTable.cc)
  set_target_properties(utMSDataDescriptionTable PROPERTIES
    CXX_STANDARD 17
    COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
  add_test(
    NAME MSDataDescriptionTableUnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
    ./utMSDataDescriptionTable -ll:io 1)
  set_tests_properties(
    MSDataDescriptionTableUnitTest PROPERTIES
    FIXTURES_REQUIRED T0MS)
endif()

add_subdirectory(data)
