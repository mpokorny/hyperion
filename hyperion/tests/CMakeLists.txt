message(WARNING "Tests will be run with '-lg:inorder' flag")
set(LEGION_ARGS -lg:inorder)
if(hyperion_USE_KOKKOS)
  if(hyperion_USE_OPENMP)
    list(APPEND LEGION_ARGS -ll:ocpu 1 -ll:onuma 0)
  endif()
  if(hyperion_USE_CUDA)
    list(APPEND LEGION_ARGS -ll:gpu 1)
  endif()
endif()

# add_executable(utIndexColumnTask utIndexColumnTask.cc)
# set_target_properties(utIndexColumnTask PROPERTIES
#   CXX_STANDARD ${CXX_STANDARD}
#   COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
# add_test(
#   NAME IndexColumnTaskUnitTest
#   COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
#   ./utIndexColumnTask)

add_executable(utReindexed utReindexed.cc)
set_host_target_properties(utReindexed)
target_link_libraries(utReindexed hyperion_testing)
add_test(
  NAME ReindexedUnitTest
  COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
          ./utReindexed ${LEGION_ARGS})

add_executable(utTable utTable.cc)
set_host_target_properties(utTable)
target_link_libraries(utTable hyperion_testing)
add_test(
  NAME TableUnitTest
  COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
          ./utTable ${LEGION_ARGS})

add_executable(utPartition utPartition.cc)
set_host_target_properties(utPartition)
target_link_libraries(utPartition hyperion_testing)
add_test(
  NAME PartitionUnitTest
  COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
          ./utPartition ${LEGION_ARGS})

add_executable(utMapper utMapper.cc)
set_host_target_properties(utMapper)
target_link_libraries(utMapper hyperion_testing)
add_test(
  NAME DefaultMapperUnitTest
  COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
          ./utMapper ${LEGION_ARGS})

if (USE_HDF5)
  add_executable(utHdf5 utHdf5.cc)
  set_host_target_properties(utHdf5)
  target_link_libraries(utHdf5 hyperion_testing)
  add_test(
    NAME Hdf5UnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
            ./utHdf5 ${LEGION_ARGS})
endif()

if (USE_CASACORE)
  add_executable(utMS utMS.cc)
  set_host_target_properties(utMS)
  target_link_libraries(utMS hyperion_testing)
  add_test(
    NAME MSUnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
            ./utMS -ll:io 1 ${LEGION_ARGS})
  set_tests_properties(
    MSUnitTest PROPERTIES
    FIXTURES_REQUIRED T0MS)

  add_executable(utMeasRef utMeasRef.cc)
  set_host_target_properties(utMeasRef)
  target_link_libraries(utMeasRef hyperion_testing)
  add_test(
    NAME MeasRefUnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
            ./utMeasRef ${LEGION_ARGS})

  add_executable(utMSAntennaTable utMSAntennaTable.cc)
  set_host_target_properties(utMSAntennaTable)
  target_link_libraries(utMSAntennaTable hyperion_testing)
  add_test(
    NAME MSAntennaTableUnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
            ./utMSAntennaTable -ll:io 1 ${LEGION_ARGS})
  set_tests_properties(
    MSAntennaTableUnitTest PROPERTIES
    FIXTURES_REQUIRED T0MS)

  add_executable(utMSSpWindowTable utMSSpWindowTable.cc)
  set_host_target_properties(utMSSpWindowTable)
  target_link_libraries(utMSSpWindowTable hyperion_testing)
  add_test(
    NAME MSSpWindowTableUnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
            ./utMSSpWindowTable -ll:io 1 ${LEGION_ARGS})
  set_tests_properties(
    MSSpWindowTableUnitTest PROPERTIES
    FIXTURES_REQUIRED T0MS)

  add_executable(utMSFieldTable utMSFieldTable.cc)
  set_host_target_properties(utMSFieldTable)
  target_link_libraries(utMSFieldTable hyperion_testing)
  add_test(
    NAME MSFieldTableUnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
            ./utMSFieldTable -ll:io 1 ${LEGION_ARGS})
  set_tests_properties(
    MSFieldTableUnitTest PROPERTIES
    FIXTURES_REQUIRED T0MS)

  add_executable(utMSDataDescriptionTable utMSDataDescriptionTable.cc)
  set_host_target_properties(utMSDataDescriptionTable)
  target_link_libraries(utMSDataDescriptionTable hyperion_testing)
  add_test(
    NAME MSDataDescriptionTableUnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
            ./utMSDataDescriptionTable -ll:io 1 ${LEGION_ARGS})
  set_tests_properties(
    MSDataDescriptionTableUnitTest PROPERTIES
    FIXTURES_REQUIRED T0MS)

  add_executable(utMSMainTable utMSMainTable.cc)
  set_host_target_properties(utMSMainTable)
  target_link_libraries(utMSMainTable hyperion_testing)
  add_test(
    NAME MSMainTableUnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
            ./utMSMainTable -ll:io 1 ${LEGION_ARGS})
  set_tests_properties(
    MSMainTableUnitTest PROPERTIES
    FIXTURES_REQUIRED T0MS)

  add_executable(utMSFeedTable utMSFeedTable.cc)
  set_host_target_properties(utMSFeedTable)
  target_link_libraries(utMSFeedTable hyperion_testing)
  add_test(
    NAME MSFeedTableUnitTest
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../testing/TestRunner.py
            ./utMSFeedTable -ll:io 1 ${LEGION_ARGS})
  set_tests_properties(
    MSFeedTableUnitTest PROPERTIES
    FIXTURES_REQUIRED T0MS)
endif()

add_subdirectory(data)
