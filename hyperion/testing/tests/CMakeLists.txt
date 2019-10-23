include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/..
  ${Legion_INSTALL_DIR}/include)
link_libraries(hyperion_testing)

add_executable(utTestLog utTestLog.cc)
set_target_properties(utTestLog PROPERTIES
  CXX_STANDARD 17
  COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
add_test(
  NAME cpTestLogExpected
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/utTestLog.expected
             ${CMAKE_CURRENT_BINARY_DIR}/utTestLog.expected)
set_tests_properties(
  cpTestLogExpected PROPERTIES
  FIXTURES_SETUP TestLogExpected)
add_test(
  NAME TestLogUnitTest
  COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../TestRunner.py
          --compare-with ${CMAKE_CURRENT_BINARY_DIR}/utTestLog.expected
          --compare-by line
          ./utTestLog)
set_tests_properties(
  TestLogUnitTest PROPERTIES
  FIXTURES_REQUIRED TestLogExpected)

add_executable(utTestRecorder utTestRecorder.cc)
set_target_properties(utTestRecorder PROPERTIES
  CXX_STANDARD 17
  COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
add_test(
  NAME cpTestRecorderExpected
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/utTestRecorder.expected
             ${CMAKE_CURRENT_BINARY_DIR}/utTestRecorder.expected)
set_tests_properties(
  cpTestRecorderExpected PROPERTIES
  FIXTURES_SETUP TestRecorderExpected)
add_test(
  NAME TestRecorderUnitTest
  COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/../TestRunner.py
          --compare-with ${CMAKE_CURRENT_BINARY_DIR}/utTestRecorder.expected
          --compare-by line
          ./utTestRecorder)
set_tests_properties(
  TestRecorderUnitTest PROPERTIES
  FIXTURES_REQUIRED TestRecorderExpected)
