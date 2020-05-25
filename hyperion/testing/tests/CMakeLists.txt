add_executable(utTestLog utTestLog.cc)
set_host_target_properties(utTestLog)
target_link_libraries(utTestLog hyperion_testing)
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
set_host_target_properties(utTestRecorder)
target_link_libraries(utTestRecorder hyperion_testing)
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
