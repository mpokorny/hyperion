include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/..
  ${Legion_INSTALL_DIR}/include)
link_libraries(hyperion_testing)

add_executable(utTestLog utTestLog.cc)
set_target_properties(utTestLog PROPERTIES
  CXX_STANDARD 17
  COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
configure_file(
  utTestLog.expected ${CMAKE_CURRENT_BINARY_DIR}/utTestLog.expected COPYONLY)
add_test(
  NAME TestLogUnitTest
  COMMAND ../TestRunner.py --compare-with utTestLog.expected
                           --compare-by line
                           ./utTestLog)

add_executable(utTestRecorder utTestRecorder.cc)
set_target_properties(utTestRecorder PROPERTIES
  CXX_STANDARD 17
  COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
configure_file(
  utTestRecorder.expected ${CMAKE_CURRENT_BINARY_DIR}/utTestRecorder.expected COPYONLY)
add_test(
  NAME TestRecorderUnitTest
  COMMAND ../TestRunner.py --compare-with utTestRecorder.expected
                           --compare-by line
                           ./utTestRecorder)
