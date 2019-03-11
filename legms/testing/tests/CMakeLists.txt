enable_testing()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${Legion_INSTALL_DIR}/include
  ${CASACORE}/include)
link_libraries(legms_testing)

add_executable(utTestLog utTestLog.cc)
set_target_properties(utTestLog PROPERTIES
  CXX_STANDARD 17)
configure_file(
  utTestLog.expected ${CMAKE_CURRENT_BINARY_DIR}/utTestLog.expected COPYONLY)
add_test(
  NAME TestLogUnitTest
  COMMAND ../TestRunner.py --compare-with utTestLog.expected
                           --compare-by line
                           ./utTestLog)
