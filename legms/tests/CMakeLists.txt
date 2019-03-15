include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${Legion_INSTALL_DIR}/include
  ${CASACORE}/include)
link_libraries(legms_testing)

add_executable(utIndexColumnTask utIndexColumnTask.cc)
set_target_properties(utIndexColumnTask PROPERTIES
  CXX_STANDARD 17)
# configure_file(
#   utIndexColumnTask.expected ${CMAKE_CURRENT_BINARY_DIR}/utIndexColumnTask.expected COPYONLY)
add_test(
  NAME IndexColumnTaskUnitTest
  COMMAND ../testing/TestRunner.py ./utIndexColumnTask)
