include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${Legion_INSTALL_DIR}/include
  ${CASACORE}/include)
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
