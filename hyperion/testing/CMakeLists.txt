set(SOVERSION 1)

#------------------------------------------------------------------------------#
# hyperion testing library
#------------------------------------------------------------------------------#

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/TestRunner.py
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/TestRunner.py ${CMAKE_CURRENT_BINARY_DIR}/TestRunner.py
  MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/TestRunner.py)
add_custom_target(
  TestRunner ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/TestRunner.py)

add_library(hyperion_testing STATIC)
target_sources(hyperion_testing PRIVATE
  TestLog.h
  TestLog.cc
  TestRecorder.h
  TestExpression.h
  TestSuiteDriver.h
  TestSuiteDriver.cc)

set_host_target_properties(hyperion_testing)
add_dependencies(hyperion_testing TestRunner)
target_link_libraries(hyperion_testing PUBLIC hyperion)
set_target_properties(hyperion_testing PROPERTIES
  VERSION 0
  SOVERSION 0.1.0)

add_subdirectory(tests)
