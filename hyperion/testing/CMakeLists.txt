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

list(APPEND TESTING_SRC
  TestLog.h
  TestLog.cc
  TestRecorder.h
  TestExpression.h
  TestSuiteDriver.h
  TestSuiteDriver.cc
)

set(TESTING_INCLUDE TESTING_SRC)
list(FILTER TESTING_INCLUDE INCLUDE REGEX ".*\.h")

add_library(hyperion_testing SHARED ${TESTING_SRC})
set_host_target_properties(hyperion_testing)
add_dependencies(hyperion_testing hyperion TestRunner)
target_link_libraries(hyperion_testing PUBLIC hyperion)
set_target_properties(hyperion_testing PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(hyperion_testing PROPERTIES SOVERSION ${SOVERSION})

add_subdirectory(tests)
