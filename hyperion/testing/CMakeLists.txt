set(SOVERSION 1)

#------------------------------------------------------------------------------#
# hyperion testing library
#------------------------------------------------------------------------------#

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/TestRunner.py
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/TestRunner.py ${CMAKE_CURRENT_BINARY_DIR}/TestRunner.py)
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
set_target_properties(hyperion_testing PROPERTIES
  CXX_STANDARD 17
  COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
add_dependencies(hyperion_testing hyperion TestRunner)
target_include_directories(hyperion_testing
  PRIVATE ${Legion_INSTALL_DIR}/include
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}../..
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
if (USE_CASACORE)
target_include_directories(hyperion_testing PRIVATE ${casacore_INCLUDE_DIR})
endif()
target_link_directories(hyperion_testing PUBLIC ${Legion_INSTALL_DIR}/lib64)
target_link_libraries(hyperion_testing PUBLIC legion)
target_link_libraries(hyperion_testing PUBLIC realm)
target_link_libraries(hyperion_testing PUBLIC hyperion)
set_target_properties(hyperion_testing PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(hyperion_testing PROPERTIES SOVERSION ${SOVERSION})

add_subdirectory(tests)
