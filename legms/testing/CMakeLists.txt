set(SOVERSION 1)

#------------------------------------------------------------------------------#
# legms testing library
#------------------------------------------------------------------------------#

list(APPEND TESTING_SRC
  TestLog.h
  TestLog.cc
)

set(TESTING_INCLUDE TESTING_SRC)
list(FILTER TESTING_INCLUDE INCLUDE REGEX ".*\.h")

add_library(legms_testing SHARED ${TESTING_SRC})
set_target_properties(legms_testing PROPERTIES
  CXX_STANDARD 17)
add_dependencies(legms_testing legms)
target_include_directories(legms_testing
  PRIVATE ${Legion_INSTALL_DIR}/include
  PRIVATE ${CASACORE}/include
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_directories(legms_testing PUBLIC ${Legion_INSTALL_DIR}/lib64)
target_link_libraries(legms_testing PUBLIC legion)
target_link_libraries(legms_testing PUBLIC realm)
target_link_libraries(legms_testing PUBLIC legms)
set_target_properties(legms_testing PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(legms_testing PROPERTIES SOVERSION ${SOVERSION})

macro(add_python_target tgt)
  foreach(file ${ARGN})
    set(OUT ${CMAKE_CURRENT_BINARY_DIR}/${file})
    list(APPEND OUT_FILES ${OUT})
    configure_file(${file} ${OUT} COPYONLY)
  endforeach()
  add_custom_target(${tgt} ALL DEPENDS ${OUT_FILES})
endmacro()

add_python_target(TestRunner TestRunner.py)

add_subdirectory(tests)
