#------------------------------------------------------------------------------#
# hyperion synthesis library
#------------------------------------------------------------------------------#
set(SOVERSION 1)

list(APPEND SYNTHESIS_SRC
  CFTableBase.h
  CFTableBase.cc
  CFTable.h
  CFPhysicalTable.h
  PSTermTable.h
  PSTermTable.cc
  WTermTable.h
  WTermTable.cc)

set(SYNTHESIS_INCLUDE SYNTHESIS_SRC)
list(FILTER SYNTHESIS_INCLUDE INCLUDE REGEX ".*\.h")

add_library(hyperion_synthesis SHARED ${SYNTHESIS_SRC})
set_target_properties(hyperion_synthesis PROPERTIES
  CXX_STANDARD 17
  COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
add_dependencies(hyperion_synthesis hyperion)
target_include_directories(hyperion_synthesis
  PRIVATE ${Legion_INSTALL_DIR}/include
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
target_include_directories(hyperion_synthesis PUBLIC ${casacore_INCLUDE_DIR})
target_link_libraries(hyperion_synthesis PUBLIC hyperion)
set_target_properties(hyperion_synthesis PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(hyperion_synthesis PROPERTIES SOVERSION ${SOVERSION})
