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
  CXX_STANDARD ${CXX_STANDARD}
  CXX_EXTENSIONS ON)
target_compile_options(hyperion_synthesis PRIVATE ${HOST_ONLY})
target_link_libraries(hyperion_synthesis PUBLIC hyperion)
set_target_properties(hyperion_synthesis PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(hyperion_synthesis PROPERTIES SOVERSION ${SOVERSION})
