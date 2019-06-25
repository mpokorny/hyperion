#------------------------------------------------------------------------------#
# Legion libraries
#------------------------------------------------------------------------------#

ExternalProject_Get_Property(legion_build INSTALL_DIR)
set(Legion_INSTALL_DIR ${INSTALL_DIR})

ExternalProject_Get_Property(terra_build INSTALL_DIR)
set(Terra_INSTALL_DIR ${INSTALL_DIR})

set(LEGION_DIR ${Legion_INSTALL_DIR})
set(REGENT_DIR ${Legion_INSTALL_DIR}/share/Regent)
set(TERRA_DIR ${Terra_INSTALL_DIR})
set(LEGMS_SOURCE_DIR ${CMAKE_SOURCE_DIR}/legms)
set(LEGMS_BINARY_DIR ${CMAKE_BINARY_DIR}/legms)

configure_file(regent.py.in regent.py)
add_custom_target(regent ALL)
