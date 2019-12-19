set(SOVERSION 1)

find_package(Filesystem REQUIRED COMPONENTS Final Experimental)

#------------------------------------------------------------------------------#
# Legion libraries
#------------------------------------------------------------------------------#

ExternalProject_Get_Property(legion_build INSTALL_DIR)
set(Legion_INSTALL_DIR ${INSTALL_DIR})
if (BUILD_CASACORE)
  ExternalProject_Get_Property(casacore_build INSTALL_DIR)
  set(
    CASACORE_LIBRARIES
    "${INSTALL_DIR}/lib/libcasa_measures.so"
    "${INSTALL_DIR}/lib/libcasa_scimath.so"
    "${INSTALL_DIR}/lib/libcasa_scimath_f.so"
    "${INSTALL_DIR}/lib/libcasa_tables.so"
    "${INSTALL_DIR}/lib/libcasa_casa.so")
  set(CASACORE_INCLUDE_DIR ${INSTALL_DIR}/include)
endif ()

#------------------------------------------------------------------------------#
#
# Legion compile option compatibility
#
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Architecture
#------------------------------------------------------------------------------#
if(BUILD_MARCH AND BUILD_MCPU)
  message(FATAL_ERROR "BUILD_MARCH and BUILD_MCPU are incompatible")
endif()

# Try -march first. On platforms that don't support it, GCC will issue
# a hard error, so we'll know not to use it.
if(BUILD_MARCH)
  set(INTERNAL_BUILD_MARCH ${BUILD_MARCH})
else()
  set(INTERNAL_BUILD_MARCH "native")
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=${INTERNAL_BUILD_MARCH}" COMPILER_SUPPORTS_MARCH)
if(COMPILER_SUPPORTS_MARCH)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${INTERNAL_BUILD_MARCH}")
elseif(BUILD_MARCH)
  message(FATAL_ERROR "The flag -march=${INTERNAL_BUILD_MARCH} is not supported by the compiler")
else()
  unset(INTERNAL_BUILD_MARCH)
endif()

# Try -mcpu. We do this second because it is deprecated on x86, but
# GCC won't issue a hard error, so we can't tell if it worked or not.
if (NOT INTERNAL_BUILD_MARCH)
  if(BUILD_MCPU)
    set(INTERNAL_BUILD_MCPU ${BUILD_MCPU})
  else()
    set(INTERNAL_BUILD_MCPU "native")
  endif()

  check_cxx_compiler_flag("-mcpu=${INTERNAL_BUILD_MCPU}" COMPILER_SUPPORTS_MCPU)
  if(COMPILER_SUPPORTS_MCPU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=${INTERNAL_BUILD_MCPU}")
  elseif(BUILD_MCPU)
    message(FATAL_ERROR "The flag -mcpu=${INTERNAL_BUILD_MCPU} is not supported by the compiler")
  else()
    unset(INTERNAL_BUILD_MCPU)
  endif()
endif()

# Add flags for Power architectures
check_cxx_compiler_flag("-maltivec -Werror" COMPILER_SUPPORTS_MALTIVEC)
if(COMPILER_SUPPORTS_MALTIVEC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maltivec")
endif()
check_cxx_compiler_flag("-mabi=altivec -Werror" COMPILER_SUPPORTS_MABI_ALTIVEC)
if(COMPILER_SUPPORTS_MABI_ALTIVEC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mabi=altivec")
endif()
check_cxx_compiler_flag("-mvsx -Werror" COMPILER_SUPPORTS_MVSX)
if(COMPILER_SUPPORTS_MVSX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mvsx")
endif()

# Legion can't handle certain optimizations. See:
# https://github.com/StanfordLegion/legion/issues/168#issuecomment-244582958
check_cxx_compiler_flag("-fno-strict-aliasing" COMPILER_SUPPORTS_NO_STRICT_ALIASING)
if(COMPILER_SUPPORTS_NO_STRICT_ALIASING)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing")
endif()

#------------------------------------------------------------------------------#
# hyperion library
#------------------------------------------------------------------------------#

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -Wall -fvisibility=hidden")
# For clang, will need the following
# set(CMAKE_CXX_FLAGS
#   "${CMAKE_CXX_FLAGS} -Wall -fvisibility=hidden -frelaxed-template-template-args") 
# TODO: add -Werror

set(CMAKE_CXX_FLAGS_DEBUG
  "${CMAKE_CXX_FLAGS_DEBUG} -Og")

list(APPEND HYPERION_SRC
  hyperion_c.h
  Grids_c.cc
  Grids_c.h
  Grids.h
  IndexTree.h
  tree_index_space.cc
  tree_index_space.h
  utility.cc
  utility.h
  utility_c.cc
  utility_c.h
  Column.cc
  Column.h
  Column_c.cc
  Column_c.h
  ColumnBuilder.h
  ColumnPartition.h
  ColumnPartition.cc
  ColumnPartition_c.cc
  ColumnPartition_c.h
  c_util.h
  Keywords.h
  Keywords.cc
  MSTable.cc
  MSTable.h
  MSTable_c.cc
  MSTable_c.h
  MSTableColumns_c.h
  MSTableColumns.h
  MSTableColumns.cc
  Table.cc
  Table.h
  Table_c.cc
  Table_c.h
  KeywordsBuilder.h
  MSAntennaColumns.h
  MSAntennaColumns.cc
  MSFieldColumns.h
  MSFieldColumns.cc
  MSSpWindowColumns.h
  MSSpWindowColumns.cc
  x/Column.h
  x/ColumnSpace.h
  x/ColumnSpace.cc
  x/Table.h
  x/Table.cc
  x/ColumnSpacePartition.h
  x/ColumnSpacePartition.cc)

if(USE_CASACORE)
  list(APPEND HYPERION_SRC
    TableBuilder.h
    TableBuilder.cc
    TableReadTask.cc
    TableReadTask.h
    TableReadTask_c.cc
    TableReadTask_c.h
    Measures.h
    Measures.cc
    MeasRef.h
    MeasRef.cc
    MeasRefDict.h
    MeasRefDict.cc
    MeasRefContainer.h
    MeasRefContainer.cc)
  set(HYPERION_USE_CASACORE ON)
endif()
if(USE_HDF5)
  list(APPEND HYPERION_SRC
    hdf5.h
    hdf5.cc)
  set(HYPERION_USE_HDF5 ON)
endif()
if (USE_YAML)
  set(HYPERION_USE_YAML ON)
endif ()

configure_file(hyperion_config.h.in hyperion_config.h)

set(HYPERION_INCLUDE HYPERION_SRC)
list(FILTER HYPERION_INCLUDE INCLUDE REGEX ".*\.h")

add_library(hyperion SHARED ${HYPERION_SRC})
set(hyperion_COMPILE_DEFINITIONS "")
if(USE_HDF5)
  list(APPEND
    hyperion_COMPILE_DEFINITIONS "H5_NO_DEPRECATED_SYMBOLS")
endif()

list(APPEND
  hyperion_COMPILE_DEFINITIONS "LEGION_MAX_DIM=${MAX_DIM}")
list(APPEND
  hyperion_COMPILE_DEFINITIONS "REALM_MAX_DIM=${MAX_DIM}")

set_target_properties(hyperion PROPERTIES
  CXX_STANDARD 17
  COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
add_dependencies(hyperion legion_build)
target_include_directories(hyperion
  PUBLIC $<BUILD_INTERFACE:${Legion_INSTALL_DIR}/include>
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
  PUBLIC $<INSTALL_INTERFACE:include>)
target_link_directories(hyperion
  PUBLIC $<BUILD_INTERFACE:${Legion_INSTALL_DIR}/lib64>
  PUBLIC $<BUILD_INTERFACE:${Legion_INSTALL_DIR}/lib>)
if (USE_CASACORE)
  if (BUILD_CASACORE)
    add_dependencies(hyperion casacore_build)
    target_include_directories(hyperion
      PUBLIC $<BUILD_INTERFACE:${CASACORE_INCLUDE_DIR}>)
  else()
    target_include_directories(hyperion
      PUBLIC ${CASACORE_INCLUDE_DIR})
  endif ()
  target_link_libraries(hyperion PUBLIC ${CASACORE_LIBRARIES})
endif()
if (USE_HDF5)
  target_include_directories(hyperion PUBLIC ${HDF5_INCLUDE_DIRS})
  target_link_libraries(hyperion PUBLIC ${HDF5_LIBRARIES})
endif()
target_link_libraries(hyperion PUBLIC std::filesystem)
target_link_libraries(hyperion PUBLIC legion)
target_link_libraries(hyperion PUBLIC realm)
set_target_properties(hyperion PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(hyperion PROPERTIES SOVERSION ${SOVERSION})
install(TARGETS hyperion EXPORT HyperionConfig
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES HYPERION_INCLUDE DESTINATION include/hyperion)
install(EXPORT HyperionConfig DESTINATION share/hyperion/cmake)
export(TARGETS hyperion FILE HyperionConfig.cmake)

if (USE_HDF5 AND USE_CASACORE)
  add_executable(ms2h5 ms2h5.cc)
  set_target_properties(ms2h5 PROPERTIES
    CXX_STANDARD 17
    COMPILE_DEFINITIONS "${hyperion_COMPILE_DEFINITIONS}")
  target_link_libraries(ms2h5 hyperion)
  target_include_directories(ms2h5
    PUBLIC ${Legion_INSTALL_DIR}/include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/..
    PUBLIC ${CASACORE_INCLUDE_DIR}
    PUBLIC ${HDF5_INCLUDE_DIRS})
  install(TARGETS ms2h5 RUNTIME DESTINATION bin)
endif()

add_subdirectory(gridder)

add_subdirectory(testing)
add_subdirectory(tests)
