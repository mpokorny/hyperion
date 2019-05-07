set(SOVERSION 1)

#------------------------------------------------------------------------------#
# Legion libraries
#------------------------------------------------------------------------------#

ExternalProject_Get_Property(legion_build INSTALL_DIR)
set(Legion_INSTALL_DIR ${INSTALL_DIR})

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
# legms library
#------------------------------------------------------------------------------#

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -Wall") # TODO: add -Werror

set(CMAKE_CXX_FLAGS_DEBUG
  "${CMAKE_CXX_FLAGS_DEBUG} -Og")

list(APPEND LEGMS_SRC
  legms_c.h
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
  ColumnBuilder.h
  Column.cc
  Column.h
  Column_c.cc
  Column_c.h
  ColumnPartition_c.cc
  ColumnPartition_c.h
  c_util.h
  MSTable.cc
  MSTable.h
  MSTable_c.cc
  MSTable_c.h
  TableBuilder.h
  Table.cc
  Table.h
  Table_c.cc
  Table_c.h
  TableReadTask.cc
  TableReadTask.h
  TableReadTask_c.cc
  TableReadTask_c.h
  WithKeywordsBuilder.h
  WithKeywords.h)

set(LEGMS_INCLUDE LEGMS_SRC)
list(FILTER LEGMS_INCLUDE INCLUDE REGEX ".*\.h")

set(CASACORE
  "/users/mpokorny/projects/casacore.git/casacore-install"
  CACHE PATH "Path to casacore installation")

add_library(legms SHARED ${LEGMS_SRC})
set(legms_COMPILE_DEFINITIONS LEGION_USE_CMAKE REALM_USE_CMAKE)
list(APPEND
  legms_COMPILE_DEFINITIONS "LEGMS_MAX_STRING_SIZE=${legms_MAX_STRING_SIZE}")
# TODO: find a better way of passing the legms_COMPILE_DEFINITIONS to all
# targets, both in this directory and all sub-directories...evil runtime errors
# occur when the wrong or inconsistent LEGMS_MAX_DIM values are used
set_target_properties(legms PROPERTIES
  CXX_STANDARD 17
  COMPILE_DEFINITIONS "${legms_COMPILE_DEFINITIONS}")
add_dependencies(legms legion_build)
target_include_directories(legms
  PRIVATE ${Legion_INSTALL_DIR}/include
  PRIVATE ${CASACORE}/include
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(legms PUBLIC ${CASACORE}/lib)
target_link_directories(legms PUBLIC ${Legion_INSTALL_DIR}/lib64)
target_link_libraries(legms PUBLIC stdc++fs)
target_link_libraries(legms PUBLIC legion)
target_link_libraries(legms PUBLIC realm)
target_link_libraries(legms PUBLIC casa_tables)
target_link_libraries(legms PUBLIC casa_casa)
set_target_properties(legms PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(legms PROPERTIES SOVERSION ${SOVERSION})

install(TARGETS legms
  LIBRARY DESTINATION lib)
install(FILES LEGMS_INCLUDE
  DESTINATION include)

add_subdirectory(testing)
add_subdirectory(tests)
