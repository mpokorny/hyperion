cmake_minimum_required(VERSION 3.13)
project(hyperion
  LANGUAGES CXX C)

cmake_policy(SET CMP0077 NEW)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(CheckLanguage)

find_package(ZLIB REQUIRED)
find_package(Casacore COMPONENTS ms)
find_package(HDF5 QUIET)
find_package(yaml-cpp QUIET)
find_package(Filesystem REQUIRED COMPONENTS Final Experimental)

set(MAX_DIM 7 CACHE STRING "Maximum index space dimension")
set(MAX_STRING_SIZE 64 CACHE STRING "Maximum string size")
set(MAX_NUM_TABLE_COLUMNS 30 CACHE STRING "Maximum number of table columns")
set(BUILD_REGENT OFF CACHE BOOL "Build Regent")
set(USE_HDF5 ${HDF5_FOUND} CACHE BOOL "Use HDF5")
set(USE_CASACORE ON CACHE BOOL "Use casacore")
set(CASACORE_ROOT_DIR "" CACHE STRING "casacore installation root path")
set(USE_YAML ${yaml-cpp_FOUND} CACHE BOOL "Use YAML")
check_language(CUDA)
if (NOT "${CMAKE_CUDA_COMPILER}" STREQUAL "NOTFOUND")
  set(HAVE_CUDA ON)
else()
  set(HAVE_CUDA OFF)
endif()
set(USE_CUDA ${HAVE_CUDA} CACHE BOOL "Use CUDA")

if (USE_HDF5)
  find_package(HDF5 REQUIRED)
endif ()
if (USE_YAML)
  find_package(yaml-cpp REQUIRED)
endif()
if (USE_CUDA)
  enable_language(CUDA)
endif()

if (USE_CASACORE AND NOT CASACORE_FOUND)
  message(STATUS "Building casacore within project")
  set(BUILD_CASACORE ON)
else ()
  message(STATUS "Using casacore installation at ${CASACORE_ROOT_DIR}")
  set(BUILD_CASACORE OFF)
  if (USE_CASACORE AND NOT CASACORE_ROOT_DIR STREQUAL "" AND NOT CASACORE_FOUND)
    message(FATAL_ERROR "USE_CASACORE is ON, yet casacore package was not found at ${CASACORE_ROOT_DIR}")
  endif ()
endif ()

enable_testing()

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_FULL_LIBDIR}/hyperion" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}/hyperion")
endif("${isSystemDir}" STREQUAL "-1")

add_subdirectory(legion)
if (BUILD_REGENT)
  add_subdirectory(terra)
  add_subdirectory(regent)
endif ()
if (BUILD_CASACORE)
  add_subdirectory(casacore)
endif ()
add_subdirectory(hyperion)
add_subdirectory(docs)
