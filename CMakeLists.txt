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

set(Legion_MAX_DIM ${MAX_DIM} STRING "Maximum number of dimensions")
list(APPEND Legion_ARGS Legion_MAX_DIM)

set(Legion_OUTPUT_LEVEL "DEBUG" CACHE STRING "Compile time logging level")
list(APPEND Legion_ARGS Legion_OUTPUT_LEVEL)

set(Legion_USE_OpenMP OFF CACHE BOOL "Use OpenMP")
list(APPEND Legion_ARGS Legion_USE_OpenMP)

set(Legion_USE_Python OFF CACHE BOOL "Use Python")
list(APPEND Legion_ARGS Legion_USE_Python)

set(Legion_USE_CUDA ${USE_CUDA} CACHE BOOL "Enable support for the CUDA runtime")
list(APPEND Legion_ARGS Legion_USE_CUDA)

set(Legion_USE_GASNet OFF CACHE BOOL "Enable the distributed GASNet backend")
list(APPEND Legion_ARGS Legion_USE_GASNet)

set(Legion_GASNet_CONDUIT "ibv" CACHE STRING "GASNet conduit")
list(APPEND Legion_ARGS Legion_GASNet_CONDUIT)

set(Legion_GASNet_ROOT "" CACHE STRING "GASNet root directory")
list(APPEND Legion_ARGS Legion_GASNet_ROOT)

set(Legion_USE_LLVM OFF CACHE BOOL "Use LLVM JIT operations")
list(APPEND Legion_ARGS Legion_USE_LLVM)

if(Legion_USE_LLVM)
  set(Legion_ALLOW_MISSING_LLVM_LIBS ON BOOL)
  set(Legion_LINK_LLVM_LIBS OFF BOOL)
  # TODO: set LLVM_CONFIG_EXECUTABLE to env var LLVM_CONFIG, if set
else()
  set(Legion_ALLOW_MISSING_LLVM_LIBS OFF BOOL)
  set(Legion_LINK_LLVM_LIBS ON BOOL)
endif()

set(Legion_USE_HDF5 ${USE_HDF5} BOOL "Enable support for HDF5")
list(APPEND Legion_ARGS Legion_USE_HDF5)

set(Legion_USE_LIBDL ON CACHE BOOL "Enable support for libdl")
list(APPEND Legion_ARGS Legion_USE_LIBDL)

set(Legion_USE_ZLIB ${ZLIB_FOUND} BOOL "Use zlib")
list(APPEND Legion_ARGS Legion_USE_ZLIB)

set(Legion_USE_HWLOC OFF CACHE BOOL "Use hwloc for topology awareness")
list(APPEND Legion_ARGS Legion_USE_HWLOC)

set(Legion_MAX_FIELDS 512 CACHE STRING "Maximum number of fields allocated to a single field space")
list(APPEND Legion_ARGS Legion_MAX_FIELDS)

set(Legion_ENABLE_TLS OFF CACHE BOOL "Enable support for TLS storage of Legion context")
list(APPEND Legion_ARGS Legion_ENABLE_TLS)

enable_testing()

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
