cmake_minimum_required(VERSION 3.13)
project(legms)

find_package(ZLIB)
find_package(HDF5)
find_package(casacore)

set(MAX_DIM 7 CACHE STRING "Maximum index space dimension")
set(MAX_STRING_SIZE 80 CACHE STRING "Maximum string size")
set(MAX_NUM_KEYWORDS 16 CACHE STRING "Maximum number of table/column keywords")
set(MAX_NUM_TABLE_COLUMNS 35 CACHE STRING "Maximum number of table columns")
set(BUILD_REGENT ON CACHE BOOL "Build Regent")
option(USE_HDF5 "Enable HDF5" ${HDF5_FOUND})
option(USE_CASACORE "Enable cascore" ${casacore_FOUND})

enable_testing()

add_subdirectory(legion_build)
if(BUILD_REGENT)
  add_subdirectory(terra_build)
  add_subdirectory(regent)
endif(BUILD_REGENT)
add_subdirectory(legms)
