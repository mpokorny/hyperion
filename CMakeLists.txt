cmake_minimum_required(VERSION 3.13)
project(legms)

find_package(HDF5)

set(legms_MAX_DIM 4 CACHE STRING "Maximum index space dimension")
set(legms_MAX_STRING_SIZE 80 CACHE STRING "Maximum string size")
option(legms_USE_HDF "Enable HDF5" ${HDF5_FOUND})

enable_testing()

add_subdirectory(legion_build)
add_subdirectory(terra_build)
add_subdirectory(legms)
add_subdirectory(regent)
