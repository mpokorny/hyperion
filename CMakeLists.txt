cmake_minimum_required(VERSION 3.1)
project(legms)

set(legms_MAX_DIM 4 CACHE STRING "Maximum index space dimension")
set(legms_MAX_STRING_SIZE 80 CACHE STRING "Maximum string size")

enable_testing()

add_subdirectory(legion_build)
add_subdirectory(terra_build)
add_subdirectory(legms)
add_subdirectory(regent)
