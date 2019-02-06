cmake_minimum_required(VERSION 3.1)
project(legms)

set(legms_MAX_DIM "3" CACHE STRING "Maximum index space dimension")

add_subdirectory(legion_build)
add_subdirectory(terra_build)
add_subdirectory(legms)
add_subdirectory(regent)
