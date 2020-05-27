cmake_minimum_required(VERSION 3.14)
project(hyperion)
enable_testing()

cmake_policy(SET CMP0077 NEW)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# Set a default build type if none was specified
set(default_build_type Release)
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(default_build_type Debug)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

option(USE_HDF5 "Use HDF5" ON)
option(USE_CASACORE "Use casacore" ON)
option(USE_OPENMP "Use OpenMP" OFF)
option(USE_CUDA "Use CUDA" OFF)
option(USE_KOKKOS "Use Kokkos" OFF)
option(BUILD_CASACORE "Build casacore" OFF)
mark_as_advanced(BUILD_CASACORE)

set(MAX_DIM 7 CACHE STRING "Maximum index space dimension")
set_property(CACHE MAX_DIM PROPERTY STRINGS 1 2 3 4 5 6 7 8 9)

set(BUILD_ARCH "native" CACHE STRING "Target CPU microarchitecture")
string(TOLOWER ${BUILD_ARCH} BUILD_ARCH)

if (USE_HDF5)
  find_package(HDF5 REQUIRED)
endif()

if (USE_CASACORE AND NOT BUILD_CASACORE)
  find_package(Casacore REQUIRED)
endif()

if (USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

if(USE_CUDA)
  set(CUDA_ARCH "" CACHE STRING "Comma-separated list of target CUDA architectures (e.g. 60,70)")
endif()

set(KOKKOS_COMPONENTS kokkos kokkoscore)
set(KOKKOS_DEVICES Serial)
if(USE_OPENMP)
  list(APPEND KOKKOS_DEVICES OpenMP)
endif()
if(USE_CUDA)
  list(APPEND KOKKOS_COMPONENTS CUDA)
  list(APPEND KOKKOS_DEVICES CUDA)
endif()

macro(map_to_arch arch_map input_list output_list)
  foreach(arch IN LISTS input_list)
    list(FIND ${arch_map} ${arch} arch_index)
    if(NOT arch_index EQUAL -1)
      math(EXPR arch_index "${arch_index} + 1")
      list(GET arch_map ${arch_index} arch_name)
      list(APPEND output_list ${arch_name})
    else()
      message(WARNING "'${arch}' does not map to a known architecture")
    endif()
  endforeach()
endmacro()

if(USE_KOKKOS)
  find_package(Kokkos REQUIRED COMPONENTS ${KOKKOS_COMPONENTS})
  kokkos_check(DEVICES ${KOKKOS_DEVICES})

  if(NOT BUILD_ARCH STREQUAL "none")
    set(kokkos_cpu_arch_map
      thunderx2 THUNDERX2
      zen ZEN
      zen2 ZEN2
      steamroller KAVERI
      excavator CARIZO
      power7 POWER7
      power8 POWER8
      power9 POWER9
      power8le POWER8
      power9le POWER9
      sandybridge SNB
      haswell HSW
      mic-knl KNL
      knl KNL
      cannonlake SKX
      cascadelake SKX
      westmere WSM
      ivybridge SNB
      broadwell BDW
      skylake SKX
      icelake SKX
      skylake-avx512 SKX
      skylake_avx512 SKX)

    map_to_arch(kokkos_cpu_arch_map BUILD_ARCH cpu_arch_name)
    kokkos_check(ARCH ${cpu_arch_name})
  endif()

  if(USE_CUDA)
    set(kokkos_cuda_arch_map
      30 KEPLER30
      32 KEPLER32
      35 KEPLER35
      37 KEPLER37
      50 MAXWELL50
      52 MAXWELL52
      53 MAXWELL53
      60 PASCAL60
      61 PASCAL61
      70 VOLTA70
      72 VOLTA72
      75 TURING75)
    string(REPLACE "," ";" cuda_arch_list CUDA_ARCH)
    map_to_arch(kokkos_cuda_arch_map cuda_arch_list cuda_arch_names)
    kokkos_check(ARCH ${cuda_arch_names})
  endif()
endif()

include(GNUInstallDirs)

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

include(CheckCXXCompilerFlag)

# include(CheckLanguage)
if(USE_KOKKOS)
  # for nvcc_wrapper
  check_cxx_compiler_flag(--host-only COMPILER_SUPPORTS_HOST_ONLY)
  if(COMPILER_SUPPORTS_HOST_ONLY)
    set(HOST_ONLY --host-only)
  endif()
endif()

#find_package(ZLIB REQUIRED)
#find_package(HDF5 QUIET)

add_subdirectory(dependencies)
add_subdirectory(hyperion)

#add_subdirectory(docs)
