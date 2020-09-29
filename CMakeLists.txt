cmake_minimum_required(VERSION 3.14)
project(hyperion)
enable_testing()

enable_language(CXX)

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

# Set a default install prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT
    AND NOT WIN32
    AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CMAKE_INSTALL_PREFIX "/opt/nrao.edu/${PROJECT_NAME}" CACHE PATH "..." FORCE)
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(MIN_CXX_STANDARD 14)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD ${MIN_CXX_STANDARD})
endif()
if(${CMAKE_CXX_STANDARD} LESS ${MIN_CXX_STANDARD})
  message(FATAL_ERROR
    "C++ language standard must not be less than ${MIN_CXX_STANDARD}")
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

option(hyperion_USE_CASACORE "Use casacore" OFF)
option(hyperion_USE_HDF5 "Use HDF5" OFF)
option(hyperion_USE_YAML "Use YAML" OFF)
option(hyperion_USE_OPENMP "Use OpenMP" OFF)
option(hyperion_USE_CUDA "Use CUDA" OFF)
option(hyperion_USE_KOKKOS "Use Kokkos" OFF)
option(hyperion_BUILD_CASACORE "Build casacore" OFF)
mark_as_advanced(hyperion_BUILD_CASACORE)

set(hyperion_MAX_STRING_SIZE 64 CACHE STRING "Maximum string size")
set(hyperion_MAX_NUM_TABLE_COLUMNS 100 CACHE STRING
  "Maximum number of table columns (> 50)")
if(${hyperion_MAX_NUM_TABLE_COLUMNS} LESS_EQUAL 50)
  message(FATAL_ERROR "hyperion_MAX_STRING_SIZE must be greater than 50")
endif()

set(hyperion_MAX_DIM 8 CACHE STRING "Maximum index space dimension")
set_property(CACHE hyperion_MAX_DIM PROPERTY STRINGS 1 2 3 4 5 6 7 8 9)

set(hyperion_BUILD_ARCH "native" CACHE STRING "Target CPU microarchitecture")
string(TOLOWER ${hyperion_BUILD_ARCH} hyperion_BUILD_ARCH)
string(REPLACE "_" "-" hyperion_BUILD_ARCH ${hyperion_BUILD_ARCH}) # for Spack naming

if(hyperion_USE_HDF5)
  find_package(HDF5 REQUIRED)
endif()

if(hyperion_USE_CASACORE AND NOT hyperion_BUILD_CASACORE)
  find_package(Casacore REQUIRED)
endif()

if(hyperion_USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

if(hyperion_USE_CUDA)
  set(hyperion_CUDA_ARCH "" CACHE STRING
    "Comma-separated list of target CUDA architectures (e.g. 60,70)")
endif()

set(KOKKOS_COMPONENTS kokkos kokkoscore)
set(KOKKOS_DEVICES Serial)
if(hyperion_USE_OPENMP)
  list(APPEND KOKKOS_DEVICES OpenMP)
endif()
if(hyperion_USE_CUDA)
  list(APPEND KOKKOS_COMPONENTS CUDA)
  list(APPEND KOKKOS_DEVICES CUDA)
endif()

function(map_to_arch arch_map input_list output_list)
  foreach(arch IN LISTS input_list)
    list(FIND arch_map ${arch} arch_index)
    if(NOT arch_index EQUAL -1)
      math(EXPR arch_index "${arch_index} + 1")
      list(GET arch_map ${arch_index} arch_name)
      list(APPEND arch_names ${arch_name})
    else()
      message(WARNING "'${arch}' does not map to a known architecture")
    endif()
  endforeach()
  set(${output_list} "${arch_names}" PARENT_SCOPE)
endfunction()

if(NOT hyperion_USE_KOKKOS)
  if(hyperion_BUILD_ARCH STREQUAL "none")
    message(
      FATAL_ERROR
      "hyperion_BUILD_ARCH value of 'none' is invalid without hyperion_USE_KOKKOS")
  endif()
else()
  if(hyperion_BUILD_ARCH STREQUAL "native")
    message(
      FATAL_ERROR
      "hyperion_BUILD_ARCH value of 'native' is invalid with hyperion_USE_KOKKOS")
  endif()
  if(hyperion_BUILD_ARCH STREQUAL "none")
    get_property(kokkos_compile_options
      TARGET Kokkos::kokkoscore PROPERTY INTERFACE_COMPILE_OPTIONS)
    string(REGEX MATCH "\\$<\\$<COMPILE_LANGUAGE:CXX>:[^<>]*-march=([^<>;]*)"
      kokkos_march "${kokkos_compile_options}")
    if(NOT kokkos_march OR kokkos_march STREQUAL "")
      message(FATAL_ERROR
        "Failed to get CPU architecture from Kokkos compile options")
    endif()
    set(hyperion_BUILD_ARCH ${CMAKE_MATCH_1})
    message(STATUS "Kokkos-provided BUILD_ARCH: ${hyperion_BUILD_ARCH}")
  endif()

  find_package(Kokkos REQUIRED COMPONENTS ${KOKKOS_COMPONENTS})
  kokkos_check(DEVICES ${KOKKOS_DEVICES})

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
    skylake BDW # Kokkos doesn't know plain skylake, only skylake-x...see Kokkos Spack package file
    icelake SKX
    skylake-avx512 SKX
    skylake_avx512 SKX)

  map_to_arch("${kokkos_cpu_arch_map}" ${hyperion_BUILD_ARCH} cpu_arch_name)
  kokkos_check(ARCH ${cpu_arch_name})

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
    string(REPLACE "," ";" cuda_arch_list ${hyperion_CUDA_ARCH})
    map_to_arch("${kokkos_cuda_arch_map}" "${cuda_arch_list}" cuda_arch_names)
    kokkos_check(ARCH "${cuda_arch_names}")
  endif()
endif()

if(hyperion_USE_OPENMP)
  set(HYPERION_FFTW_COMPONENTS
    FLOAT_OPENMP_LIB FLOAT_LIB DOUBLE_OPENMP_LIB DOUBLE_LIB)
else()
  set(HYPERION_FFTW_COMPONENTS FLOAT_LIB DOUBLE_LIB)
endif()
find_package(FFTW REQUIRED COMPONENTS ${HYPERION_FFTW_COMPONENTS})

if(hyperion_USE_YAML)
  find_package(yaml-cpp REQUIRED)
endif()

if(NOT hyperion_USE_KOKKOS AND hyperion_USE_CUDA)
  enable_language(CUDA)
endif()

#find_package(ZLIB REQUIRED)
#find_package(HDF5 QUIET)

add_subdirectory(dependencies)
add_subdirectory(hyperion)

#add_subdirectory(docs)
