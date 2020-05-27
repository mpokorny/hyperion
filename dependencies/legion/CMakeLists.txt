include(FetchContent)

# forward some build parameters, all of which are defined by options in Legion,
# to Legion
set(Legion_USE_HDF5 ${USE_HDF5})
set(Legion_USE_OpenMP ${USE_OPENMP})
set(Legion_USE_CUDA ${USE_CUDA})
set(Legion_USE_Kokkos ${USE_KOKKOS})

# forward some build parameters, all which are defined by cache variables in
# Legion, to Legion
set(Legion_MAX_DIM ${MAX_DIM} CACHE STRING "Legion maximum index space dimension")
set(Legion_CUDA_ARCH ${CUDA_ARCH} CACHE STRING "Legion build CUDA architectures")

# hyperion-defined Legion-specific build parameters
set(Legion_CMAKE_BUILD_TYPE "Release" CACHE STRING "Legion CMake build type")
option(Legion_BOUNDS_CHECKS OFF)
option(Legion_PRIVILEGE_CHECKS OFF)

if(Legion_BOUNDS_CHECKS)
  add_compile_options("BOUNDS_CHECKS")
endif()
if(Legion_PRIVILEGE_CHECKS)
  add_compile_options( "PRIVILEGE_CHECKS")
endif()

set(BUILD_MARCH ${BUILD_ARCH})
set(SYSCONFDIR ${SYSCONFDIR}/hyperion)

# save some build parameters that will be changed for Legion build
set(CMAKE_BUILD_TYPE_COPY ${CMAKE_BUILD_TYPE})
set(BUILD_SHARED_LIBS_COPY ${BUILD_SHARED_LIBS})

# set Legion build parameters
set(CMAKE_BUILD_TYPE ${Legion_CMAKE_BUILD_TYPE})
set(BUILD_SHARED_LIBS ON)

set(CMAKE_PROJECT_INCLUDE_BEFORE "${CMAKE_CURRENT_LIST_DIR}/cmp0077new.cmake")
FetchContent_MakeAvailable(legion)

# restore original build parameters
set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_COPY})
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_COPY})
