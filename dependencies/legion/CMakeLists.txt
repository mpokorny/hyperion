include(FetchContent)

set(BUILD_SHARED_LIBS ON)

set(Legion_CMAKE_BUILD_TYPE "Release" CACHE STRING "Legion CMake build type")
option(Legion_BOUNDS_CHECKS OFF)
option(Legion_PRIVILEGE_CHECKS OFF)

set(CMAKE_BUILD_TYPE ${Legion_CMAKE_BUILD_TYPE})
set(Legion_MAX_DIM ${MAX_DIM} CACHE STRING "Legion maximum index space dimension")
option(Legion_USE_HDF5 "Enable HDF5 use in Legion" ${USE_HDF5})
option(Legion_USE_OpenMP "Enable OpenMP use in Legion" ${USE_OPENMP})
option(Legion_USE_CUDA "Enable CUDA in Legion" ${USE_CUDA})
set(Legion_CUDA_ARCH ${CUDA_ARCH} CACHE STRING "Legion build CUDA architectures")
option(Legion_USE_Kokkos "Enable Kokkos use in Legion" ${USE_KOKKOS})

if(Legion_BOUNDS_CHECKS)
  add_compile_options("BOUNDS_CHECKS")
endif()
if(Legion_PRIVILEGE_CHECKS)
  add_compile_options( "PRIVILEGE_CHECKS")
endif()

set(BUILD_MARCH ${BUILD_ARCH})
set(SYSCONFDIR ${SYSCONFDIR}/hyperion)

FetchContent_MakeAvailable(legion)
