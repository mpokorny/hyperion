set(hyperion_LEGION_MP_GIT_REPO OFF CACHE BOOL "Use Martin's fork of Legion git repository")
if(hyperion_LEGION_MP_GIT_REPO)
  set(hyperion_LEGION_GIT_REPO "git@github.com:mpokorny/legion.git")
else()
  set(hyperion_LEGION_GIT_REPO "https://github.com/StanfordLegion/legion.git")
endif()
set(hyperion_LEGION_TAG "stable" CACHE STRING "Legion repository git tag")

set(Legion_MAX_DIM ${MAX_DIM} CACHE STRING "Maximum number of dimensions")
list(APPEND Legion_ARGS Legion_MAX_DIM)

set(Legion_OUTPUT_LEVEL "DEBUG" CACHE STRING "Compile time logging level")
list(APPEND Legion_ARGS Legion_OUTPUT_LEVEL)

set(Legion_USE_OpenMP OFF CACHE BOOL "Use OpenMP")
list(APPEND Legion_ARGS Legion_USE_OpenMP)

set(Legion_USE_Python OFF CACHE BOOL "Use Python")
list(APPEND Legion_ARGS Legion_USE_Python)

set(Legion_USE_CUDA OFF CACHE BOOL "Enable support for the CUDA runtime")
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

set(Legion_USE_HDF5 ${USE_HDF5} CACHE BOOL "Enable support for HDF5")
list(APPEND Legion_ARGS Legion_USE_HDF5)

set(Legion_USE_LIBDL ON CACHE BOOL "Enable support for libdl")
list(APPEND Legion_ARGS Legion_USE_LIBDL)

set(Legion_USE_ZLIB ${ZLIB_FOUND} CACHE BOOL "Use zlib")
list(APPEND Legion_ARGS Legion_USE_ZLIB)

set(Legion_USE_HWLOC OFF CACHE BOOL "Use hwloc for topology awareness")
list(APPEND Legion_ARGS Legion_USE_HWLOC)

set(Legion_MAX_FIELDS 512 CACHE STRING "Maximum number of fields allocated to a single field space")
list(APPEND Legion_ARGS Legion_MAX_FIELDS)

set(Legion_ENABLE_TLS OFF CACHE BOOL "Enable support for TLS storage of Legion context")
list(APPEND Legion_ARGS Legion_ENABLE_TLS)

set(Legion_CMAKE_BUILD_TYPE "Release" CACHE STRING "CMAKE_BUILD_TYPE definition for Legion")

# to define CXX_FLAGS for Legion build, add something like this to
# ExternalProject_add command:
#
# CMAKE_ARGS -DCMAKE_CXX_FLAGS=-DPRIVILEGE_CHECKS\ -DBOUNDS_CHECKS
include(ExternalProject)
ExternalProject_Add(legion_build
  GIT_REPOSITORY ${hyperion_LEGION_GIT_REPO}
  GIT_TAG ${hyperion_LEGION_TAG}
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${Legion_CMAKE_BUILD_TYPE}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  CMAKE_ARGS -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
  CMAKE_ARGS -DBUILD_SHARED_LIBS=ON
  CMAKE_ARGS -DLegion_BUILD_BINDINGS=${BUILD_REGENT}
  CMAKE_ARGS -DLegion_OUTPUT_LEVEL=${Legion_OUTPUT_LEVEL}
  CMAKE_ARGS -DLegion_MAX_DIM=${Legion_MAX_DIM}
  CMAKE_ARGS -DLegion_USE_OpenMP=${Legion_USE_OpenMP}
  CMAKE_ARGS -DLegion_USE_Python=${Legion_USE_Python}
  CMAKE_ARGS -DLegion_USE_CUDA=${Legion_USE_CUDA}
  CMAKE_ARGS -DLegion_USE_GASNet=${Legion_USE_GASNet}
  CMAKE_ARGS -DGASNet_CONDUIT=${Legion_GASNet_CONDUIT}
  CMAKE_ARGS -DGASNet_ROOT_DIR=${Legion_GASNet_ROOT}
  CMAKE_ARGS -DLegion_USE_LLVM=${Legion_USE_LLVM}
  CMAKE_ARGS -DLegion_ALLOW_MISSING_LLVM_LIBS=${Legion_ALLOW_MISSING_LLVM_LIBS}
  CMAKE_ARGS -DLegion_LINK_LLVM_LIBS=${Legion_LINK_LLVM_LIBS}
  CMAKE_ARGS -DLegion_USE_HDF5=${Legion_USE_HDF5}
  CMAKE_ARGS -DLegion_USE_LIBDL=${Legion_USE_LIBDL}
  CMAKE_ARGS -DLegion_USE_ZLIB=${Legion_USE_ZLIB}
  CMAKE_ARGS -DLegion_USE_HWLOC=${Legion_USE_HWLOC}
  CMAKE_ARGS -DLegion_MAX_FIELDS=${Legion_MAX_FIELDS}
  CMAKE_ARGS -DLegion_ENABLE_TLS=${Legion_ENABLE_TLS}
  CMAKE_ARGS -DLegion_USE_PACKAGE_REGISTRY=OFF
  INSTALL_COMMAND make install && cp -R <SOURCE_DIR>/language/src <INSTALL_DIR>/share/Regent && cp -R <SOURCE_DIR>/bindings/regent <INSTALL_DIR>/include)
