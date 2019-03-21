set(legms_LEGION_MP_GIT_REPO OFF CACHE BOOL "Use Martin's fork of Legion git repository")
if(legms_LEGION_MP_GIT_REPO)
  set(legms_LEGION_GIT_REPO "git@github.com:mpokorny/legion.git")
else()
  set(legms_LEGION_GIT_REPO "https://github.com/StanfordLegion/legion.git")
endif()
set(legms_LEGION_TAG "master" CACHE STRING "Legion repository git tag")

set(Legion_MAX_DIM ${legms_MAX_DIM} CACHE STRING "Maximum number of dimensions")
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

set(Legion_USE_LLVM OFF CACHE BOOL "Use LLVM JIT operations")
list(APPEND Legion_ARGS Legion_USE_LLVM)

set(Legion_USE_HDF5 ${legms_USE_HDF5} CACHE BOOL "Enable support for HDF5")
list(APPEND Legion_ARGS Legion_USE_HDF5)

set(Legion_USE_LIBDL ON CACHE BOOL "Enable support for libdl")
list(APPEND Legion_ARGS Legion_USE_LIBDL)

set(Legion_USE_HWLOC OFF CACHE BOOL "Use hwloc for topology awareness")
list(APPEND Legion_ARGS Legion_USE_HWLOC)

set(Legion_MAX_FIELDS 512 CACHE STRING "Maximum number of fields allocated to a single field space")
list(APPEND Legion_ARGS Legion_MAX_FIELDS)

set(Legion_ENABLE_TLS OFF CACHE BOOL "Enable support for TLS storage of Legion context")
list(APPEND Legion_ARGS Legion_ENABLE_TLS)

include(ExternalProject)
ExternalProject_Add(legion_build
  GIT_REPOSITORY ${legms_LEGION_GIT_REPO}
  GIT_TAG ${legms_LEGION_TAG}
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  CMAKE_ARGS -DBUILD_SHARED_LIBS=ON
  CMAKE_ARGS -DLegion_BUILD_BINDINGS=ON
  CMAKE_ARGS -DLegion_OUTPUT_LEVEL=${Legion_OUTPUT_LEVEL}
  CMAKE_ARGS -DLegion_MAX_DIM=${Legion_MAX_DIM}
  CMAKE_ARGS -DLegion_USE_OpenMP=${Legion_USE_OpenMP}
  CMAKE_ARGS -DLegion_USE_Python=${Legion_USE_Python}
  CMAKE_ARGS -DLegion_USE_CUDA=${Legion_USE_CUDA}
  CMAKE_ARGS -DLegion_USE_GASNet=${Legion_USE_GASNet}
  CMAKE_ARGS -DLegion_USE_LLVM=${Legion_USE_LLVM}
  CMAKE_ARGS -DLegion_USE_HDF5=${Legion_USE_HDF5}
  CMAKE_ARGS -DLegion_USE_LIBDL=${Legion_USE_LIBDL}
  CMAKE_ARGS -DLegion_USE_HWLOC=${Legion_USE_HWLOC}
  CMAKE_ARGS -DLegion_MAX_FIELDS=${Legion_MAX_FIELDS}
  CMAKE_ARGS -DLegion_ENABLE_TLS=${Legion_ENABLE_TLS}
  CMAKE_ARGS -DLegion_USE_PACKAGE_REGISTRY=OFF
  INSTALL_COMMAND make install && cp -R <SOURCE_DIR>/language/src <INSTALL_DIR>/share/Regent && cp -R <SOURCE_DIR>/bindings/regent <INSTALL_DIR>/include)
