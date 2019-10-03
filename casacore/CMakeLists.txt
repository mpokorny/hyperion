set(legms_CASACORE_GIT_REPO "https://github.com/casacore/casacore.git")
set(legms_CASACORE_TAG "v3.1.2" CACHE STRING "casacore repository git tag")

set(MEASURES_URL "")

set(casacore_BUILD_PYTHON OFF CACHE BOOL "Build casacore Python bindings")
set(casacore_BUILD_PYTHON3 OFF CACHE BOOL "Build casacore Python3 bindings")
set(casacore_ENABLE_SHARED ON CACHE BOOL "Build casacore shared libraries")
set(casacore_ENABLE_RPATH ON CACHE BOOL "Include rpath in casacore executables and shared libraries")
set(casacore_ENABLE_TABLELOCKING ON CACHE BOOL "Enable casacore table locking")
set(casacore_USE_ADIOS2 OFF CACHE BOOL "Use ADIOS2 from casacore")
set(casacore_USE_HDF5 OFF CACHE BOOL "Use HDF5 from casacore")
set(casacore_USE_FFTW3 OFF CACHE BOOL "Use FFTW3 from casacore")
set(casacore_USE_THREADS ON CACHE BOOL "Support multithreading in casacore")
set(casacore_USE_OPENMP OFF CACHE BOOL "Use OpenMP from casacore")
set(casacore_USE_MPI OFF CACHE BOOL "Use MPI from casacore")
set(casacore_USE_STACKTRACE OFF CACHE BOOL "Show stacktrace in case of casacore exception")
set(casacore_CASA_DEFAULT_ALIGNMENT "32" CACHE STRING "Default alignment of casa::AlignedAllocator")
set(casacore_MODULE "measures" STRING BOOL "casacore modules to build")
set(casacore_CMAKE_BUILD_TYPE "Release" CACHE STRING "CMAKE_BUILD_TYPE definition for casacore")
set(casacore_BUILD_TESTING OFF CACHE BOOL "Build casacore test suite")
set(casacore_DATA_DIR "" CACHE STRING "Measures data directory")

include(ExternalProject)
if (casacore_DATA_DIR STREQUAL "")
  set(casacore_DATA_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/share/casacore/data"
    CACHE STRING "Measures data directory" FORCE)
  set(MEASURES_SRC_PATH "${CMAKE_CURRENT_BINARY_DIR}/src")
  set(MEASURES_TAR_PATH "${MEASURES_SRC_PATH}/WSRT_Measures.ztar")
  set(MEASURES_DOWNLOAD_CMAKE_PATH
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/download_measures_data.cmake")
  file(MAKE_DIRECTORY ${MEASURES_SRC_PATH})
  file(WRITE
    ${MEASURES_DOWNLOAD_CMAKE_PATH}
    "file(DOWNLOAD
      \"ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar\"
      \"${MEASURES_SRC_PATH}/WSRT_Measures.ztar\")")
  add_custom_command(
    OUTPUT ${MEASURES_TAR_PATH}
    COMMAND ${CMAKE_COMMAND} -P ${MEASURES_DOWNLOAD_CMAKE_PATH}
    VERBATIM)
  add_custom_command(
    OUTPUT "${casacore_DATA_DIR}/.stamp"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${casacore_DATA_DIR}
    COMMAND ${CMAKE_COMMAND} -E chdir ${casacore_DATA_DIR} tar zxf ${MEASURES_TAR_PATH}
    COMMAND ${CMAKE_COMMAND} -E touch ${casacore_DATA_DIR}/.stamp
    DEPENDS ${MEASURES_TAR_PATH}
    VERBATIM)
  add_custom_target(
    measures_data ALL
    DEPENDS "${casacore_DATA_DIR}/.stamp")
endif ()

# to define CXX_FLAGS for Legion build, add something like this to
# ExternalProject_add command:
#
# CMAKE_ARGS -DCMAKE_CXX_FLAGS=-DFOO\ -DBAR
ExternalProject_Add(casacore_build
  GIT_REPOSITORY ${legms_CASACORE_GIT_REPO}
  GIT_TAG ${legms_CASACORE_TAG}
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${casacore_CMAKE_BUILD_TYPE}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  CMAKE_ARGS -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
  CMAKE_ARGS -DBUILD_PYTHON=${casacore_BUILD_PYTHON}
  CMAKE_ARGS -DBUILD_PYTHON3=${casacore_BUILD_PYTHON3}
  CMAKE_ARGS -DENABLE_SHARED=${casacore_ENABLE_SHARED}
  CMAKE_ARGS -DENABLE_RPATH=${casacore_ENABLE_RPATH}
  CMAKE_ARGS -DENABLE_TABLELOCKING=${casacore_ENABLE_TABLELOCKING}
  CMAKE_ARGS -DUSE_ADIOS2=${casacore_USE_ADIOS2}
  CMAKE_ARGS -DUSE_HDF5=${casacore_USE_HDF5}
  CMAKE_ARGS -DUSE_FFTW3=${casacore_USE_FFTW3}
  CMAKE_ARGS -DUSE_THREADS=${casacore_USE_THREADS}
  CMAKE_ARGS -DUSE_OPENMP=${casacore_USE_OPENMP}
  CMAKE_ARGS -DUSE_MPI=${casacore_USE_MPI}
  CMAKE_ARGS -DUSE_STACKTRACE=${casacore_USE_STACKTRACE}
  CMAKE_ARGS -DCASA_DEFAULT_ALIGNMENT=${casacore_CASA_DEFAULT_ALIGNMENT}
  CMAKE_ARGS -DMODULE=${casacore_MODULE}
  CMAKE_ARGS -DDATA_DIR=${casacore_DATA_DIR}
  CMAKE_ARGS -DBUILD_TESTING=${casacore_BUILD_TESTING}
  INSTALL_COMMAND make install)

add_dependencies(casacore_build measures_data)
