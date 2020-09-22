# Hyperion

[![Build Status](https://travis-ci.com/mpokorny/hyperion.svg?branch=master)](https://travis-ci.com/mpokorny/hyperion)

A prototype high performance radio astronomy data calibration, imaging and analysis project. This project provides a basis for conducting an exploration of the suitability and performance of various modern high-performance computing oriented methods, libraries, frameworks and hardware architectures for radio astronomy data imaging and analysis.

An initial implementation of [MeasurementSet](https://casa.nrao.edu/Memos/229.html) data access using the [Legion programming system](https://legion.stanford.edu/) is underway. A major short-term goal is to implement an application for the gridding of visibility data, in order to gain an understanding of the performance, flexibility, and complexity of a *Legion*-based solution for some of the significant computational challenges faced by next-generation radio telescope arrays, such as the [ngVLA](https://ngvla.nrao.edu/).

## Building the software

### Prerequisites
Several dependencies are optional, and while the software will build without them, functionality may be somewhat limited as a result. Flags used when `cmake` is invoked will determine whether a dependency is required or not. In the list below, after every optional requirement, in brackets, the *CMake* conditions that determine how a dependency is used are shown.

* Required
  * [CMake](https://cmake.org/), version 3.14 or later
  * zlib
  * git
  * git-lfs [for test data, currently no way to avoid this]
  * gcc compiler (Clang may work, but is untested)
* Optional
  * [HDF5®](https://www.hdfgroup.org/solutions/hdf5/), version 1.10.5 or later [with `USE_HDF5=ON`]
  * [LLVM](https://llvm.org/), any version acceptable to *Legion* [with `Legion_USE_LLVM=ON`]
  * [GASNet](https://gasnet.lbl.gov/), any version acceptable to *Legion* [with `Legion_USE_GASNET=ON`]
  * [yaml-cpp](https://github.com/jbeder/yaml-cpp/), version 0.6.2 or later [with `USE_YAML=ON`]
  * [casacore](https://github.com/casacore/casacore) [with `USE_CASACORE=ON`]
  * [pkgconf](https:://pkgconf.org) or [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config) [with `USE_CASACORE=ON`]
  * [CUDA](https://developer.nvidia.com/cuda-downloads), version 10.2.89 [with `USE_CUDA=ON`]
  * [Kokkos](https://github.com/kokkos/kokkos), version 3.1.1 or later [with `USE_KOKKOS=ON`]
  * [Kokkos Kernels](https://github.com/kokkos/kokkos-kernels) [with `hyperion_USE_KOKKOS_KERNELS=ON`, or `USE_KOKKOS=ON`]
  * [nvcc_wrapper](https://github.com/kokkos/nvcc_wrapper) [with `USE_CUDA=ON`, `USE_KOKKOS=ON` and gcc compiler]
  * OpenMP, provided by gcc [with `USE_OPENMP=ON`]
* Currently unimplemented, but when internal *casacore* build is again available (planned)
  * Required for internal *casacore* build [`USE_CASACORE=ON` and *casacore* auto-detection fails]
    * [GFortran](https://gcc.gnu.org/wiki/GFortran)
    * [flex](https://github.com/westes/flex)
    * [Bison](https://www.gnu.org/software/bison/)
    * [BLAS](http://www.netlib.org/blas/)
    * [LAPACK](http://www.netlib.org/lapack/)

#### In-project casacore build
**The following paragraph is not correct for the master branch of this project**. However, it describes the functionality available in a previous version, which may become available again in the near future.

The dependence on *casacore* is optional, but *hyperion* can be built using *casacore* whether or not *casacore* is already installed on your system. To build *hyperion* without any dependency on *casacore*, simply use `-DUSE_CASACORE=OFF` in the arguments to `cmake`. When `cmake` arguments include `-DUSE_CASACORE=ON` and *casacore* is found by *CMake*, *hyperion* will be built against your *casacore* installation. To provide a hint to locating *casacore*, you may use the `CASACORE_ROOT_DIR` *CMake* variable. If `-DUSE_CASACORE=ON` and no *casacore* installation is found on your system (which can also be forced by setting `-DCASACORE_ROOT_DIR=""`), *casacore* will be downloaded and built as a *CMake* external project for *hyperion*. One variable to consider using when building *casacore* through *hyperion* is `casacore_DATA_DIR`, which provides the path to an instance of the *casacore* data directory. If, however, *casacore* will be built through *hyperion* and `casacore_DATA_DIR` is left undefined, the build script will download, and install within the build directory, a recent copy of the geodetic and ephemerides data automatically. For Ubuntu systems, the required *casacore* components and the *casacore* data are available in the `casacore-dev` and `casacore-data` packages, respectively.

#### In-project Legion
*Legion* itself is always built in-project using CMake's `FetchContent` module. A few of its configuration variables have been lifted to the top level `CMakeLists.txt` file, whereas others are set from the *hyperion* configuration to ensure consistency. Other *Legion* configuration variables should be available directly when invoking `cmake`.

### Build instructions
First, clone the repository.
``` shell
$ cd /my/hyperion/source/directory
$ git clone https://github.com/mpokorny/hyperion .
```

Create a build directory, and invoke `cmake`.
``` shell
$ cd /my/hyperion/build/directory
$ cmake [CMAKE OPTIONS] /my/hyperion/source/directory
```
Note that both Unix Makefile and Ninja project file generators are known to work. If using Ninja, in the following instructions for build and test, substitute "ninja" for "make".

Build the software
``` shell
$ cd /my/hyperion/build/directory
$ make [-j N]
```

Test the software
``` shell
$ cd /my/hyperion/build/directory
$ make test
```

### Install instructions

Executable binaries are installed into `CMAKE_INSTALL_PREFIX/bin`, header files into `CMAKE_INSTALL_PREFIX/include/hyperion`, and libraries into `CMAKE_INSTALL_PREFIX/lib64`. *Legion* header files are also installed into `CMAKE_INSTALL_PREFIX/include/hyperion`, and libraries into `CMAKE_INSTALL_PREFIX/lib64/hyperion`. Note that the `RPATH` of the main *hyperion* library is set to include the directory into which the *Legion* libraries are installed.

``` shell
make install
```

## Build and install via Spack

A preliminary Spack package file for *hyperion* exists in this repository.

## Using the software

There is currently one complete application, *ms2h5*, and another, *gridder*, undergoing continuing development.

### ms2h5

This application converts a *MeasurementSet* in standard format (CTDS, or "casacore table data system") into a custom *HDF5* format. Note that the format of the *HDF5* files created by this application is almost certainly not the same as that expected by *casacore* `Table` libraries, and trying to access the *HDF5* files through those libraries will fail.

Usage: `ms2h5 [OPTION...] MS [TABLE...] OUTPUT`, where `MS` is the path to the *MeasurementSet*, and `OUTPUT` is the name of the *HDF5* file to be created. If `OUTPUT` already exists, the application will not write over or delete that file, and will exit immediately. `[TABLE...]` is a list of tables in the *MeasurementSet* to write into the *HDF5* file; if this list is not present on the command line, all tables will be written. `[OPTION...]` is a list options, primarily *Legion* and *GASNet* options. Please be aware that the argument parsing done by `ms2h5` is currently not robust; for the time being, specifying flags before options with values may ameliorate any problems.

### gridder

This application will implement a gridding code using A-projection and W-projection, with the possible extensions of FFT/IFFT and de-gridding. It is intended to measure the performance of algorithms implemented using *Legion* for the gridding of visibility data, as would be needed by currently planned radio telescope arrays. The algorithms should be very close to what would be expected to be required for a real instrument, and should not contain any computationally significant shortcuts or workarounds.

`gridder` is a work in progress, and is not yet functional.
