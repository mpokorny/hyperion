find_package(Curses REQUIRED)

set(legms_TERRA_MP_GIT_REPO OFF CACHE BOOL "Use Martin's fork of Terra git repository")
if(legms_TERRA_MP_GIT_REPO)
  set(legms_TERRA_GIT_REPO "git@github.com:mpokorny/terra.git")
else()
  set(legms_TERRA_GIT_REPO "git@github.com:zdevito/terra.git")
endif()
set(legms_TERRA_TAG "master" CACHE STRING "Terra repository git tag")

# TODO: might be nice to pass LLVM_CONFIG to Terra's make to prevent lousy
# messages that come from trying to find various versions of llvm-config
set(REEXPORT_LLVM_COMPONENTS "irreader mcjit x86")

include(ExternalProject)
ExternalProject_Add(terra_build
  GIT_REPOSITORY ${legms_TERRA_GIT_REPO}
  GIT_TAG ${legms_TERRA_TAG}
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND make all -j 4 REEXPORT_LLVM_COMPONENTS=${REEXPORT_LLVM_COMPONENTS}
  BUILD_IN_SOURCE ON
  INSTALL_COMMAND cp -R <SOURCE_DIR>/release/bin <SOURCE_DIR>/release/include <SOURCE_DIR>/release/lib <SOURCE_DIR>/release/share <INSTALL_DIR>)
