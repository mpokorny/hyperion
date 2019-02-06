find_package(Curses REQUIRED)

set(legms_TERRA_TAG "master" CACHE STRING "Terra repository git tag")

set(CURSES ${CURSES_LIBRARIES})
list(TRANSFORM CURSES REPLACE "/" "\\\\/")
set(LCURSES)
string(REPLACE ";" " " LCURSES "${CURSES}")
set(PATCH_LCURSES "s/SUPPORT_LIBRARY_FLAGS += -lcurses/SUPPORT_LIBRARY_FLAGS += ${LCURSES}/g")

# TODO: might be nice to pass LLVM_CONFIG to Terra's make to prevent lousy
# messages that come from trying to find various versions of llvm-config
include(ExternalProject)
ExternalProject_Add(terra_build
  GIT_REPOSITORY git@github.com:zdevito/terra.git
  GIT_TAG ${legms_TERRA_TAG}
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND make all -j 4 REEXPORT_LLVM_COMPONENTS="irreader mcjit x86"
  BUILD_IN_SOURCE ON
  PATCH_COMMAND sed -i ${PATCH_LCURSES} <SOURCE_DIR>/Makefile
  INSTALL_COMMAND cp -R <SOURCE_DIR>/release/bin <SOURCE_DIR>/release/include <SOURCE_DIR>/release/lib <SOURCE_DIR>/release/share <INSTALL_DIR>)
