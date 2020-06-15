include(FetchContent)

FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG 59312838eaf9bc54ae446ef1f422621051b5c311) # master 2020.06.11
add_subdirectory(mdspan)

# FetchContent_Declare(
#   kokkos
#   GIT_REPOSITORY https://github.com/kokkos/kokkos.git
#   GIT_TAG 953d7968e8fc5908af954f883e2e38d02c279cf2) # master 2020.06.12
# add_subdirectory(kokkos)

FetchContent_Declare(
  legion
  GIT_REPOSITORY https://github.com/StanfordLegion/legion.git
  GIT_TAG 42095dd56f9411e95cd9236a7f61556851906042) # master 2020.06.11
add_subdirectory(legion)

if(USE_CASACORE AND BUILD_CASACORE)
  FetchContent_Declare(
    casacore
    GIT_REPOSITORY https://github.com/casacore/casacore.git
    GIT_TAG 41122f70708f61a935d5b4647e85efefa1fd71c6) # v3.3.0
  add_subdirectory(casacore)
endif()
