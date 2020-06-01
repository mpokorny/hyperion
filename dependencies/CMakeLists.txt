include(FetchContent)

FetchContent_Declare(
  legion
  GIT_REPOSITORY https://github.com/mpokorny/legion.git
  GIT_TAG 864548544deec74f382abe0291955096868d0380) # ~legion-20.03.0
add_subdirectory(legion)

if(USE_CASACORE AND BUILD_CASACORE)
  FetchContent_Declare(
    casacore
    GIT_REPOSITORY https://github.com/casacore/casacore.git
    GIT_TAG 41122f70708f61a935d5b4647e85efefa1fd71c6) # v3.3.0
  add_subdirectory(casacore)
endif()
