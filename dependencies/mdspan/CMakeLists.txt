include(FetchContent)

set(SYSCONFDIR ${SYSCONFDIR}/hyperion)

# save some build parameters that will be changed for Legion build
set(CMAKE_BUILD_TYPE_COPY ${CMAKE_BUILD_TYPE})

# set Legion build parameters
set(CMAKE_BUILD_TYPE Release)

FetchContent_MakeAvailable(mdspan)

# restore original build parameters
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_COPY})
