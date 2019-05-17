add_custom_command(
  OUTPUT t0.ms
  COMMAND ${CMAKE_COMMAND} -E tar Jxf ${CMAKE_CURRENT_SOURCE_DIR}/t0.ms.txz
  MAIN_DEPENDENCY t0.ms.txz
  VERBATIM)
add_custom_target(
  t0_unpack ALL
  DEPENDS t0.ms)
