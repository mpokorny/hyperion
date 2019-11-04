add_test(
  NAME UnpackT0MS
  COMMAND ${CMAKE_COMMAND} -E tar Jxf ${CMAKE_CURRENT_SOURCE_DIR}/t0.ms.txz)
set_tests_properties(
  UnpackT0MS PROPERTIES
  FIXTURES_SETUP T0MS)
