set(WOODLAND_MARMOT_SRCS
  bezier_surface.cpp
  geometry.cpp
  greens_matrix.cpp
  marmot_capi.cpp
  unittest.cpp)

set(WOODLAND_MARMOT_HEADERS
  bezier_surface.hpp
  geometry.hpp
  greens_matrix.hpp
  marmot.hpp
  marmot_capi.h
  unittest.hpp)

add_library(woodland_marmot ${WOODLAND_MARMOT_SRCS})
target_link_libraries(woodland_marmot PUBLIC woodland_squirrel woodland_acorn)
target_include_directories(woodland_marmot PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../..>  # for acorn_config.h
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>  # for headers at build time
  $<INSTALL_INTERFACE:include>)                         # for headers from install

add_executable(woodland_marmot_unittest testmain.cpp)
target_link_libraries(woodland_marmot_unittest PRIVATE woodland_marmot)
add_test(NAME marmot_unittest COMMAND $<TARGET_FILE:woodland_marmot_unittest>)

install(TARGETS woodland_marmot DESTINATION lib EXPORT woodland)
install(TARGETS woodland_marmot_unittest DESTINATION bin EXPORT woodland)
install(FILES ${WOODLAND_MARMOT_HEADERS} DESTINATION include/woodland/marmot)
