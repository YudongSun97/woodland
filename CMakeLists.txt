cmake_minimum_required(VERSION 3.13)

project(
  woodland
  VERSION 0.0
  LANGUAGES CXX Fortran)

include(CTest)

find_package(OpenMP)
if (APPLE AND OpenMP_CXX_FOUND AND OpenMP_libomp_LIBRARY)
  # FindOpenMP links clang's runtime as @rpath/libomp.dylib. Preserve its
  # directory in installed executables so they run directly from bin/.
  get_filename_component(WOODLAND_OPENMP_CXX_LIBDIR
    "${OpenMP_libomp_LIBRARY}" DIRECTORY)
  list(APPEND CMAKE_INSTALL_RPATH "${WOODLAND_OPENMP_CXX_LIBDIR}")
endif()
if (OpenMP_CXX_FOUND)
  link_libraries(OpenMP::OpenMP_CXX)
endif()
if (OpenMP_Fortran_FOUND)
  link_libraries(OpenMP::OpenMP_Fortran)
endif()

add_subdirectory(woodland/acorn)
add_subdirectory(woodland/squirrel)

install(EXPORT woodland DESTINATION cmake)
install(FILES cmake/woodland-config.cmake DESTINATION cmake)
