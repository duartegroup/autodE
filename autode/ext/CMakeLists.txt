cmake_minimum_required(VERSION 3.16)
project(ade_ext)

set(CMAKE_CXX_STANDARD 11)
include_directories(include/)

add_library(autode STATIC
            include/dihedrals.h src/dihedrals.cpp
            include/molecule.h src/molecule.cpp
            include/optimisers.h src/optimisers.cpp
            include/points.h src/points.cpp
            include/potentials.h src/potentials.cpp
            include/utils.h src/utils.cpp
            )

find_package(Catch2 REQUIRED)
add_executable(test
               tests/test_main.cpp
               tests/test_point_gen.cpp
               tests/test_global_dihedral_min.cpp)

target_link_libraries(test PRIVATE Catch2::Catch2)
target_link_libraries(test PUBLIC autode)
