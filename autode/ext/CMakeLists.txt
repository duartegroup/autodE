cmake_minimum_required(VERSION 3.16)
project(ade_ext)

set(CMAKE_CXX_STANDARD 11)
if (WIN32)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-permissive->")
endif (WIN32)

include_directories(include/)

add_library(autode STATIC
            include/dihedrals.h src/dihedrals.cpp
            include/molecule.h src/molecule.cpp
            include/optimisers.h src/optimisers.cpp
            include/points.h src/points.cpp
            include/potentials.h src/potentials.cpp
            include/utils.h src/utils.cpp
            )

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v2.13.7
)

FetchContent_MakeAvailable(Catch2)

enable_testing()

add_executable(unit_tests
               tests/test_main.cpp
               tests/test_point_gen.cpp
               tests/test_global_dihedral_min.cpp)

target_link_libraries(unit_tests PRIVATE Catch2::Catch2)
target_link_libraries(unit_tests PUBLIC autode)

add_test(test_all unit_tests)
