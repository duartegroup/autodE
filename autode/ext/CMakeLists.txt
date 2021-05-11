cmake_minimum_required(VERSION 3.19)
project(ade_ext)

set(CMAKE_CXX_STANDARD 11)
include_directories(include/)

add_executable(ade_ext ade_ext.cpp
               include/dihedrals.h src/dihedrals.cpp
               include/molecule.h src/molecule.cpp
               include/optimisers.h src/optimisers.cpp
               include/points.h src/points.cpp
               include/potentials.h src/potentials.cpp
               include/utils.h src/utils.cpp
               )
