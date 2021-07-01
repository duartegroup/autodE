#include "dihedrals.h"
#include "optimisers.h"
#include "molecule.h"
#include <cmath>
#include <vector>
#include <catch2/catch.hpp>
#include <iostream>

using namespace autode;
using namespace std;

TEST_CASE("Test a simple dihedral optimisation of eclipsed butane"){

    vector<double> coords = {-0.229521,  1.579629, -0.793611,  // x, y, z etc.
                             0.526700,  0.459300, -0.125900,
                             -0.533500, -0.461800,  0.494500,
                             -1.873700,  0.093300,  0.023300,
                             -0.309835,  1.403837, -1.900453,
                             0.203041,  2.565908, -0.562444,
                             -1.255677,  1.581283, -0.353641,
                             1.122306,  0.920508,  0.696217,
                             1.193881, -0.091622, -0.780221,
                             -0.439200, -0.372600,  1.585900,
                             -0.376500, -1.502200,  0.196300,
                             -1.770000,  0.535200, -0.997800,
                             -2.625900, -0.707300, -0.064600,
                             -2.173700,  0.918700,  0.694900};

    Molecule mol = Molecule(coords);
    REQUIRE(mol.n_atoms == 14);

    vector<bool> rotate_idxs(mol.n_atoms, false);
    vector<int> idxs_to_rotate = {0, 4, 5, 6, 1, 7, 8};
    for (size_t i: idxs_to_rotate){
        rotate_idxs[i] = true;
    }

    vector<int> axis = {1, 2};

    mol._dihedrals.emplace_back(0.0,       // angle
                                axis,
                                rotate_idxs,
                                1);      // origin

    // Pairs to consider repelling
    vector<bool> pairs(mol.n_atoms*mol.n_atoms, false);
    pairs[0 + 3] = true;
    pairs[3*mol.n_atoms + 0] = true;

    RDihedralPotential potential = RDihedralPotential(4,
                                                      pairs);
    SDDihedralOptimiser optimiser = SDDihedralOptimiser();
    optimiser.run(potential,
                  mol,
                  100,   // Maximum iterations
                  1E-5,    // Energy tolerance
                  0.1);   // initial step size


    // Distance between the end two carbons in the molecule should be > 3Å
    // for the staggered conformation of butane
    REQUIRE(mol.distance(0, 3) > 3.0);
}
