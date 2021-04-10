#include "molecule.h"


namespace autode {

    // Cython needs empty constructors to initialise
    Molecule::Molecule() = default;

    // Overloaded constructor
    Molecule::Molecule(std::vector<double> coords,
                       std::vector<bool> bonds){
        /* autodE Molecule
         *
         * Arguments:
         *    coords: Coordinates as a flat vector 3*n_atoms with the structure
         *            x0, y0, z0, x1, y1, z1, ...
         *
         *    bonds: Existence of a bond between a pair of atoms i, j as a flat
         *           vector. i,e, n_atoms*n_atoms long in row major indexing
         */

        this->coords = std::move(coords);
        this->bonds = std::move(bonds);

        this->n_atoms = static_cast<int>(this->coords.size()) / 3;

        // Initialise a zero initial energy
        this->energy = 0.0;

        // Initialise a zero gradient vector for all components
        this->grad = std::vector<double>(3 * this->n_atoms, 0.0);

    }

}
