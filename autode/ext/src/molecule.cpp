#include "stdexcept"
#include "cmath"
#include "molecule.h"
#include "dihedrals.h"


namespace autode {

    // Cython needs null constructors to initialise
    Molecule::Molecule() = default;

    // Overloaded constructor
    Molecule::Molecule(std::vector<double> coords){
        /* autodE Molecule
         *
         * Arguments:
         *    coords: Coordinates as a flat vector 3*n_atoms with the structure
         *            x0, y0, z0, x1, y1, z1, ...
         *
         *    bonds: Existence of a bond between a pair of atoms i, j as a flat
         *           vector. i,e, n_atoms*n_atoms long in row major indexing
         */

        if (coords.size() % 3 != 0){
            throw std::runtime_error("Coordinate must be a flat vector with "
                                     "size 3xn_atoms");
        }

        this->coords = std::move(coords);
        this->n_atoms = static_cast<int>(this->coords.size()) / 3;

        // Initialise a zero initial energy
        this->energy = 0.0;

        // Initialise a zero gradient vector for all components
        this->grad = std::vector<double>(3 * this->n_atoms, 0.0);

    }


    double Molecule::sq_distance(int i, int j) {
        /*
         * Square pairwise euclidean distance between two atoms. Be careful
         * - no bounds check!
         *
         * Arguments:
         *
         *  i: First atom index
         *
         *  j: Second atom index
         */

        double dx = coords[3*i + 0] - coords[3*j + 0];
        double dy = coords[3*i + 1] - coords[3*j + 1];
        double dz = coords[3*i + 2] - coords[3*j + 2];

        return dx*dx + dy*dy + dz*dz;
    }


    double Molecule::distance(int i, int j) {
        /*
         * Pairwise euclidean distance between two atoms. Be careful - no
         * bounds check!
         *
         * Arguments:
         *
         *  i: First atom index
         *
         *  j: Second atom index
         */

        return sqrt(sq_distance(i, j));
    }


    void Molecule::rotate(autode::Dihedral &dihedral){
        // Rotate a single dihedral angle

        dihedral.update_origin(coords);
        dihedral.update_axis(coords);
        dihedral.update_rotation_matrix();

        // Apply the rotation to only the required atoms
        for (int atom_idx=0; atom_idx < n_atoms; atom_idx++){

            if (!dihedral.rotate_idx[atom_idx]){
                continue;
            }

            dihedral.shift_atom_to_origin(coords, atom_idx);
            dihedral.apply_rotation(coords, atom_idx);
            dihedral.shift_atom_from_origin(coords, atom_idx);

        } // atoms
    }


    void Molecule::rotate(std::vector<autode::Dihedral> &dihedrals) {
        // Perform a rotation to the desired angle on each dihedral

        for (auto &dihedral: dihedrals){
            rotate(dihedral);
        }
    }


    void Molecule::zero_dihedrals() {
         // Set all the dihedral angle changes to be applied to zero

        for (auto &dihedral: _dihedrals){
            dihedral.angle = 0.0;
        }
    }


    void Molecule::rotate_dihedrals(){
        rotate(_dihedrals);
    }

}
