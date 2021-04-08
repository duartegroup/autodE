#include "ext.h"
#include <utility>
#include <cmath>
#include <stdexcept>

namespace autode {

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

    } // Molecule

    RBPotential::RBPotential(int rep_exponent,
                             std::vector<double> r0,
                             std::vector<double> k,
                             std::vector<double> c) {
        /*
         *   Simple repulsion + bonded (RB) potential of the form:
         *
         *  V(X) = Σ_ij k(r - r0)^2 + Σ_mn c/r^half_rep_exponent
         *
         *  for a pair (i, j) in bonds and (m, n) in non-bonded repulsion
         *  term
         *
         *  Arguments:
         *      rep_exponent: Power of the repulsive r^-n term
         *
         *      r0: Row major vector of equilibrium distances. Only used for
         *          bonded pairs
         *
         *      k: Row major vector of force constants for the harmonic terms.
         *         Only used for bonded pairs
         *
         *      c: Row major vector of the repulsion coefficient. Applies to
         *         all unique pairs
         */
        if (rep_exponent % 2 != 1){
            throw std::runtime_error("Only even powers for the repulsive "
                                     "component of an RB energy are supported");
        }

        this->half_rep_exponent = rep_exponent / 2;
        this->r0 = std::move(r0);
        this->k = std::move(k);
        this->c = std::move(c);

    } // RBPotential

    double RBPotential::energy(autode::Molecule mol) {
        // Repulsion + bonded energy

        double energy = 0.0;
        double r_sq;            // r^2
        int pair_idx;           // Compound index of an i,j pair in a flat list

        for (int i = 0; i < mol.n_atoms; i++){
            for (int j = 0; j < mol.n_atoms; j++) {

                // Only loop over unique pairs
                if (i < j) {
                    continue;
                }

                pair_idx = i * mol.n_atoms + j;

                // Square euclidean distance
                r_sq = (pow(mol.coords[3*i + 0] - mol.coords[3*j + 0], 2) +
                        pow(mol.coords[3*i + 1] - mol.coords[3*j + 1], 2) +
                        pow(mol.coords[3*i + 2] - mol.coords[3*j + 1], 2));

                energy += c[pair_idx] / pow(r_sq, half_rep_exponent);

                // Check the bond matrix for if these atoms are bonded, don't
                // add a harmonic term if not
                if (! mol.bonds[pair_idx]){
                    continue;
                }

                energy += k[pair_idx] * pow(sqrt(r_sq) - r0[pair_idx], 2);

            } // j
        } // i

        return energy;
    }


}