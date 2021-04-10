#include <stdexcept>
#include <utility>
#include <cmath>
#include "potentials.h"


namespace autode{

    RBPotential::RBPotential() = default;

    RBPotential::RBPotential(int rep_exponent,
                             std::vector<double> r0,
                             std::vector<double> k,
                             std::vector<double> c) {
        /*
         *   Simple repulsion + bonded (RB) potential of the form:
         *
         *  V(X) = Σ_ij k(r - r0)^2 + Σ_mn c/r^rep_exponent
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

        this->rep_exponent = rep_exponent ;
        this->r0 = std::move(r0);
        this->k = std::move(k);
        this->c = std::move(c);

    } // RBPotential

    void RBPotential::set_energy(autode::Molecule &mol) {
        // Repulsion + bonded energy

        double energy = 0.0;

        for (int i = 0; i < mol.n_atoms; i++){
            for (int j = 0; j < mol.n_atoms; j++) {

                // Only loop over unique pairs
                if (i < j) {
                    continue;
                }

                int pair_idx = i * mol.n_atoms + j;  // Compound index

                double dx = mol.coords[3*i + 0] - mol.coords[3*j + 0];
                double dy = mol.coords[3*i + 1] - mol.coords[3*j + 1];
                double dz = mol.coords[3*i + 2] - mol.coords[3*j + 1];

                // Square euclidean distance
                double r = sqrt(dx*dx + dy*dy + dz*dz);
                energy += c[pair_idx] / pow(r, rep_exponent);;

                // Check the bond matrix for if these atoms are bonded, don't
                // add a harmonic term if not
                if (! mol.bonds[pair_idx]){
                    continue;
                }

                energy += k[pair_idx] * pow(r - r0[pair_idx], 2);
            } // j
        } // i

        mol.energy = energy;
    }

    void RBPotential::set_energy_and_grad(autode::Molecule &mol) {
        // Repulsion + bonded energy

        double energy = 0.0;

        for (int i = 0; i < mol.n_atoms; i++){

            // Zero the gradient of all (x, y, z) components of this atom
            mol.grad[3*i + 0] = 0.0;
            mol.grad[3*i + 1] = 0.0;
            mol.grad[3*i + 2] = 0.0;

            for (int j = 0; j < mol.n_atoms; j++) {

                // Only loop over unique pairs
                if (i == j) {
                    continue;
                }

                int pair_idx = i * mol.n_atoms + j;  // Compound index

                double dx = mol.coords[3*i + 0] - mol.coords[3*j + 0];
                double dy = mol.coords[3*i + 1] - mol.coords[3*j + 1];
                double dz = mol.coords[3*i + 2] - mol.coords[3*j + 1];

                // Square euclidean distance
                double r = sqrt(dx*dx + dy*dy + dz*dz);

                double e_rep = c[pair_idx] / pow(r, rep_exponent);

                // Add half the energy term to prevent double counting
                energy += 0.5 * e_rep;

                // Set the component of the derivative
                auto rep_ftr = - (e_rep * static_cast<double>(rep_exponent)
                                  / pow(r, 2));

                mol.grad[3*i + 0] += rep_ftr * dx;
                mol.grad[3*i + 1] += rep_ftr * dy;
                mol.grad[3*i + 2] += rep_ftr * dz;

                // Check the bond matrix for if these atoms are bonded, don't
                // add a harmonic term if not
                if (! mol.bonds[pair_idx]){
                    continue;
                }

                // Calculate the harmonic energy
                double e_bonded = k[pair_idx] * pow(r - r0[pair_idx], 2);
                energy += 0.5 * e_bonded;

                // and set the gradient contribution from the harmonic bonds
                auto bonded_ftr = 2.0 * e_bonded / r;
                mol.grad[3*i + 0] += bonded_ftr * dx;
                mol.grad[3*i + 1] += bonded_ftr * dy;
                mol.grad[3*i + 2] += bonded_ftr * dz;

            } // j
        } // i

        mol.energy = energy;
    }

}

