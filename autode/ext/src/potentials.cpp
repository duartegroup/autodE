#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include "potentials.h"


namespace autode{

    void CartesianPotential::set_energy_and_num_grad(autode::Molecule &mol,
                                                     double eps) {
        /*
         * Calculate a finite difference gradient
         *
         * Arguments:
         *     mol:
         *
         *     eps: δ on each x
         */
        set_energy(mol);
        auto energy = mol.energy;

        for (int i = 0; i < mol.n_atoms; i++){  // atoms
            for (int j = 0; j < 3; j++){             // x, y, z
                int idx = 3 * i + j;

                // Shift by δ in each coordinate, calculate the energy
                mol.coords[idx] += eps;
                set_energy(mol);

                // and the finite difference gradient
                mol.grad[idx] = (mol.energy - energy) / eps;

                // and shift the modified coordinates back
                mol.coords[idx] -= eps;

            } // j
        } // i
    }


    void Potential::check_grad(autode::Molecule &mol,
                               double tol) {
        /*
         * Check the analytic gradient against the numerical analogue
         *
         * Arguments:
         *    mol:
         *
         * Raises:
         *    runtime_error: If the norm is greater than tol
         */
        set_energy_and_grad(mol);

        // copy of the analytic gradient
        std::vector<double> analytic_grad(mol.grad);

        set_energy_and_num_grad(mol, 1E-10);

        double sq_norm = 0;
        for (int i = 0; i < 3 * mol.n_atoms; i++){
            sq_norm += pow(mol.grad[i] - analytic_grad[i], 2);
        }

        if (sqrt(sq_norm) > tol){
            throw std::runtime_error("Difference between the analytic and "
                                     "numerical gradients exceeded the tolerance");
        }
    }


    RDihedralPotential::RDihedralPotential() = default;
    RDihedralPotential::RDihedralPotential(int rep_exponent,
                                           std::vector<bool> rep_pairs) {
        /* Purely repulsive energy
         *
         *   V(X) = Σ_ij 1 / r_ij^rep_exponent
         *
         * Arguments:
         *     rep_exponent: Integer exponent
         *
         *     rep_pairs: Boolean row major vector (flat matrix) of atom pairs
         *                that should be considered for a pairwise repulsion
         *                should be a flat symmetric matrix, but only the
         *                upper triangle is used (column index > row index)
         */

        if (rep_exponent % 2 != 0){
            throw std::runtime_error("Repulsion exponent must be even");
        }

        this->half_rep_exponent = rep_exponent / 2;
        this->rep_pairs = std::move(rep_pairs);
    }

    void RDihedralPotential::set_energy(autode::Molecule &mol) {
        // Repulsion energy

        mol.energy = 0.0;

        for (int i = 0; i < mol.n_atoms; i++){
            for (int j = i+1; j < mol.n_atoms; j++) {

                if (rep_pairs[i * mol.n_atoms + j]){

                    if (half_rep_exponent == 1){
                        mol.energy += 1.0 / mol.sq_distance(i, j);
                    }
                    else {
                        mol.energy += 1.0 / pow(mol.sq_distance(i, j),
                                                half_rep_exponent);
                    }
                }

            } // j
        } // i
    }


    void DihedralPotential::set_energy_and_num_grad(autode::Molecule &mol,
                                                     double eps) {
        set_energy(mol);
        auto energy = mol.energy;

        for (auto &dihedral : mol._dihedrals){

            // Shift by δ on a dihedral
            dihedral.angle = eps;
            mol.rotate(dihedral);
            set_energy(mol);

            // calculate the finite difference gradient
            dihedral.grad = (mol.energy - energy) / eps;

            // and shift the modified coordinates back
            dihedral.angle = -eps;
            mol.rotate(dihedral);

        } // i
    }


    RRingDihedralPotential::RRingDihedralPotential() = default;
    RRingDihedralPotential::RRingDihedralPotential(int rep_exponent,
                                                   std::vector<bool> rep_pairs,
                                                   std::vector<int> close_pair,
                                                   double close_distance) {
        /* Repulsive dihedral potential in a ring e.g.::
         *
         *     X       Z
         *     |      /
         *     Y---- W
         *
         *  for a 4 membered ring with a single dihedral (X, Y, W, Z)
         *
         *       V(x) = 1/n Σ_ij 1 / r_ij^rep_exponent + (r_XZ - r0)^2
         *
         *  Arguments:
         *
         *      rep_exponent:
         */

        if (rep_exponent % 2 != 0){
            throw std::runtime_error("Repulsion exponent must be even");
        }

        this->half_rep_exponent = rep_exponent / 2;
        this->rep_pairs = std::move(rep_pairs);

        if (close_pair.size() != 2){
            throw std::runtime_error("Must have a pair of atoms that close a "
                                     "ring");
        }

        if ((close_distance < 0.5) || (close_distance > 3.0)){
            throw std::runtime_error("Had an unexpected distance between two "
                                     "atoms that close a ring");
        }

        // Sort the pair so i <= j
        std::sort(close_pair.begin(), close_pair.end());

        this->close_idx_i = close_pair[0];
        this->close_idx_j = close_pair[1];
        this->close_distance = close_distance;
    }

    void RRingDihedralPotential::set_energy(autode::Molecule &mol) {
        // Energy for repulsion + closure

        mol.energy = 0.0;
        double c = 1.0 / mol.n_atoms;

        for (int i = 0; i < mol.n_atoms; i++){
            for (int j = i+1; j < mol.n_atoms; j++) {

                // 1 / r_ij^rep_exponent
                if (rep_pairs[i * mol.n_atoms + j]){
                    mol.energy += c / pow(mol.sq_distance(i, j), half_rep_exponent);
                }

                // (r_ij - r0)^2
                if (i == close_idx_i && j == close_idx_j){
                    mol.energy += pow(mol.distance(i, j) - close_distance, 2);
                }

            } // j
        } // i
    }

    RBPotential::RBPotential() = default;

    RBPotential::RBPotential(int rep_exponent,
                             std::vector<bool> bonds,
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
         *      Bonds: Row major vector of the existence of a 'bond' for each
         *             pairwise interation
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
        this->bonds = std::move(bonds);
        this->r0 = std::move(r0);
        this->k = std::move(k);
        this->c = std::move(c);

    } // RBPotential

    void RBPotential::set_energy(autode::Molecule &mol) {
        // Repulsion + bonded energy

        double energy = 0.0;

        for (int i = 0; i < mol.n_atoms; i++){
            for (int j = i+1; j < mol.n_atoms; j++) {

                int pair_idx = i * mol.n_atoms + j;  // Compound index

                // Square euclidean distance
                double r = mol.distance(i, j);
                energy += c[pair_idx] / pow(r, rep_exponent);;

                // Check the bond matrix for if these atoms are bonded, don't
                // add a harmonic term if not
                if (! bonds[pair_idx]){
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

                // Only loop over non identical pairs
                if (i == j) {
                    continue;
                }

                int pair_idx = i * mol.n_atoms + j;  // Compound index

                double dx = mol.coords[3*i + 0] - mol.coords[3*j + 0];
                double dy = mol.coords[3*i + 1] - mol.coords[3*j + 1];
                double dz = mol.coords[3*i + 2] - mol.coords[3*j + 2];

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
                if (! bonds[pair_idx]){
                    continue;
                }

                // Calculate the harmonic energy
                energy += 0.5 * k[pair_idx] * pow(r - r0[pair_idx], 2);

                // and set the gradient contribution from the harmonic bonds
                auto bonded_ftr = 2.0 * k[pair_idx] * (1.0 - r0[pair_idx] / r);
                mol.grad[3*i + 0] += bonded_ftr * dx;
                mol.grad[3*i + 1] += bonded_ftr * dy;
                mol.grad[3*i + 2] += bonded_ftr * dz;

            } // j
        } // i

        mol.energy = energy;
    }

}

