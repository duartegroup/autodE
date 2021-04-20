#include "optimisers.h"
#include <cmath>


namespace autode {

    void SDOptimiser::step(autode::Molecule &molecule, double step_factor){
        // Perform a SD step in the direction of the gradient
        for (int i = 0; i < 3 * molecule.n_atoms; i++) {
            molecule.coords[i] -= step_factor * molecule.grad[i];
        }
    }


    void SDOptimiser::trust_step(autode::Molecule &molecule,
                                 double step_factor,
                                 double trust_radius){
        // Perform a SD step in the direction of the gradient with a maximum
        // absolute displacement of the trust radius
        double max_abs_delta = 0.0;

        for (int i = 0; i < 3 * molecule.n_atoms; i++) {
            max_abs_delta = std::max(max_abs_delta,
                                     std::abs(step_factor * molecule.grad[i]));
        }
        // std::cout << max_abs_delta << "\n\n\n\n" << std::endl;
        double trust_factor = std::min(trust_radius / max_abs_delta, 1.0);

        for (int i = 0; i < 3 * molecule.n_atoms; i++) {
            molecule.coords[i] -= step_factor * trust_factor * molecule.grad[i];
        }
    }

    void SDOptimiser::run(autode::Potential &potential,
                          autode::Molecule &molecule,
                          int max_iterations,
                          double energy_tol,
                          double init_step_size) {
        /*
         * Steepest decent optimiser
         *
         * Arguments:
         *     potential:
         *
         *     molecule:
         *
         *     max_iterations: Maximum number of macro iterations
         *                     (total = max_iterations * max_micro_iterations)
         *
         *     energy_tol: ΔE between iterations to signal convergence
         *
         *     init_step_size: (Å)
         */
        double max_micro_iterations = 20;
        double curr_energy = 999999999999999;
        double curr_micro_energy;

        int iteration = 0;

        while (std::abs(molecule.energy - curr_energy) > energy_tol
               and iteration <= max_iterations){

            curr_energy = molecule.energy;
            int micro_iteration = 0;
            double step_size = init_step_size;

            // Recompute the energy and gradient
            potential.set_energy_and_grad(molecule);

            // Run a set of micro iterations as a line search in the steepest
            // decent direction: -∇V
            while (micro_iteration < max_micro_iterations){

                curr_micro_energy = molecule.energy;
                step(molecule, step_size);

                potential.set_energy(molecule);

                // Backtrack if the energy rises
                if (micro_iteration == 0
                    and molecule.energy > curr_micro_energy){

                    step(molecule, -step_size);
                    molecule.energy = curr_micro_energy;
                    step_size *= 0.5;

                    continue;
                }

                if (molecule.energy > curr_micro_energy){
                    // Energy has risen but at least one step has been taken
                    // so return the previous step
                    step(molecule, -step_size);
                    break;
                }

                // Prevent very small step sizes
                if (step_size < 1E-3){
                    break;
                }

                micro_iteration += 1;
            } // Micro

            iteration += 1;
        } // Macro
    }

    void SDDihedralOptimiser::step(autode::Molecule &molecule,
                                   double step_factor) {
        /*
         * Apply a set of dihedral rotations given a new gradient
         *
         * Arguments:
         *
         *     molecule:
         *
         *     step_factor: Multiplier on the SD step
         */

        for (auto &dihedral: molecule._dihedrals){
            dihedral.angle = -step_factor * dihedral.grad;
            molecule.rotate(dihedral);
        }
    }
}