#include "optimisers.h"
#include <utility>
#include <cmath>
#include <stdexcept>
#include "iostream"

namespace autode {

    void SDOptimiser::step(autode::Molecule &molecule, double step_factor){
        for (int i = 0; i < 3 * molecule.n_atoms; i++) {
            molecule.coords[i] -= step_factor * molecule.grad[i];
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
        double max_micro_iterations = 10;
        double curr_energy = 0.0;
        int iteration = 0;

        // Set an initial energy and gradient
        potential.set_energy_and_grad(molecule);

        while (std::abs(molecule.energy - curr_energy) > energy_tol
               and iteration < max_iterations){

            int micro_iteration = 0;
            double step_factor = init_step_size;

            // Run a set of micro iterations as a line search in the steepest
            // decent direction: -∇V
            while (micro_iteration < max_micro_iterations){

                curr_energy = molecule.energy;
                step(molecule, step_factor);

                potential.set_energy(molecule);

                // Backtrack if the energy rises
                if (molecule.energy > curr_energy){
                    step(molecule, -step_factor);
                    molecule.energy = curr_energy;
                    step_factor /= 2.0;
                }

                micro_iteration += 1;
            } // Micro

            iteration += 1;
        } // Macro
    }

}