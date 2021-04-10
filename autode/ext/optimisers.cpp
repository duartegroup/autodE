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
         *
         */
        double max_micro_iterations = 10;
        double curr_energy = 0.0;
        int iteration = 0;

        // Set an initial energy and gradient
        potential.set_energy_and_grad(molecule);

        while (std::abs(molecule.energy - curr_energy) > energy_tol
               and iteration < max_iterations){

            int micro_iteration = 0;
            curr_energy = molecule.energy;
            double step_factor = init_step_size;

            // Run a set of micro iterations as a line search in the steepest
            // decent direction: -∇V
            while (micro_iteration < max_micro_iterations){

                step(molecule, step_factor);
                potential.set_energy(molecule);
                micro_iteration += 1;

                // Backtrack until the new energy is lower than the current
                // if the step was too large
                while (molecule.energy > curr_energy) {

                    // Σ_n=1 2^-n = 1 ! with enough steps this must work..
                    step_factor /= 2.0;
                    step(molecule, -step_factor);
                    potential.set_energy(molecule);
                    std::cout << iteration << '\t' << micro_iteration << '\t' << curr_energy << '\t' << molecule.energy << std::endl;

                    // This still counts to the number of iterations, possibly
                    // a very large number required..
                    if (micro_iteration > max_micro_iterations){
                        break;
                    }

                    micro_iteration += 1;
                }
                std::cout << iteration << '\t' << micro_iteration << '\t' << curr_energy << '\t' << molecule.energy << std::endl;


            } // Micro
        iteration += 1;

        std::cout << iteration << '\t' << curr_energy << '\t' << molecule.energy << std::endl;
        } // Macro
    }

}