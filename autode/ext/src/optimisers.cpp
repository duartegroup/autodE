#include "optimisers.h"
#include "utils.h"
#include "points.h"
#include <stdexcept>
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
        double curr_energy = 99999999.9;
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

                // Prevent very small step sizes
                if (step_size < 1E-3){
                    break;
                }

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

                micro_iteration += 1;
            } // Micro

            iteration += 1;
        } // Macro
    }


    void SDDihedralOptimiser::check_dihedrals(autode::Molecule &molecule) {
        /*  Ensure that there is at least one dihedral to minimise on
         *
         *  Arguments:
         *
         *      molecule:
         */
        if (molecule._dihedrals.empty()){
            throw std::runtime_error("Cannot run a dihedral minimisation "
                                     "without any dihedrals!");
        }
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


    void GridDihedralOptimiser::run(autode::Potential &potential,
                                    autode::Molecule &molecule,
                                    int max_num_points,
                                    double energy_tol,
                                    double init_step_size) {
        /*  Complete optimisation of the dihedral surface using a grid with
         *  a spacing defined by max_num_points^(1/n_dihedrals), plus  a final
         *  steepest decent (SD) minimisation. Can be 1D, 2D... nD grid
         *
         *  Arguments:
         *      potential:
         *
         *      molecule:
         *
         *      max_num_points: Maximum number of points in the grid, optimiser
         *                      may exceed this number of evaluations
         *
         *      energy_tol: Tolerance on the final SD minimisation
         *
         *      init_step_size: For SD
         */
        SDDihedralOptimiser::check_dihedrals(molecule);

        n_angles = static_cast<int>(molecule._dihedrals.size());

        if (n_angles == 0){
            // Nothing to be done with no dihedrals to rotate
            return;
        }

        if (n_angles > 5){
            throw std::runtime_error("Number of points required to use a "
                                     "reasonably spaced grid is huge. Not "
                                     "supported");
        }

        num_1d_points = autode::utils::fpowi(max_num_points, n_angles);
        num_points = autode::utils::powi(num_1d_points, n_angles);

        molecule.zero_dihedrals();

        // Spacing is taken over 0 -> 2π - π/3, as the dihedral is periodic
        spacing = 5.2 / static_cast<double>(num_1d_points);

        /*  Generate a counter wheel for each dihedral, which when it reaches
         *  num_1d_points then the adjacent wheel is incremented etc.
         *
         *   |                    |
         *   |  0    0    0   ... |
         *   |                    |
         *
         *   Initialised at 0 on every angle
         */
        counter = std::vector<int>(n_angles, 0);

        std::vector<double> min_coords;
        double min_energy = 99999999.9;

        for (int i=0; i < num_points; i++){

            apply_single_rotation(molecule, i);
            potential.set_energy(molecule);

            if (molecule.energy < min_energy) {
                min_coords = std::vector<double>(molecule.coords);
                min_energy = molecule.energy;
            }
        }

        molecule.coords = min_coords;

        // Perform a steepest decent optimisation from this point
        SDDihedralOptimiser::run(potential,
                                 molecule,
                                 100,
                                 energy_tol,
                                 init_step_size);

    }


    void GridDihedralOptimiser::step(autode::Molecule &molecule,
                                     double step_factor) {
        // Take a step on the grid
        SDDihedralOptimiser::step(molecule, step_factor);
    }


    void GridDihedralOptimiser::apply_single_rotation(autode::Molecule &molecule,
                                                      int curr_step) {
        /* Apply a single rotation to one dihedral in sequence, depending on
         * the value of the counter
         *
         * Arguments:
         *
         *     molecule:
         *
         *
         *     curr_step: Current step on the N-dimensional grid
         */
        for (int j=0; j < n_angles; j++){

            int value = ((curr_step
                          / autode::utils::powi(num_1d_points, j)
                         ) % num_1d_points);

            if (value == num_1d_points - 1){
                counter[j] = 0;
                continue;
            }

            if (value != counter[j]){
                molecule._dihedrals[j].angle = spacing;
                molecule.rotate(molecule._dihedrals[j]);

                counter[j] = value;
                break;
            }
        }
    }


    void SGlobalDihedralOptimiser::run(autode::Potential &potential,
                                       autode::Molecule &molecule,
                                       int max_init_points,
                                       double energy_tol,
                                       double init_step_size) {
        /* Stochastic global minimisation
         *
         * Arguments:
         *
         *     potential:
         *
         *     molecule:
         *
         *     max_init_points: Number of initial points to start SD
         *                      optimisations from, each of which is subject to
         *                      a fast (few step) optimisation
         *
         *     energy_tol: Energy tolerance on the final SD minimisation
         *
         *
         *     init_step_size: Initial step size for all (search + final)
         *                     optimisations
         */
        SDDihedralOptimiser::check_dihedrals(molecule);

        std::vector<double> min_coords;
        double min_energy = 99999999.9;

        // Generate a set of points in a n-dimensional space, where n is the
        // number of dihedrals in the system
        CubePointGenerator generator = CubePointGenerator(max_init_points, molecule._dihedrals.size());
        generator.run(1E-4, 0.01, 200);

        for (int iteration=0; iteration < max_init_points; iteration++){

            for (size_t i=0; i < molecule._dihedrals.size(); i++){
                molecule._dihedrals[i].angle = generator.points[iteration][i];
                molecule.rotate(molecule._dihedrals[i]);
            }

            // Apply a steepest decent minimisation for a few steps
            SDDihedralOptimiser::run(potential,
                                     molecule,
                                     10,
                                     1E-1,
                                     init_step_size);

            if (molecule.energy < min_energy) {
                min_coords = std::vector<double>(molecule.coords);
                min_energy = molecule.energy;
            }
        }

        // Set the minimum energy coordinates
        molecule.coords = min_coords;
    }


    void SGlobalDihedralOptimiser::step(autode::Molecule &molecule,
                                        double step_factor) {
        // Apply a steepest decent step
        SDDihedralOptimiser::step(molecule, step_factor);
    }

}