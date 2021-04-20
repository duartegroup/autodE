#ifndef ADE_EXT_OPTIMISERS_H
#define ADE_EXT_OPTIMISERS_H
#include "vector"
#include "potentials.h"


namespace autode {


    class Optimiser{
        // Type of optimiser

        public:
            // Abstract functions
            virtual void run(autode::Potential &potential,
                             autode::Molecule &molecule,
                             int max_iterations,
                             double energy_tol,
                             double init_step_size) = 0;

            virtual void step(autode::Molecule &molecule,
                              double step_factor) = 0;
    };

    class SDOptimiser: public Optimiser{
        // Steepest decent optimiser

        public:
            void run(autode::Potential &potential,
                     autode::Molecule &molecule,
                     int max_iterations,
                     double energy_tol,
                     double init_step_size) override;

        private:
            void step(autode::Molecule &molecule, double step_factor) override;
            void trust_step(autode::Molecule &molecule,
                            double step_factor,
                            double trust_radius = 0.1);

    };


    class SDDihedralOptimiser: public SDOptimiser{
        // Steepest decent optimiser on a set of dihedral angles

        private:
            void step(autode::Molecule &molecule, double step_factor) override;
    };
}


#endif //ADE_EXT_OPTIMISERS_H
