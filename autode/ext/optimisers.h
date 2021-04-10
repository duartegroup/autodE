#ifndef ADE_EXT_OPTIMISERS_H
#define ADE_EXT_OPTIMISERS_H
#include "vector"
#include "molecule.h"
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
            static void step(autode::Molecule &molecule, double step_factor);
    };
}


#endif //ADE_EXT_OPTIMISERS_H
