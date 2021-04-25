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

        protected:
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

        protected:
            void step(autode::Molecule &molecule, double step_factor) override;
            void trust_step(autode::Molecule &molecule,
                            double step_factor,
                            double trust_radius = 0.1);

    };


    class SDDihedralOptimiser: public SDOptimiser{
        // Steepest decent optimiser on a set of dihedral angles

        protected:
        void check_dihedrals(autode::Molecule &molecule);
        void step(autode::Molecule &molecule, double step_factor) override;
    };


    class GridDihedralOptimiser: public SDDihedralOptimiser{
        // Steepest decent optimiser

        double spacing = 0.0;

        int num_1d_points = 0;
        int num_points = 0;
        int n_angles = 0;

        std::vector<int> counter;

        public:
            void run(autode::Potential &potential,
                     autode::Molecule &molecule,
                     int max_num_points,
                     double energy_tol,
                     double init_step_size) override;

        protected:
            void apply_single_rotation(autode::Molecule &molecule, int curr_step);
            void step(autode::Molecule &molecule, double step_factor) override;
    };


    class SGlobalDihedralOptimiser: public SDDihedralOptimiser{

        public:
            void run(autode::Potential &potential,
                     autode::Molecule &molecule,
                     int max_init_points,
                     double energy_tol,
                     double init_step_size) override;

        protected:
            void step(autode::Molecule &molecule, double step_factor) override;
    };
}


#endif //ADE_EXT_OPTIMISERS_H
