#ifndef ADE_EXT_POTENTIALS_H
#define ADE_EXT_POTENTIALS_H
#include "vector"
#include "molecule.h"

namespace autode {

    class Potential{

        // Potential energy function that given a function can return the
        // energy and gradient
    public:

        void check_grad(autode::Molecule &molecule, double tol = 1E-6);
        void set_energy_and_numerical_grad(autode::Molecule &molecule,
                                           double eps = 1E-10);

        // All potentials have a gradient method that defaults to numerical
        virtual void set_energy_and_grad(autode::Molecule &molecule){
            set_energy_and_numerical_grad(molecule);
        };

        // Abstract functions that must be implemented in derived classes
        virtual void set_energy(autode::Molecule &molecule) = 0;

    };

    class RBPotential: public Potential{

    public:
        int rep_exponent = 0;
        std::vector<double> r0;
        std::vector<double> k;
        std::vector<double> c;

        explicit RBPotential();
        explicit RBPotential(int rep_exponent,
                             std::vector<double> r0,
                             std::vector<double> k,
                             std::vector<double> c);

        void set_energy_and_grad(autode::Molecule &molecule) override;
        void set_energy(autode::Molecule &molecule) override;

    };

}

#endif //ADE_EXT_POTENTIALS_H
