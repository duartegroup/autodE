#ifndef ADE_EXT_POTENTIALS_H
#define ADE_EXT_POTENTIALS_H
#include "vector"
#include "molecule.h"

namespace autode {

    class Potential{

        // Potential energy function that given a function can return the
        // energy and gradient
    public:

        // Abstract functions that must be implemented in derived classes
        virtual void set_energy_and_grad(autode::Molecule &molecule) = 0;
        virtual void set_energy(autode::Molecule &molecule) = 0;

        void check_grad(autode::Molecule &molecule, double tol = 1E-6);
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
