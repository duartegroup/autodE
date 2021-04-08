#ifndef ADE_EXT_EXT_H
#define ADE_EXT_EXT_H
#include "vector"

namespace autode {

    class Molecule {
        public:

            int n_atoms;
            std::vector<double> coords;
            std::vector<bool> bonds;

            double energy;
            std::vector<double> grad;


            explicit Molecule(std::vector<double> coords,
                              std::vector<bool> bonds);
    };

    class Potential{

        // Potential energy function that given a function can return the
        // energy and gradient
        public:

            // Virtual (abstract) functions that must be implemented in
            // derived classes
            virtual double energy(autode::Molecule molecule) = 0;
            virtual std::vector<double> gradient(autode::Molecule molecule) = 0;
    };

    class RBPotential: public Potential{

        public:
            int half_rep_exponent;
            std::vector<double> r0;
            std::vector<double> k;
            std::vector<double> c;

            explicit RBPotential(int rep_exponent,
                                 std::vector<double> r0,
                                 std::vector<double> k,
                                 std::vector<double> c);

            double energy(autode::Molecule mol) override;
    };

    class Optimiser{
        // Type of optimiser

        public:
            virtual void run(autode::Potential potential,
                             autode::Molecule molecule);
    };

    class SDOptimiser: public Optimiser{
        // Steepest decent optimiser
    };
}


#endif //ADE_EXT_EXT_H
