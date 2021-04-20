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

        // Abstract functions that must be implemented in derived classes
        virtual void set_energy(autode::Molecule &molecule) = 0;
        virtual void set_energy_and_num_grad(autode::Molecule &molecule, double eps) = 0;

        // All potentials have a gradient method that defaults to numerical
        virtual void set_energy_and_grad(autode::Molecule &molecule){
            set_energy_and_num_grad(molecule, 1E-10);
        };

    };


    class CartesianPotential: public Potential{
        // Potential energy the gradient of which is in cartesian coordinates
        void set_energy_and_num_grad(autode::Molecule &molecule,
                                     double eps) override;
    };


    class DihedralPotential: public Potential{
        // Potential energy the gradient of which is with respect to dihedral
        // rotations

        void set_energy_and_num_grad(autode::Molecule &molecule,
                                     double eps) override;
    };


    class RDihedralPotential: public DihedralPotential{
        // Simple purely repulsive (R) potential for dihedrals

    public:
        int half_rep_exponent = 0;
        std::vector<bool> rep_pairs;

        explicit RDihedralPotential();
        explicit RDihedralPotential(int rep_exponent,
                                    std::vector<bool> rep_pairs);

        void set_energy(autode::Molecule &molecule) override;

    };


    class RBPotential: public CartesianPotential{
    // Simple repulsion (R) plus bonded (B) potential

    public:
        int rep_exponent = 0;
        std::vector<bool> bonds;
        std::vector<double> r0;
        std::vector<double> k;
        std::vector<double> c;

        explicit RBPotential();
        explicit RBPotential(int rep_exponent,
                             std::vector<bool> bonds,
                             std::vector<double> r0,
                             std::vector<double> k,
                             std::vector<double> c);

        void set_energy_and_grad(autode::Molecule &molecule) override;
        void set_energy(autode::Molecule &molecule) override;

    };

}

#endif //ADE_EXT_POTENTIALS_H
