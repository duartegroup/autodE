#ifndef ADE_EXT_MOLECULE_H
#define ADE_EXT_MOLECULE_H
#include "vector"
#include "dihedrals.h"


namespace autode {

    class Molecule {
    public:

        int n_atoms = 0;
        double energy = 0.0;

        std::vector<autode::Dihedral> _dihedrals;

        std::vector<double> coords;
        std::vector<double> grad;

        // Constructors
        explicit Molecule();
        explicit Molecule(std::vector<double> coords);

        double sq_distance(int i, int j);
        double distance(int i, int j);

        void rotate(autode::Dihedral &dihedral);
        void rotate(std::vector<autode::Dihedral> &dihedrals);

        void zero_dihedrals();
        void rotate_dihedrals();
    };

}

#endif //ADE_EXT_MOLECULE_H
