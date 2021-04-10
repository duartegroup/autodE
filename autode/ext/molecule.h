#ifndef ADE_EXT_MOLECULE_H
#define ADE_EXT_MOLECULE_H
#include "vector"


namespace autode {

    class Molecule {
    public:

        int n_atoms = 0;
        std::vector<double> coords;
        std::vector<bool> bonds;

        double energy = 0.0;
        std::vector<double> grad;

        explicit Molecule();
        explicit Molecule(std::vector<double> coords,
                          std::vector<bool> bonds);
    };

}

#endif //ADE_EXT_MOLECULE_H
