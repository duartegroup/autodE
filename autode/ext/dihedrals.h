#ifndef ADE_EXT_DIHEDRALS_H
#define ADE_EXT_DIHEDRALS_H
#include "vector"

namespace autode {

    class Dihedral {
        // Single dihedral angle that can be rotated given a molecule

    public:

        double angle = 0.0;             // Value in radians
        double grad = 0.0;              // Gradient of a potential with respect
                                        // to this dihedral

        std::vector<int> axis_idxs;     // Atom pair defining the axis
        std::vector<bool> rotate_idx;   // Atom indexes that should be rotated
        int origin_idx = 0;             // Atom index of the origin atom

        std::vector<double> origin;     // Origin vector shape = (3,)
        std::vector<double> axis;       // Axis vector shape = (3,)
        std::vector<double> rot_mat;    // Rotation matrix, (3,3) -> (9,)

        explicit Dihedral();
        explicit Dihedral(double angle,
                          std::vector<int> axis,
                          std::vector<bool> rot_idxs,
                          int origin);

        void update_origin(const std::vector<double> &coordinates);

        void update_axis(const std::vector<double> &coordinates);

        void update_rotation_matrix();

        void shift_atom_to_origin(std::vector<double> &coordinates,
                                  int atom_idx);

        void shift_atom_from_origin(std::vector<double> &coordinates,
                                    int atom_idx);

        void apply_rotation(std::vector<double> &coordinates,
                            int atom_idx);
    };

}

#endif //ADE_EXT_DIHEDRALS_H
