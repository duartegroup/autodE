#include "cmath"
#include "dihedrals.h"


namespace autode{

    Dihedral::Dihedral() = default;

    Dihedral::Dihedral(double angle,
                       std::vector<int> axis,
                       std::vector<bool> rot_idxs,
                       int origin){
        /* Single dihedral angle, e.g.
         *
         *                   CH3
         *                  /
         *         H2C ---- CH2
         *         /
         *      H3C                    Carbon atoms [0, 1, 2, 3]
         *
         *   Has: angle = π, axis = [1, 2], rot_idxs = [2, 3, ...], origin = 2
         *
         * Arguments:
         *
         *     angle: Desired change in angle for this dihedral
         *
         *     axis: Atom pair defining the axis about which to rotate around
         *
         *     rot_idxs: Booleans for all atoms in the structure, if true then
         *               will perform a rotation
         *
         *     origin: Atom index for the origin about which to rotate
         */

        this->angle = angle;
        this->axis_idxs = std::move(axis);
        this->rotate_idx = std::move(rot_idxs);
        this->origin_idx = origin;

        // Current origin in the molecule and axis (vector)
        this->origin = std::vector<double>(3, 0.0);
        this->axis = std::vector<double>(3, 0.0);

        // Flat 3x3 rotation matrix that will be updated
        this->rot_mat = std::vector<double>(9, 0.0);
    }

    void Dihedral::update_origin(const std::vector<double> &coordinates) {
        /* Update the origin in 3D space given a set of coordinates
         * and an atom index
         */
        for (int k = 0; k < 3; k++){
            origin[k] = coordinates[3*origin_idx + k];
            }
        }

    void Dihedral::update_axis(const std::vector<double> &coordinates){
        /*
         *
         */
        for (int k = 0; k < 3; k++){
            axis[k] = (coordinates[3*axis_idxs[0] + k]
                         - coordinates[3*axis_idxs[1] + k]);
            }

        }

    void Dihedral::update_rotation_matrix(){


        double norm = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
        double sin_theta = sin(angle/2.0);

        double a = cos(angle/2.0);
        double b = -axis[0] * sin_theta / norm;
        double c = -axis[1] * sin_theta / norm;
        double d = -axis[2] * sin_theta / norm;

        // Set the components of the flat rotation matrix
        rot_mat[0*3 + 0] = a*a+b*b-c*c-d*d;
        rot_mat[0*3 + 1] = 2.0*(b*c+a*d);
        rot_mat[0*3 + 2] = 2.0*(b*d-a*c);

        rot_mat[1*3 + 0] = 2.0*(b*c-a*d);
        rot_mat[1*3 + 1] = a*a+c*c-b*b-d*d;
        rot_mat[1*3 + 2] = 2.0*(c*d+a*b);

        rot_mat[2*3 + 0] = 2.0*(b*d+a*c);
        rot_mat[2*3 + 1] = 2.0*(c*d-a*b);
        rot_mat[2*3 + 2] = a*a+d*d-b*b-c*c;
    }

    void Dihedral::shift_atom_to_origin(std::vector<double> &coordinates,
                                        int atom_idx){
        // Shift coordinates to the current dihedral origin

        coordinates[3*atom_idx + 0] -= origin[0];
        coordinates[3*atom_idx + 1] -= origin[1];
        coordinates[3*atom_idx + 2] -= origin[2];
    }

    void Dihedral::shift_atom_from_origin(std::vector<double> &coordinates,
                                          int atom_idx){
        // Shift coordinates from the current dihedral origin back to the
        // previous position

        coordinates[3*atom_idx + 0] += origin[0];
        coordinates[3*atom_idx + 1] += origin[1];
        coordinates[3*atom_idx + 2] += origin[2];
    }

    void Dihedral::apply_rotation(std::vector<double> &coordinates,
                                  int atom_idx){
        /* Apply a rotation to a single atom, moving some a coordinate
         * using the current rotation matrix
         */

        double x = coordinates[3*atom_idx + 0];
        double y = coordinates[3*atom_idx + 1];
        double z = coordinates[3*atom_idx + 2];

        coordinates[3*atom_idx + 0] = (rot_mat[0*3 + 0] * x +
                                       rot_mat[0*3 + 1] * y +
                                       rot_mat[0*3 + 2] * z);

        coordinates[3*atom_idx + 1] = (rot_mat[1*3 + 0] * x +
                                       rot_mat[1*3 + 1] * y +
                                       rot_mat[1*3 + 2] * z);

        coordinates[3*atom_idx + 2] = (rot_mat[2*3 + 0] * x +
                                       rot_mat[2*3 + 1] * y +
                                       rot_mat[2*3 + 2] * z);
    }

}
