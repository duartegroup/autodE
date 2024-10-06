#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace autode{

    void idpp_en_grad(const std::vector<double> &image_coords,
                      const std::vector<int> &bonds,
                      const std::vector<double> &target_ds,
                      std::vector<double> &idpp_grad,
                      double &idpp_en) {
        /*
         * Calculates the IDPP energy and gradient
         *
         * Arguments:
         *
         *      image_coords: Image coordinates N*3 array
         *
         *      bonds: Vector of bond indices N*(N-1)/2 array
         *
         *      target_ds: Target values of internal coordinates
         *
         *      idpp_grad: Gradient N*3 array (out)
         *
         *      idpp_en: Energy (out)
         *
         */

        idpp_en = 0;
        const int n_bonds = bonds.size() / 2;

        for (int i = 0; i < n_bonds; i += 1){

            int atom1 = bonds[i * 2];
            int atom2 = bonds[i * 2 + 1];

            double target_d = target_ds[i];

            auto u_x = image_coords[3 * atom1] - image_coords[3 * atom2];
            auto u_y = image_coords[3 * atom1 + 1] - image_coords[3 * atom2 + 1];
            auto u_z = image_coords[3 * atom1 + 2] - image_coords[3 * atom2 + 2];
            auto dist = std::sqrt(u_x * u_x + u_y * u_y + u_z * u_z);

            idpp_en += 1.0 / std::pow(dist, 4) * std::pow(target_d - dist, 2);

            auto grad_prefac = -2.0 * 1.0 / std::pow(dist, 4)
                               + 6.0 * target_d / std::pow(dist, 5)
                               - 4.0 * std::pow(target_d, 2) / std::pow(dist, 6);

            idpp_grad[3 * atom1] += grad_prefac * u_x;
            idpp_grad[3 * atom1 + 1] += grad_prefac * u_y;
            idpp_grad[3 * atom1 + 2] += grad_prefac * u_z;

            idpp_grad[3 * atom2] -= grad_prefac * u_x;
            idpp_grad[3 * atom2 + 1] -= grad_prefac * u_y;
            idpp_grad[3 * atom2 + 2] -= grad_prefac * u_z;

        }
    }

    void get_bonds_target_ds(const std::vector<double> &init_coords,
                             const std::vector<double> &final_coords,
                             const int n_images,
                             std::vector<<std::vector<double>> &all_target_ds,
                             std::vector<int> &bonds){
        /* Obtain the list of bonds and target values of those bonds
         * from the initial and final coordinates for all images.
         *
         * Arguments:
         *
         *      init_coords: Initial coordinates N*3 array
         *
         *      final_coords: Final coordinates N*3 array
         *
         *      n_images: Number of images including end points
         *
         *      target_ds: Target values of internal coordinates (out)
         *
         *      bonds: Vector of bond indices N*2 array (out)
         */
        // TODO: Implement scaled interatomic distances also
        bonds.clear();
        const int n_atoms = init_coords.size() / 3;

        // precalculate the bond list (list of pairs of atom indices)
        for (int i = 0; i < n_atoms; i++) {
            for (int j = i + 1; j < n_atoms; j++) {
                if i < j:
                    bonds.push_back(i);
                    bonds.push_back(j);
            }
        }
        // calculate the pairwise distances for initial and final coordinates
        std::vector<double> init_ds;
        std::vector<double> final_ds;
        const int n_bonds = bonds.size() / 2;
        init_ds.reserve(n_bonds);
        final_ds.reserve(n_bonds);

        for (int i = 0; i < n_bonds; i += 1) {
            int atom1 = bonds[i * 2];
            int atom2 = bonds[i * 2 + 1];
            auto u_x = init_coords[3 * atom1] - init_coords[3 * atom2];
            auto u_y = init_coords[3 * atom1 + 1] - init_coords[3 * atom2 + 1];
            auto u_z = init_coords[3 * atom1 + 2] - init_coords[3 * atom2 + 2];
            auto dist = std::sqrt(u_x * u_x + u_y * u_y + u_z * u_z);
            init_ds.push_back(dist);

            u_x = final_coords[3 * atom1] - final_coords[3 * atom2];
            u_y = final_coords[3 * atom1 + 1] - final_coords[3 * atom2 + 1];
            u_z = final_coords[3 * atom1 + 2] - final_coords[3 * atom2 + 2];
            dist = std::sqrt(u_x * u_x + u_y * u_y + u_z * u_z);
            final_ds.push_back(dist);
        }

        // linear interpolation of interatomic distances
        all_target_ds.clear();

        for (int i = 0; i < n_images; i++) {
            std::vector<double> image_ds;
            image_ds.resize(n_bonds, 0.0);

            for (int j = 0; j < n_bonds; j++) {
                image_ds[j] = init_ds[j] + (final_ds[j] - init_ds[j]) * i / (n_images - 1);
            }
            all_target_ds.push_back(image_ds);
        }
    }

    void normalise_vector(std::vector<double> &vec) {
        double norm = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0));
        for (int i = 0; i < vec.size(); i++) {
            vec[i] /= norm;
        }
    }


    void calc_neb_forces(const std::vector<std::vector<double>> &image_coords,
                         const std::vector<std::vector<double>> &image_grads,
                         const std::vector<double> &image_energies,
                         const double k_spr,
                         const int n_total_images,
                         const int n_images,
                         const int image_idx,
                         std::vector<double> &neb_grad) {
        /*
         * Calculate the NEB forces for a given image using IDPP,
         * also allowing for sequential image addition.
         *
         * Arguments:
         *
         *
         *      image_coords: Coordinates arrays of all movable images
         *
         *      image_grads: Gradient of the potential for each image
         *
         *      image_energies: Energies of each image
         *
         *      k_spr: Spring constant for NEB
         *
         *      n_total_images: Total number of images required,
         *                      including end points
         *
         *      n_images: Number of active images including end points
         *
         *      image_idx: Index of the image to calculate forces for
         *
         *      neb_grad: NEB gradients for all images, should be an
         *                n_images*N*3 array (out)
         *
         */

        // initial and final points are fixed, so ignore
        if (idx == 0 || idx == n_images - 1) return;

        // allocate arrays and alias some variables for readability
        const int len = image_coords[0].size();
        std::vector<double> tau_p(len, 0.0);
        std::vector<double> tau_m(len, 0.0);
        std::vector<double> tau;  // final normalised tangent
        tau.reserve(len);

        const std::vector<double>& r_l_p1 = image_coords[image_idx + 1];
        const std::vector<double>& r_l_m1 = image_coords[image_idx - 1];
        const std::vector<double>& r_l = image_coords[image_idx];
        const double S_l_p1 = image_energies[image_idx + 1];
        const double S_l_m1 = image_energies[image_idx - 1];
        const double S_l = image_energies[image_idx];

        // calculate the upwinding tangent, notation follows eqn. 6-9
        // in J. Chem. Theory Comput. 2024, 20, 155-163
        for (int i = 0; i < len; i++) {
            tau_p[i] = r_l_p1[i] - r_l[i];
        }
        normalise_vector(tau_p);
        for (int i = 0; i < len; i++) {
            tau_m[i] = r_l[i] - r_l_m1[i];
        }
        normalise_vector(tau_m);
        if (S_l_p1 > S_l && S_l > S_l_m1) {
            tau = tau_p;
        }
        else if (S_l_p1 < S_l && S_l < S_l_m1) {
            tau = tau_m;
        } else {
            tau.resize(len, 0.0);
            auto dS_max = std::max(std::abs(S_l_p1 - S_l), std::abs(S_l_m1 - S_l));
            auto dS_min = std::min(std::abs(S_l_p1 - S_l), std::abs(S_l_m1 - S_l));
            if (S_l_p1 > S_l_m1):
                for (int i = 0; i < len; i++) {
                    tau[i] = dS_max * tau_p[i] + dS_min * tau_m[i];
                }
            else {
                for (int i = 0; i < len; i++) {
                    tau[i] = dS_min * tau_p[i] + dS_max * tau_m[i];
                }
            }
            normalise_vector(tau);
        }


        //std::vector<double> tau;

    }

}