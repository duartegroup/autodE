#include <vector>
#include <utility>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <utility>
#include <iostream>
#include <cassert>
// TODO: remove cassert and iostream once checks are done

using std::vector;

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
                             std::vector<int>& bonds){
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
            if (S_l_p1 > S_l_m1)
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

    double vec_dot(const std::vector<double>& v1, const std::vector<double>& v2) {
        // calculate the dot product of two vectors
        assert(v1.size() == v2.size());
        return inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
    }

    double vec_norm(const vector<double>& v1) {
        // vector norm
        return std::sqrt(vec_dot(v1, v1));
    }

    void vec_sub(const vector<double>& v_in1,
                 const vector<double>& v_in2,
                 vector<double>& v_out) {
        // v_out = v_in1 - v_in2
        assert(v_in1.size() == v_in2.size() && v_in1.size() == v_out.size());
        auto n = v_in1.size();
        for (int i = 0; i < n; i++) {
            v_out[i] = v_in1[i] - v_in2[i];
        }
    }

    void vec_add(const vector<double>& v_in1,
                 const vector<double>& v_in2,
                 vector<double>& v_out) {
        // v_out = v_in1 + v_in2
        assert(v_in1.size() == v_in2.size() && v_in1.size() == v_out.size());
        auto n = v_in1.size();
        for (int i = 0; i < n; i++) {
            v_out[i] = v_in1[i] + v_in2[i];
        }
    }

    void vec_mul_scalar(vector<double>& v, double scalar) {
        // multiply a vector with a scalar number in-place
        auto n = v.size();
        for (int i = 0; i < n; i++) {
            v[i] *= scalar;
        }
    }

    double vec_diff_norm(vector<double>& v1, vector<double>& v2) {
        // np.linalg.norm(v1 - v2)
        assert(v1.size() == v2.size());
        auto n = v1.size();
        double norm = 0.0;
        for (int i = 0; i < n; i++) {
            norm += std::pow(v1[i] - v2[i], 2);
        }
        return std::sqrt(norm);
    } // is this function needed?

    class IDPP {

    public:

        vector<std::vector<double>> image_coords;  // coordinates of all images
        vector<int> target_ds;  // interpolated values of internal coords for all images
        vector<vector<double>> idpp_grads;  // idpp gradients of all images
        vector<double> idpp_energies;  // idpp energies of all images

        std::vector<double> neb_coords;  // flat neb coordinate array
        std::vector<double> neb_grad;  // flat neb gradient array
        double k_spr;  // spring constant
        int n_req_images;  // request number of images
        int n_curr_images;  // current number of images
        std::pair<int, int> frontier_images;  // innermost images
        bool use_seq;  // whether to use the S-IDPP or not

        IDPP(const std::vector<double> &init_coords,
               const std::vector<double> &final_coords,
               const double k_spr,
               const int num_images,
               const bool sequential);

        double ideal_distance() {
            // calculate the ideal interimage distance between the datapoints
            auto n = image_coords.size();
            double total_dist = 0.0;
            vector<double> deltaX;
            deltaX.resize(this->image_coords[0].size(), 0.0);

            for (int i = 0; i < n - 1; i++) {
                vec_sub(this->image_coords[i], this->image_coords[i+1], deltaX);
                total_dist += vec_norm(deltaX);
            }
            // for N images, there are N-1 segments
            return total_dist / static_cast<double>(this->n_req_images - 1);
        }

        void add_image_at(int idx) {
            // TODO: add an n_atoms?
            // add an image next to image index idx
            assert(idx == this->frontier_images.first || idx == this->frontier_images.second);
            double d_id = this->ideal_distance();
            std::cout << "Adding image, ideal distance = " << d_id << " Angstrom" << std::endl;
            int other_idx;
            if (idx == this->frontier_images.first) {
                other_idx = frontier_images.second;
            } else {
                other_idx = frontier_images.first;
            }
            // get vector from this image towards other image
            vector<double> dtoX(this->image_coords[0].size(), 0.0);
            vec_sub(this->image_coords[other_idx], this->image_coords[idx], dtoX);
            // rescale the vector to be equal in size to d_id
            vec_mul_scalar(dtoX, d_id / vec_norm(dtoX));
            vector<double> newX(this->image_coords[0].size(), 0.0);
            vec_add(this->image_coords[idx], dtoX, newX);
            if (idx == this->frontier_images.first) {
                this->image_coords.insert(this->image_coords.begin()+idx, newX);
                this->frontier_images.first += 1;
                this->frontier_images.second += 1;
            } else {
                this->image_coords.insert(this->image_coords.begin()+(idx-1), newX);
                this->frontier_images.second -= 1;
            }
            // todo update the frontier image indices check the formula works
        }

    };

    IDPP::IDPP(const std::vector<double> &init_coords,
               const std::vector<double> &final_coords,
               const double k_spr,
               const int num_images,
               const bool sequential) {
        /* Creates an IDPP object
         *
         */
        assert(init_coords.size() == final_coords.size());
        assert(init_coords.size() > 0);
        this->image_coords.push_back(init_coords);
        this->image_coords.push_back(final_coords);
        this->k_spr = k_spr;
        this->n_req_images = num_images;
        this->use_seq = sequential;
    }

    //IDPP::add_image(int idx) {
        // add an image near the index
    //}

}