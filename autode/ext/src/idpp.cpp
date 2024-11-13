#include <vector>
#include <cmath>
#include <iostream>
#include <string>

#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xnorm.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xadapt.hpp" // DEBUG only?

#include "idpp.h"
#include "utils.h"


using Array1D = xt::xtensor<double, 1>;


namespace autode {

    // Global variables
    namespace idpp_config {
        bool debug = true;  // whether to print debug messages or not
        double add_img_tol = 1.e-3;  // RMSG tolerance for adding image
        double rms_gtol = 5.e-4;  // RMSG tolerance for total path
        int lbfgs_maxvecs = 10;  // Max. number of update vectors
        int add_img_maxiter = 50;  // Max iterations for each image adding step
        int path_maxiter = 1000;  // Max iterations for total path
    }

    Image::Image(int num_atoms) : n_atoms(num_atoms) {
        /* Create an image with empty coordinates and gradients
         * for a given number of atoms
         *
         * Arguments:
         *
         *   num_atoms: number of atoms in the image
         */
        autode::utils::assert_exc(this->n_atoms > 0, "Number of atoms must be > 0");
        this->coords = xt::zeros<double>({n_atoms * 3});
        this->grad = xt::zeros<double>({n_atoms * 3});
    }

    void Image::update_neb_force(const Image& img_m1,
                                 const Image& img_p1,
                                 double k_spr) {
        /* Update the NEB force, using the previous (left) image and the next (right)
         * image and the spring constants
         *
         * Arguments:
         *
         *  img_m1: the previous image
         *
         *  img_p1: the next image
         *
         *  k_spr: the spring constant
         */
        autode::utils::assert_exc(n_atoms == img_m1.n_atoms && n_atoms == img_p1.n_atoms,
                                  "Incompatible number of atoms");

        auto tau_p = img_p1.coords - coords;
        auto tau_m = coords - img_m1.coords;

        double tau_p_norm = xt::norm_l2(tau_p)();
        double tau_m_norm = xt::norm_l2(tau_m)();
        autode::utils::assert_exc(tau_p_norm > 1e-8 && tau_m_norm > 1e-8,
                "Length of tangent vector(s) too small, something went wrong");

        auto tau_p_hat = tau_p / tau_p_norm;
        auto tau_m_hat = tau_m / tau_m_norm;

        const auto dS_max = std::max(std::abs(img_p1.en - en), std::abs(img_m1.en - en));
        const auto dS_min = std::min(std::abs(img_p1.en - en), std::abs(img_m1.en - en));

        Array1D tau;
        if (img_p1.en > en && en > img_m1.en) {
            tau = tau_p_hat;
        } else if (img_p1.en < en && en < img_m1.en) {
            tau = tau_m_hat;
        } else {
            if (img_p1.en > img_m1.en) {
                tau = tau_p_hat * dS_max + tau_m_hat * dS_min;
            } else if (img_p1.en < img_m1.en) {
                tau = tau_p_hat * dS_min + tau_m_hat * dS_max;
            } else {
                throw std::runtime_error("Something unusual has happened in energies");
            }
            tau /= xt::norm_l2(tau)();
        }

        double pre_fac = (k_spr * tau_p_norm) - (k_spr * tau_m_norm);
        // final grad = grad - (grad . tau) tau - (pre_fac) tau
        double g_dot_t = xt::sum(grad * tau)();
        grad = grad - (g_dot_t * tau) - (pre_fac * tau);
    }

    double Image::rms_g() const {
        /* Obtain the RMS gradient of this image */
        return autode::utils::rms(grad);
    }


    IDPPPotential::IDPPPotential(const xt::xtensor<double, 1>& init_coords,
                                 const xt::xtensor<double, 1>& final_coords,
                                 const int num_images) : n_images(num_images) {
        /* Create an IDPP potential
         *
         * Arguments:
         *
         *  init_coords: Initial coordinates
         *
         *  final_coords: Final coordinates
         *
         *  num_images: Number of images to interpolate
         */
        autode::utils::assert_exc(init_coords.size() == final_coords.size(),
                    "Initial and final geometries must have same number of atoms");
        autode::utils::assert_exc(init_coords.size() > 0 && init_coords.size() % 3 == 0,
                     "Wrong size of coordinates!");
        autode::utils::assert_exc(num_images > 2, "Must have more than 2 images");

        this->n_atoms = static_cast<int>(init_coords.size()) / 3;

        // calculate pairwise distances
        const int n_bonds = (n_atoms * (n_atoms - 1)) / 2;  // nC2 = n! / [ 2! (n-2)!] = [n(n-1)]/2
        Array1D init_ds = xt::zeros<double>({n_bonds});
        Array1D final_ds = xt::zeros<double>({n_bonds});
        auto init_coords_reshaped = xt::reshape_view(init_coords, {n_atoms, 3});  // reshaped
        auto final_coords_reshaped = xt::reshape_view(final_coords, {n_atoms, 3});

        size_t counter = 0;
        for (int atom_i = 0; atom_i < n_atoms; atom_i++) {
            for (int atom_j = 0; atom_j < n_atoms; atom_j++) {
                if (atom_i >= atom_j) continue;

                auto coord_i = xt::row(init_coords_reshaped, atom_i);
                auto coord_j = xt::row(init_coords_reshaped, atom_j);
                double dist = xt::norm_l2(coord_i - coord_j)();
                init_ds(counter) = dist;

                auto coord_i_2 = xt::row(final_coords_reshaped, atom_i);
                auto coord_j_2 = xt::row(final_coords_reshaped, atom_j);
                dist = xt::norm_l2(coord_i_2 - coord_j_2)();
                final_ds(counter) = dist;
                counter++;
            }
        }

        // interpolate internal coordinates (distances)
        this->all_target_ds.clear();
        for (int k = 0; k < num_images; k++) {
            // S_k = S_init + (S_fin - S_init) * k / (n-1)
            double factor = static_cast<double>(k) / static_cast<double>(num_images - 1);
            Array1D target_ds = init_ds + (final_ds - init_ds) * factor;
            this->all_target_ds.push_back(target_ds);
        }
    }

    void IDPPPotential::calc_idpp_engrad(const int idx, Image& img) const {
        autode::utils::assert_exc(idx >= 0 && idx < n_images, "Index out of bounds");
        autode::utils::assert_exc(
            img.coords.size() == n_atoms * 3 && img.grad.size() == img.coords.size(),
            "Provided image does not have correct number of atoms"
        );

        // Skip the first and last images as energy/gradient should be zero
        if (idx == 0 || idx == n_images - 1) return;

        img.en = 0.0;
        img.grad.fill(0.0);

        auto target_d_ptr = all_target_ds[idx].begin();  // pointer to items of target_ds[idx]
        auto coords_reshaped = xt::reshape_view(img.coords, {n_atoms, 3});
        auto grad_reshaped = xt::reshape_view(img.grad, {n_atoms, 3});
        for (int atom_i = 0; atom_i < n_atoms; atom_i++) {
            for (int atom_j = 0; atom_j < n_atoms; atom_j++) {
                if (atom_i >= atom_j) continue;
                auto coord_i = xt::row(coords_reshaped, atom_i);
                auto coord_j = xt::row(coords_reshaped, atom_j);
                Array1D dist_vec = coord_i - coord_j;
                double dist = xt::norm_l2(dist_vec)();
                img.en += 1.0 / std::pow(dist, 4) * std::pow(*target_d_ptr - dist, 2);

                auto grad_prefac = -2.0 * 1.0 / std::pow(dist, 4)
                                   + 6.0 * (*target_d_ptr) / std::pow(dist, 5)
                                   - 4.0 * std::pow(*target_d_ptr, 2) / std::pow(dist, 6);
                target_d_ptr++;
                // gradient terms
                xt::row(grad_reshaped, atom_i) += grad_prefac * dist_vec;
                xt::row(grad_reshaped, atom_j) -= grad_prefac * dist_vec;
            }
        }
    }

    NEB::NEB(const xt::xtensor<double, 1>& init_coords,
            const xt::xtensor<double, 1>& final_coords,
            double k_spr,
            int num_images,
            bool sequential) : n_images(num_images), k_spr(k_spr) {
        /* Initialise a NEB calculation for interpolation of path between the
         * initial and final coordinates
         *
         * Arguments:
         *   init_coords: Initial coordinates
         *   final_coords: Final coordinates
         *   k_spr: Spring constant
         *   num_images: Number of images
         *   sequential: Whether to build the path sequentially or not
         */
        autode::utils::assert_exc(n_images > 2, "Must have more than 2 images");
        autode::utils::assert_exc(k_spr > 0, "Spring constant must be positive");
        autode::utils::assert_exc(init_coords.size() == final_coords.size(),
                "Initial and final coordinates have different sizes!");
        autode::utils::assert_exc(
            init_coords.size() > 0 && init_coords.size() % 3 == 0,
            "Coordinate size must be > 0 and a multiple of 3"
        );
        this->n_atoms = static_cast<int>(init_coords.size()) / 3;
         // Create empty images and set the first and last images
        this->images.resize(n_images, Image(n_atoms));
        this->images.at(0).coords = init_coords;
        this->images.at(n_images - 1).coords = final_coords;

        // Create the potential on which to use NEB
        this->idpp_pot = IDPPPotential(init_coords, final_coords, n_images);

        if (sequential) {
            this->fill_sequentially();
        } else {
            this->fill_linear_interp();
        }
    }

    void NEB::fill_linear_interp() {
        /* Fill the NEB images with a linear interpolation */
        autode::utils::assert_exc(images.size() == n_images,
                "Image vector must be initialised with correct size!");
        for (int k = 1; k < n_images - 1; k++) {
            const auto& coords_0 = images.at(0).coords;
            const auto& coords_fin = images.at(this->n_images - 1).coords;

            double fac = static_cast<double>(k) / static_cast<double>(n_images - 1);
            auto new_coords = coords_0 + (coords_fin - coords_0) * fac;
            images.at(k).coords = new_coords;
        }
    }

    void NEB::fill_sequentially() {
        /* Fill the series of images sequentially using the IDPP potential */
        autode::utils::assert_exc(images.size() == n_images,
                "Image vector must be initialised with correct size!");
        int left_idx = 0;
        int right_idx = n_images - 1;

        int n_added = 2;
        while (n_added != n_images) {
            this->add_left(left_idx, right_idx);
            left_idx++;
            n_added++;
            if (n_added == n_images) break;
            this->add_right(left_idx, right_idx);
            right_idx--;
            n_added++;
        }
    }

    void NEB::add_right(const int left_idx, const int right_idx) {
        /* Add an image near right_idx */
        autode::utils::assert_exc(right_idx > left_idx,
                "Right index must be greater than left");
        const auto& coords_l = images.at(left_idx).coords;
        const auto& coords_r = images.at(right_idx).coords;
        auto displ_vec = (coords_l - coords_r)
                         / static_cast<double>(right_idx - left_idx);

        if (idpp_config::debug) std::cout << "Placing an image at "
                        << right_idx - 1 << " this much to the left: "
                        << xt::norm_l2(displ_vec)() << "\n";
        images.at(right_idx - 1).coords = coords_r + displ_vec;

        auto opt = LBFGSMinimiser(
            idpp_config::add_img_maxiter,
            idpp_config::add_img_tol,
            idpp_config::lbfgs_maxvecs
        );
        opt.minimise_img(images.at(right_idx - 1), right_idx - 1, idpp_pot);
    }

    void NEB::add_left(const int left_idx, const int right_idx) {
        /* Add an image to the near left_idx */
        autode::utils::assert_exc(right_idx > left_idx,
                "Right index must be greater than left");
        const auto& coords_l = images.at(left_idx).coords;
        const auto& coords_r = images.at(right_idx).coords;
        auto displ_vec = (coords_r - coords_l)
                         / static_cast<double>(right_idx - left_idx);

        if (idpp_config::debug) std::cout << "Placing an image at "
                        << left_idx + 1 << " this much to the right: "
                        << xt::norm_l2(displ_vec)() << "\n";
        images.at(left_idx + 1).coords = coords_l + displ_vec;
        auto opt = LBFGSMinimiser(
            idpp_config::add_img_maxiter,
            idpp_config::add_img_tol,
            idpp_config::lbfgs_maxvecs
        );
        opt.minimise_img(images.at(left_idx + 1), left_idx + 1, idpp_pot);
    }

}
