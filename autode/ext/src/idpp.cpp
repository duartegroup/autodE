#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "arrayhelper.hpp"
#include "idpp.h"

// global variables for idpp
namespace {
    inline void ensure(const bool condition, const char* message) {
        /* Check an assertion and raise an exception if it is not true.
         * Intended as a replacement for the C assert macro - which is
         * always disabled by Cythons. Defined in an anonymous namespace
         * as it is only used in this source file.
         *
         *  Arguments:
         *
         *    condition: The condition to check
         *
         *    message: The message in the exception if condition is false
         */
        if (! condition) throw std::runtime_error(message);
    }

    bool debug_pr; // whether to print debug messages or not
}

namespace autode {

    Image::Image(int num_atoms, double k)
     : n_atoms(num_atoms), k_spr(k) {
        /* Create an image with empty coordinates and gradients
         * for a given number of atoms
         *
         * Arguments:
         *
         *   num_atoms: number of atoms in the image
         *
         *   k: spring constant
         */
        ensure(num_atoms > 0, "Number of atoms must be > 0");
        ensure(k > 0, "Spring constant must be positive");
        coords = arrx::zeros(n_atoms * 3);
        grad = arrx::zeros(n_atoms * 3);
    }

    double Image::get_tau_k_fac(arrx::array1d& tau,
                                const Image& img_m1,
                                const Image& img_p1,
                                const bool force_lc) const {
        /* Get the unit tangent and the pre-factor required to calculate
         * the spring force, from the previous and next image
         *
         * Arguments:
         *
         *   tau: (out) The array where the tangent is stored
         *
         *   img_m1: Previous image
         *
         *   img_p1: Next image
         *
         *   force_lc: Whether to force using linear combination tangent
         *             even if image is not extrema in energy
         */

        auto dv_max = std::max(
            std::abs(img_p1.en - en),
            std::abs(img_m1.en - en)
        );
        auto dv_min = std::min(
            std::abs(img_p1.en - en),
            std::abs(img_m1.en - en)
        );

        arrx::array1d tau_p = img_p1.coords - coords;
        arrx::array1d tau_m = coords - img_m1.coords;

        double tau_p_norm = arrx::norm_l2(tau_p);
        double tau_m_norm = arrx::norm_l2(tau_m);

        ensure(tau_p_norm > 1e-10 && tau_m_norm > 1e-10,
                "Failed to get tangent, images too close");

        tau_p /= tau_p_norm;
        tau_m /= tau_m_norm;

        bool use_lc = force_lc;
        if (force_lc) {

        } else if (img_m1.en < en && en < img_p1.en) {
            tau = tau_p;
        } else if (img_p1.en < en && en < img_m1.en) {
            tau = tau_m;
        } else {
            use_lc = true;
        }

        if (use_lc) {
            if (img_m1.en < img_p1.en) {
                tau = tau_p * dv_max + tau_m * dv_min;
            } else if (img_p1.en < img_m1.en) {
                tau = tau_m * dv_max + tau_p * dv_min;
            } else {  // equal energies
                tau = tau_p + tau_m;
            }
            double tau_norm = arrx::norm_l2(tau);
            if (tau_norm < 1e-10) {  // sometimes dv_max, dv_min are small
                tau = tau_p + tau_m;
                tau_norm = arrx::norm_l2(tau);
                ensure(tau_norm > 1e-10, "Tangent vector too small");
            }
            tau /= tau_norm;
        }
        return tau_p_norm * img_p1.k_spr - tau_m_norm * img_m1.k_spr;
    }

    void Image::update_neb_grad(const Image& img_m1,
                                const Image& img_p1) {
        /* Update the NEB force, using the previous image and the next
         * image
         *
         * Arguments:
         *
         *  img_m1: the previous image
         *
         *  img_p1: the next image
         */
        arrx::array1d tau_hat;
        auto k_fac = get_tau_k_fac(tau_hat, img_m1, img_p1, false);

        auto f_par = tau_hat * k_fac;
        auto g_perp = grad - arrx::dot(grad, tau_hat) * tau_hat;

        arrx::noalias(grad) = g_perp - f_par;
    }

    double Image::max_g() const {
        /* Obtain the max. abs. gradient of this image */
        return arrx::abs_max(grad);
    }


    IDPPotential::IDPPotential(const arrx::array1d& init_coords,
                                 const arrx::array1d& final_coords,
                                 const int num_images)
     : n_images(num_images) {
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
        ensure(init_coords.size() == final_coords.size(),
                "Initial and final geometries must have same number of atoms");
        ensure(init_coords.size() > 0 && init_coords.size() % 3 == 0,
                "Wrong size of coordinates!");
        ensure(num_images > 2, "Must have more than 2 images");
        n_atoms = static_cast<int>(init_coords.size()) / 3;

        // calculate pairwise distances
        const int n_dists = (n_atoms * (n_atoms - 1)) / 2;  // nC2 = [n(n-1)]/2
        arrx::array1d init_ds = arrx::zeros(n_dists);
        arrx::array1d final_ds = arrx::zeros(n_dists);

        size_t counter = 0;
        for (int atom_i = 0; atom_i < n_atoms; atom_i++) {
            for (int atom_j = 0; atom_j < n_atoms; atom_j++) {
                if (atom_i >= atom_j) continue;

                auto coord_i = arrx::slice(
                    init_coords, atom_i * 3, atom_i * 3 + 3
                );
                auto coord_j = arrx::slice(
                    init_coords, atom_j * 3, atom_j * 3 + 3
                );
                double dist = arrx::norm_l2(coord_i - coord_j);
                init_ds[counter] = dist;

                auto coord_i_2 = arrx::slice(
                    final_coords, atom_i * 3, atom_i * 3 + 3
                );
                auto coord_j_2 = arrx::slice(
                    final_coords, atom_j * 3, atom_j * 3 + 3
                );
                dist = arrx::norm_l2(coord_i_2 - coord_j_2);
                final_ds[counter] = dist;
                counter++;
            }
        }

        // interpolate internal coordinates (distances)
        all_target_ds.clear();
        for (int k = 0; k < num_images; k++) {
            // S_k = S_init + (S_fin - S_init) * k / (n-1)
            double factor = static_cast<double>(k)
                            / static_cast<double>(num_images - 1);
            arrx::array1d target_ds = init_ds + (final_ds - init_ds) * factor;
            all_target_ds.push_back(target_ds);
        }
    }

    void IDPPotential::calc_idpp_engrad(const int idx, Image& img) const {
        /* Calculate the IDPP energy/gradient for the
         * supplied image
         *
         * Arguments:
         *   idx: The index of the image
         *
         *   img: The image for which to calculate the energy/grad.
         *        The image object is modified in-place
         */

        if (idx <= 0 || idx >= n_images - 1) return;

        img.en = 0.0;
        img.grad.fill(0.0);

        // pointer to items of target_ds[idx]
        auto target_d_ptr = all_target_ds[idx].begin();

        arrx::array1d dist_vec;
        for (int atom_i = 0; atom_i < n_atoms; atom_i++) {
            for (int atom_j = 0; atom_j < n_atoms; atom_j++) {
                if (atom_i >= atom_j) continue;
                auto coord_i = arrx::slice(
                    img.coords, atom_i * 3, atom_i * 3 + 3
                );
                auto coord_j = arrx::slice(
                    img.coords, atom_j * 3, atom_j * 3 + 3
                );
                arrx::noalias(dist_vec) = coord_i - coord_j;
                double dist = arrx::norm_l2(dist_vec);
                double dist_pow_2 = dist * dist;
                double dist_pow_4 = dist_pow_2 * dist_pow_2;
                double dist_pow_5 = dist_pow_4 * dist;
                double dist_pow_6 = dist_pow_5 * dist;

                // energy terms
                img.en += 1.0 / dist_pow_4 * std::pow(*target_d_ptr - dist, 2);

                auto grad_prefac = -2.0 * 1.0 / dist_pow_4
                        + 6.0 * (*target_d_ptr) / dist_pow_5
                        - 4.0 * std::pow(*target_d_ptr, 2) / dist_pow_6;
                // gradient terms
                dist_vec *= grad_prefac;
                arrx::slice(img.grad, atom_i * 3, atom_i * 3 + 3) += dist_vec;
                arrx::slice(img.grad, atom_j * 3, atom_j * 3 + 3) -= dist_vec;
                target_d_ptr++;
            }
        }
    }

    NEB::NEB(arrx::array1d init_coords,
             arrx::array1d final_coords,
             double k,
             int num_images)
     : n_images(num_images), k_spr(k) {
        /* Initialise a NEB calculation for interpolation of path between the
         * initial and final coordinates
         *
         * Arguments:
         *   init_coords: Initial coordinates (will move the array)
         *   final_coords: Final coordinates (will move the array)
         *   k_spr: Spring constant
         *   num_images: Number of images
         *   sequential: Whether to build the path sequentially or not
         */
        ensure(n_images > 2, "Must have more than 2 images");
        ensure(k_spr > 0, "Spring constant must be positive");
        ensure(init_coords.size() == final_coords.size(),
                "Initial and final coordinates have different sizes!");
        ensure(init_coords.size() > 0 && init_coords.size() % 3 == 0,
            "Coordinate size must be > 0 and a multiple of 3");
        n_atoms = static_cast<int>(init_coords.size()) / 3;
        images.resize(n_images, Image(n_atoms, k_spr));

        // Check if the end points are too close
        if (arrx::norm_l2(init_coords - final_coords) < 0.1) {
            throw std::runtime_error(
                "Initial and final coordinates are too close!"
            );
        }
        // Create the end point images
        images.at(0).coords = std::move(init_coords);
        images.at(n_images - 1).coords = std::move(final_coords);
    }

    void NEB::fill_linear_interp() {
        /* Fill the NEB images with a linear interpolation */
        const auto& coords_0 = images.at(0).coords;
        const auto& coords_fin = images.at(n_images - 1).coords;
        for (int k = 1; k < n_images - 1; k++) {
            double fac = static_cast<double>(k)
                        / static_cast<double>(n_images - 1);
            auto new_coords = coords_0 + (coords_fin - coords_0) * fac;
            images.at(k).coords = new_coords;
        }
        reset_k_spr();
        images_prepared = true;
    }

    void NEB::fill_sequentially(const IDPPotential& pot,
                                const int add_maxiter,
                                const double add_maxgtol) {
        /* Fill the NEB path sequentially */
        ensure(n_images > 3, "Number of images must be > 3");
        add_first_two_images();
        int n_added = 4;

        while (n_added <= n_images) {
            auto opt = BBMinimiser(add_maxiter, add_maxgtol);
            auto conv_idx = opt.min_frontier(*this, frontier, pot);
            if (n_added == n_images) break;
            add_image_next_to(conv_idx);
            n_added++;
        }
        reset_k_spr();
        images_prepared = true;
    }

    double NEB::get_d_id() const {
        /* Obtain the ideal distance between images at current stage */
        double dist = 0.0;
        for (int k = 0; k < frontier.left; k++) {
            dist += arrx::norm_l2(images[k].coords - images[k+1].coords);
        }
        double d_mid = arrx::norm_l2(
            images[frontier.left].coords - images[frontier.right].coords
        );
        dist += d_mid;
        for (int k = frontier.right; k < n_images-1; k++) {
            dist += arrx::norm_l2(images[k].coords - images[k+1].coords);
        }
        return dist / static_cast<double>(n_images - 1);
    }

    double NEB::get_k_mid() const {
        /* Get the force constant between the frontier images */
        double d_mid = arrx::norm_l2(
            images[frontier.left].coords - images[frontier.right].coords
        );
        return k_spr * get_d_id() / d_mid;
    }

    void NEB::reset_k_spr() {
        /* Reset the spring constant to base value for all images */
        for (int k = 0; k < n_images; k++) images[k].k_spr = k_spr;
    }

    void NEB::get_engrad(double& en, arrx::array1d& grad) const {
        /* Obtain the average energy, and gradient of all images excluding the
         * initial and final images
         *
         * Arguments:
         *   en: (out) Average energy of the NEB
         *   grad: (out) Array where the NEB gradient will be stored.
         *          Must be of shape (n_images - 2) * n_atoms * 3.
         */
        //size_t dim = (n_images - 2) * n_atoms * 3;  // required dimension
        //ensure(grad.size() == dim, "Array is the wrong size");

        double total_en = 0.0;
        size_t loc = 0;
        for (int k = 1; k < n_images - 1; k++) {
            total_en += images[k].en;
            arrx::slice(grad, loc, loc + n_atoms * 3) = images[k].grad;
            loc += (n_atoms * 3);
        }
        en = total_en / static_cast<double>(n_images);
    }

    void NEB::get_frontier_engrad(double& en, arrx::array1d& grad) const {
        /* Obtain the average energy and the gradients for the
         * frontier indexes
         *
         * Arguments:
         *   en: (out) Average energy of image
         *   grad: (out) Array where the gradient will be
         *          stored. Must be of shape 2 * n_atoms * 3
         */
        size_t img_dim = n_atoms * 3; // dimension for one image
        //ensure(grad.size() == img_dim * 2, "Array is the wrong size");

        en = (images[frontier.left].en + images[frontier.right].en) / 2.0;
        arrx::slice(grad, 0, img_dim) = images[frontier.left].grad;
        arrx::slice(grad, img_dim, img_dim * 2) = images[frontier.right].grad;
    }

    void NEB::get_coords(arrx::array1d& flat_coords) const {
        /* Obtain the coordinates of all images excluding the initial
         * and final images
         *
         * Arguments:
         *   flat_coords: Array where the coordinates will be stored
         *                Must be of shape (n_images - 2) * n_atoms * 3
         */
        //size_t dim = (n_images - 2) * n_atoms * 3;  // required dimension
        //ensure(flat_coords.size() == dim, "Array is the wrong size");

        size_t loc = 0;
        for (int k = 1; k < n_images - 1; k++) {
            arrx::slice(flat_coords, loc, loc + n_atoms * 3) = images[k].coords;
            loc += (n_atoms * 3);
        }
    }

    void NEB::get_frontier_coords(arrx::array1d& coords) const {
        /* Obtain the coordinates of the frontier images
         *
         * Arguments:
         *   coords: (out) Array where the coordinates will be stored
         *                Must be of shape 2 * n_atoms * 3
         */
        size_t img_dim = n_atoms * 3;
        //ensure(coords.size() == img_dim * 2, "Array is the wrong size");

        arrx::slice(coords, 0, img_dim) = images.at(frontier.left).coords;
        arrx::slice(coords, img_dim, img_dim * 2)
                                        = images.at(frontier.right).coords;
    }

    void NEB::set_coords(const arrx::array1d& coords) {
        /* Set the coordinates of all images excluding the intial
         * and final images from a flat array
         *
         * Arguments:
         *   coords: Array of coordinates (must have correct size)
         */
        size_t loc = 0;
        for (int k = 1; k < n_images - 1; k++) {
            arrx::noalias(images[k].coords) =
                    arrx::slice(coords, loc, loc + n_atoms * 3);
            loc += (n_atoms * 3);
        }
    }

    void NEB::set_frontier_coords(const arrx::array1d& coords) {
        /* Set the coordinates for the frontier images from a flat array
         *
         * Arguments:
         *   coords: Array of coordinates (must have correct size)
         */
        arrx::noalias(images.at(frontier.left).coords)
                                        = arrx::slice(coords, 0, n_atoms * 3);
        arrx::noalias(images.at(frontier.right).coords)
                                = arrx::slice(coords, n_atoms * 3, n_atoms * 6);
    }

    void NEB::add_first_two_images() {
        /* Add the first two images next to both end points for S-IDPP */
        if (debug_pr) std::cout << "Adding images at "
                                    << 1 << " and " << n_images - 2 << "\n";
        const auto& coords_0 = images.at(0).coords;
        const auto& coords_fin = images.at(n_images - 1).coords;

        double fac1 = 1.0 / static_cast<double>(n_images-1);
        images.at(1).coords = coords_0 + (coords_fin - coords_0) * fac1;
        double facn_2 = static_cast<double>(n_images-2)
                        / static_cast<double>(n_images-1);
        images.at(n_images-2).coords = coords_0
                                    + (coords_fin - coords_0) * facn_2;
        frontier.left = 1;
        frontier.right = n_images - 2;
        // set force constants
        auto k_mid = get_k_mid();
        for (int k = 0; k < n_images; k++) {
            images[k].k_spr
                  = (k == frontier.left || k == frontier.right) ? k_mid : k_spr;
        }
    }

    void NEB::add_image_next_to(const int idx) {
        /* Add a new frontier image next to index idx. This idx must point
         * to one of the current frontier images
         *
         * Arguments:
         *   idx: The index next to which the image will be added
         */
        ensure(idx == frontier.left || idx == frontier.right,
                                                        "Wrong index supplied");

        double d_id = get_d_id();
        if (idx == frontier.left) {
            if (debug_pr) std::cout  << "+++ Placing new image at "
                                                              << idx+1 << "\n";
            arrx::array1d tau;
            images[frontier.left].get_tau_k_fac(
                tau, images[frontier.left-1], images[frontier.right], true
            );
            arrx::noalias(images[frontier.left+1].coords) =
                images[frontier.left].coords + tau * (d_id/arrx::norm_l2(tau));
            frontier.left++;
        } else {
            if (debug_pr) std::cout  << "+++ Placing new image at "
                                                              << idx-1 << "\n";
            arrx::array1d tau;
            images[frontier.right].get_tau_k_fac(
                tau, images[frontier.left], images[frontier.right+1], true
            );
            arrx::noalias(images[frontier.right-1].coords) =
                images[frontier.right].coords - tau * (d_id/arrx::norm_l2(tau));
            frontier.right--;
        }
        // set force constants
        auto k_mid = get_k_mid();
        for (int k = 0; k < n_images; k++) {
            images[k].k_spr
                  = (k == frontier.left || k == frontier.right) ? k_mid : k_spr;
        }
    }

    BBMinimiser::BBMinimiser(int max_iter, double tol)
    : maxiter(max_iter), gtol(tol) {
        /* Create a Barzilai-Borwein minimiser
         *
         * Arguments:
         *   max_iter: Maximum number of iterations
         *   tol: Tolerance criteria - for minimising frontier
         *        images, this is the maximum component of the gradient
         *        vector, for minimising NEB path, this is the RMS of
         *        the total gradient
         */
        ensure(max_iter > 0, "Max iterations must be positive");
        ensure(tol > 0, "Gradient tolerance must be positive");
    }

    void BBMinimiser::update_trust_radius() {
        /* Update the trust radius */
        if (iter == 0) {
            last_low_rms_g = arrx::rms_v(grad);
            last_low_rmsg_iter = 0;
            return;
        }

        auto curr_rms_g = arrx::rms_v(grad);
        auto prev_rms_g = arrx::rms_v(last_grad);
        if (curr_rms_g < last_low_rms_g) {
            last_low_rms_g = curr_rms_g;
            last_low_rmsg_iter = iter;
        }

        if (curr_rms_g < prev_rms_g) {
            Nmin++;
        } else {
            Nmin = 0;
        }

        if (iter % 5 != 0) return;  // only update every 5th iteration

        // decrease trust radius if no improvement for 4 iterations
        if (iter - last_low_rmsg_iter > 4) {
            trust = std::max(trust * 0.7, min_trust);
            if (debug_pr)
                        std::cout << "Current trust radius = " << trust << "\n";
        } else if (Nmin >= 3) {
            // increase if gradient improved for 3 consecutive iterations
            // only increase if last step was at trust radius
            auto last_dx = arrx::rms_v(coords - last_coords);
            if (std::abs(last_dx - trust) / trust < 0.01) {
                trust = std::min(trust * 1.1, max_trust);
                if (debug_pr)
                        std::cout << "Current trust radius = " << trust << "\n";
            }
        }
    }

    void BBMinimiser::take_step() {
        /* Take a single optimiser step */

        update_trust_radius();

        if (iter == 0) {
            step = -grad;
        } else {
            arrx::noalias(s_k) = coords - last_coords;
            arrx::noalias(y_k) = grad - last_grad;
            auto s_dot_y = arrx::dot(s_k, y_k);
            double alpha;
            auto y_dot_y = arrx::dot(y_k, y_k);
            if (y_dot_y > 1e-8) {
                alpha = s_dot_y / y_dot_y;  // short BB step
                if (alpha < 0) alpha *= -1.0;
                arrx::noalias(step) = -grad * alpha;
            } else if (s_dot_y > 1e-8) {
                alpha = arrx::dot(s_k, s_k) / s_dot_y;  // long BB step
                arrx::noalias(step) = -grad * alpha;
            } else {
                // cannot do BB step, take steepest descent
                arrx::noalias(step) = -grad;
            }
        }

        auto dx = arrx::rms_v(step);
        if (dx > trust) {
            step *= (trust / dx);
        }
        auto max_dx = arrx::abs_max(step);
        if (max_dx > max_step) {
            step *= (max_step / max_dx);
        }

        arrx::noalias(last_coords) = coords;
        last_en = en;
        last_grad = grad;
        arrx::noalias(coords) = coords + step;
    }

    int BBMinimiser::min_frontier(NEB& neb,
                                  const NEB::frontier_pair idxs,
                                  const IDPPotential& pot) {
        /* Minimise the frontier images of a NEB using the Barzilai-Borwein
         * method
         *
         * Arguments:
         *
         *   neb: The NEB object holding the images
         *   idxs: The pair of indices for the frontier
         *
         * Returns:
         *   (int): The index of the image that converged. If did
         *          not converge, the index of image with the lower
         *          gradient is returned
         */
        ensure(idxs.left > 0 && idxs.right < neb.n_images - 1
               && idxs.left < idxs.right, "Frontier indices are wrong");
        if (debug_pr) std::cout << "=== Minimising frontier images: "
                                << idxs.left << ", " << idxs.right << " ===\n";

        size_t dim = 2 * neb.n_atoms * 3;
        coords.resize(dim);
        grad.resize(dim);

        while (iter < maxiter) {
            pot.calc_idpp_engrad(idxs.left, neb.images[idxs.left]);
            pot.calc_idpp_engrad(idxs.right, neb.images[idxs.right]);
            neb.images[idxs.left].update_neb_grad(
                neb.images[idxs.left - 1], neb.images[idxs.right]
            );
            neb.images[idxs.right].update_neb_grad(
                neb.images[idxs.left], neb.images[idxs.right + 1]
            );
            neb.get_frontier_coords(coords);
            neb.get_frontier_engrad(en, grad);
            if (debug_pr)
                std::cout << " Energies = (" << neb.images[idxs.left].en <<
                            ", " << neb.images[idxs.right].en << ") RMS(g) = "
                            << arrx::rms_v(grad) << "\n";
            if (neb.images[idxs.left].max_g() < gtol
                && neb.images[idxs.right].max_g() < gtol)
            {
                break;
            }
            take_step();
            iter++;
            neb.set_frontier_coords(coords);
        }
        if (iter >= maxiter && debug_pr) {
            std::cout << "Warning: reached max iterations\n";
        } else if (debug_pr) {
            std::cout << "Info: Frontier images converged\n";
        }

        if (neb.images[idxs.left].max_g() < neb.images[idxs.right].max_g()) {
            return idxs.left;
        } else {
            return idxs.right;
        }
    }

    void BBMinimiser::min_path(NEB& neb, const IDPPotential& pot) {
        /* Minimise a series of NEB images using the IDPP potential
         *
         * Arguments:
         *   neb: The NEB object
         */
        ensure(neb.images_prepared, "NEB images are not filled in");
        if (debug_pr)
            std::cout << "=== Minimising NEB path ===\n";

        size_t dim = (neb.n_images - 2) * neb.n_atoms * 3;
        coords.resize(dim);
        grad.resize(dim);

        while (iter < maxiter) {
            for (int k = 1; k < neb.n_images - 1; k++) {
                pot.calc_idpp_engrad(k, neb.images[k]);
            }
            for (int k = 1; k < neb.n_images - 1; k++) {
                neb.images[k].update_neb_grad(
                    neb.images[k-1], neb.images[k+1]
                );
            }
            neb.get_coords(coords);
            neb.get_engrad(en, grad);
            auto curr_rms_g = arrx::rms_v(grad);
            if (debug_pr) std::cout << " Path energy = " << en
                                        << " RMS grad = " << curr_rms_g << "\n";
            if (curr_rms_g < gtol) {
                break;
            }
            take_step();
            iter++;
            neb.set_coords(coords);
        }

        if (iter >= maxiter && debug_pr) {
            std::cout << "Warning: reached max iterations\n";
        } else if (debug_pr) {
            std::cout << "Info: NEB path converged\n";
        }
    }

    void IdppParams::check_validity() const {
        /* Check that the parameters for IDPP are valid */
        ensure(k_spr > 0, "Spring constant must be positive");
        ensure(rmsgtol > 0, "RMS gradient tolerance must be positive");
        ensure(maxiter > 0, "Max iterations must be positive");
        ensure(add_img_maxiter > 0, "Adding image maxiter must be positive");
        ensure(add_img_maxgtol > 0, "Adding image maxgtol must be positive");
    }

    NEB calculate_neb(arrx::array1d init_coords,
                      arrx::array1d final_coords,
                      const int num_images,
                      const IdppParams& params,
                      const arrx::array1d& interm_coords,
                      const bool load_from_interm) {
        /* Create a NEB object from the initial and final coordinates
         * with the parameters supplied, and then fill the path
         *
         * Arguments:
         *   init_coords: Initial coordinates (will be moved)
         *   final_coords: Final coordinates (will be moved)
         *   num_images: Number of images
         *   params: Parameters for running the calculation
         *   interm_coords: Intermediate coordinates (which may be read from)
         *   load_from_interm: Whether to load the intermediate coordinates
         */
        params.check_validity();

        std::cout.setf(std::ios::fixed);
        std::cout.setf(std::ios::showpoint);
        std::cout.precision(5);

        debug_pr = params.debug;

        auto pot = IDPPotential(init_coords, final_coords, num_images);

        auto neb = NEB(
            std::move(init_coords), std::move(final_coords),
            params.k_spr, num_images
        );

        // load from preset intermediate coordinates if requested
        if (load_from_interm) {
            neb.set_coords(interm_coords);
            neb.reset_k_spr();
            neb.images_prepared = true;
        } else {
            // S-IDPP with less than 5 images does not make sense
            if (params.sequential && num_images > 4) {
                neb.fill_sequentially(pot, params.add_img_maxiter,
                                      params.add_img_maxgtol);
            } else {
                neb.fill_linear_interp();
            }
        }

        // relax the path
        auto opt = BBMinimiser(params.maxiter, params.rmsgtol);
        opt.min_path(neb, pot);
        return neb;
    }

    void calculate_idpp_path(double* init_coords_ptr,
                            double* final_coords_ptr,
                            int coords_len,
                            int n_images,
                            double* all_coords_ptr,
                            const IdppParams& params) {
        /* Calculate the IDPP path from initial to final coordinates. This takes
         * in the coordinates as pointers to the beginning of numpy arrays, and
         * therefore must be provided contiguous arrays. Additionally, the
         * coords_len attribute must be set correctly.
         *
         * Arguments:
         *
         *   init_coords_ptr: Pointer to the initial coordinates numpy array
         *                    of shape (N,)
         *
         *   final_coords_ptr: Pointer to the final coordinates numpy array
         *                    of shape (N,)
         *
         *   coords_len: Length of the coordinates for ONE image (N)
         *
         *   n_images: Number of images (K)
         *
         *
         *   all_coords_ptr: Pointer to the array where coordinates of the
         *                   intermediate images will be stored, must be of
         *                   shape ([K-2] x N)
         *
         *   params: The parameters for the IDPP calculation
         */
        ensure(coords_len > 0, "Incorrect coordinates length");

        auto init_coords = arrx::array1d(init_coords_ptr, coords_len);
        auto final_coords = arrx::array1d(final_coords_ptr, coords_len);

        auto neb = calculate_neb(
            std::move(init_coords), std::move(final_coords),
            n_images, params, arrx::array1d(), false
        );

        // copy the final coordinates to the output array
        arrx::array1d all_coords;
        all_coords.resize((n_images - 2) * coords_len);
        neb.get_coords(all_coords);
        auto req_dim = (n_images - 2) * coords_len;
        for (int i = 0; i < req_dim; i++) {
            all_coords_ptr[i] = all_coords[i];
        }
    }

    double get_path_length(double *init_coords_ptr,
                           double *final_coords_ptr,
                           int coords_len,
                           int n_images,
                           const IdppParams& params) {
        /* Calculate the path length for the IDPP path from the initial to
         * final set of coordinates. Takes in pointer arrays (from numpy
         * buffers) and returns the cumulative sum length.
         *
         * Arguments:
         *   init_coords_ptr: Pointer to the initial coordinates numpy array
         *                    of shape (N,)
         *
         *   final_coords_ptr: Pointer to the final coordinates numpy array
         *                    of shape (N,)
         *
         *   coords_len: Length of the coordinates for ONE image (N)
         *
         *   n_images: Number of images (K)
         *
         *   params: The parameters for the IDPP calculation
         */
        ensure(coords_len > 0, "Incorrect coordinates length");

        auto init_coords = arrx::array1d(init_coords_ptr, coords_len);
        auto final_coords = arrx::array1d(final_coords_ptr, coords_len);

        auto neb = calculate_neb(
            std::move(init_coords), std::move(final_coords),
            n_images, params, arrx::array1d(), false
        );

        double dist = 0.0;
        for (int k = 0; k < n_images - 1; k++) {
            dist +=
                   arrx::norm_l2(neb.images[k].coords - neb.images[k+1].coords);
        }
        return dist;
    }

    void relax_path(double* all_coords_ptr,
                    int coords_len,
                    int n_images,
                    const IdppParams& params) {
        /* Given a set of already generated coordinates, relax the path
         * using the IDPP potential
         *
         * Arguments:
         *  all_coords_ptr: Pointer to the array of coordinates of shape
         *                 (K x N). Including the end point coordinates.
         *
         *  coords_len: Length of the coordinates for ONE image (N)
         *
         *  n_images: Number of images (K)
         *
         *  params: The IDPP parameters
         */
        ensure(coords_len > 0, "Incorrect coordinates length");

        auto init_coords = arrx::array1d(all_coords_ptr, coords_len);
        auto final_coords = arrx::array1d(
            all_coords_ptr + (n_images - 1) * coords_len, coords_len
        );
        auto intermediate_coords = arrx::array1d(
            all_coords_ptr + coords_len, coords_len * (n_images - 2)
        );
        auto neb = calculate_neb(
            std::move(init_coords), std::move(final_coords), n_images,
            params, intermediate_coords, true
        );

        // load back the coordinates
        neb.get_coords(intermediate_coords);
        auto req_dim = (n_images - 2) * coords_len;
        for (int i = 0; i < req_dim; i++) {
            all_coords_ptr[i + coords_len] = intermediate_coords[i];
        }

    }

}
