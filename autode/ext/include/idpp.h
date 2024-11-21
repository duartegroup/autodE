#ifndef ADE_EXT_IDPP_H
#define ADE_EXT_IDPP_H

#include "xtensor/xtensor.hpp"
#include "utils.h"

namespace autode
{

    class Image {
        /* A NEB image, holding coordinates, energy and gradients */

    private:
        int n_atoms; // number of atoms

    public:
        xt::xtensor<double, 1> coords;  // coordinates
        xt::xtensor<double, 1> grad;  // gradients
        double en = 0.0;  // energy

        Image() = default;

        Image(int num_atoms);

        void update_neb_force(const Image& img_m1, const Image& img_p1, double k_spr);

        double rms_g() const;
    };

    class IDPPPotential {
        /* The IDPP potential, using weighted interatomic distances */

    private:
        int n_atoms;  // Total number of atoms
        int n_images;  // Total number of images
        std::vector<xt::xtensor<double, 1>> all_target_ds; // interpolated bond distances

    public:
        IDPPPotential() = default;

        IDPPPotential(const xt::xtensor<double, 1>& init_coords,
                    const xt::xtensor<double, 1>& final_coords,
                    const int num_images);

        void calc_idpp_engrad(const int idx, Image& img) const;
    };

    class NEB {
        /* A NEB calculation for interpolation, holding images and a potential */

    private:
        int n_images;  // total number of images
        int n_atoms;  // total number of atoms

    public:
        std::vector<Image> images;  // Set of images
        double k_spr;  // base spring constant
        IDPPPotential idpp_pot;  // the idpp potential

        NEB(const xt::xtensor<double, 1>& init_coords,
            const xt::xtensor<double, 1>& final_coords,
            double k_spr,
            int num_images,
            bool sequential);

        void fill_linear_interp();

        void fill_sequentially();

        void add_img_at(const int idx, const int neighbour_idx);

        void add_right(const int left_idx, const int right_idx);

        void add_left(const int left_idx, const int right_idx);

        void get_en_grad(double& en, xt::xtensor<double, 1>& flat_grad);

        void get_coords(xt::xtensor<double, 1>& flat_coords);

        void set_coords(const xt::xtensor<double, 1>& flat_coords);

        void update_en_grad();

        void minimise();
    };

    class LBFGSMinimiser {
        /* A class that minimiser energy using low memory BFGS */

    private:
        int iter = 0;
        int maxiter;  // maximum number of iterations
        int n_backtrack = 0;  // number of backtracks
        double rms_gtol;  // gradient tolerance

        // memory of coords and grad updates
        autode::utils::Deque_Py<xt::xtensor<double, 1>> s_ks, y_ks;
        xt::xtensor<double, 1> step;  // to store the step

        const double lbfgs_maxstep = 0.05;  // maximum step size for lbfgs
        const double sd_maxstep = 0.01;  // maximum step size for initial step

    public:
        xt::xtensor<double, 1> coords, last_coords;  // coordinates of the current and last step
        xt::xtensor<double, 1> grad, last_grad;  // gradients of the current and last step
        double en, last_en;  // energies of the current and last step

        LBFGSMinimiser(int maxiter, double rms_gtol, int max_vecs);

        void calc_lbfgs_step();

        void calc_sd_step();

        void backtrack();

        void take_step();

        void minimise_img(Image& img, const int idx, const IDPPPotential& idpp_pot);

        void minimise_neb(NEB& neb);
    };

    void calculate_idpp_path(double* init_coords_ptr,
                            double* final_coords_ptr,
                            int coords_len,
                            int n_images,
                            double k_spr,
                            bool sequential,
                            double* all_coords_ptr,
                            bool debug,
                            double gtol,
                            int maxiter);
}

#endif // ADE_EXT_IDPP_H