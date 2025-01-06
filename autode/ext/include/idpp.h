#ifndef ADE_EXT_IDPP_H
#define ADE_EXT_IDPP_H

#include <vector>
#include "arrayhelper.hpp"
#include "utils.h"

namespace autode
{

    class Image {
        /* A NEB image, holding coordinates, energy and gradients */

    private:
        int n_atoms; // number of atoms

    public:
        arrx::array1d coords;  // coordinates
        arrx::array1d grad;  // gradients
        double en = 0.0;  // energy
        double k_spr;  // force constant

        Image() = default;

        explicit Image(int num_atoms, double k);

        double get_tau_k_fac(arrx::array1d& tau,
                             const Image& img_m1,
                             const Image& img_p1,
                             const bool force_lc) const;

        void update_neb_grad(const Image& img_m1,
                             const Image& img_p1,
                             bool force_lc);

        double rms_g() const;
    };

    class IDPPPotential {
        /* The IDPP potential, using weighted interatomic distances */

    private:
        int n_atoms;  // Total number of atoms
        int n_images;  // Total number of images
        std::vector<arrx::array1d> all_target_ds; // interpolated bond distances

    public:
        IDPPPotential() = default;

        explicit IDPPPotential(const arrx::array1d& init_coords,
                               const arrx::array1d& final_coords,
                               const int num_images);

        void calc_idpp_engrad(const int idx, Image& img) const;
    };

    class NEB {
        /* A NEB calculation for interpolation, holding images and potential */

    private:
        int n_images;  // total number of images
        int n_atoms;  // total number of atoms

    public:
        struct frontier_pair {
            int left, right;
        } frontier;  // frontier image indices
        bool images_prepared = false; // are images filled in?

        std::vector<Image> images;  // Set of images
        double k_spr;  // base spring constant
        IDPPPotential idpp_pot;  // the idpp potential

        NEB(arrx::array1d&& init_coords,
            arrx::array1d&& final_coords,
            double k_spr,
            int num_images);

        void fill_linear_interp();

        void fill_sequentially();

        double get_d_id() const;

        double get_k_mid() const;

        void reset_k_spr();

        void get_engrad(double& en, arrx::array1d& grad) const;

        void get_frontier_engrad(double& en, arrx::array1d& grad) const;

        void get_coords(arrx::array1d& coords) const;

        void get_frontier_coords(arrx::array1d& coords) const;

        void set_coords(const arrx::array1d& coords);

        void set_frontier_coords(const arrx::array1d& coords);

        void update_engrad();

        void update_frontier_engrad();

        void add_first_two_images();

        void add_image_next_to(const int idx);
    };

    /* Barzilai-Borwein algorithm for minimisation */
    class BBMinimiser {

    private:
        int iter = 0;  // current number of iterations
        int maxiter;  // maximum number of iterations
        int n_backtrack = 0;  // number of backtracks
        double rmsgtol;  // gradient tolerance

        const double bb_maxstep = 0.02; // max step size for BB
        const double sd_maxstep = 0.01; // max step size for SD

    public:
        arrx::array1d coords, last_coords;  // current & last coordinates
        arrx::array1d grad, last_grad;  // current & last gradients
        double en, last_en;  // current & last energies
        arrx::array1d step;

        explicit BBMinimiser(int max_iter, double rms_gtol);

        void calc_bb_step();

        void calc_sd_step();

        void backtrack();

        void take_step();

        int min_frontier(NEB& neb, const NEB::frontier_pair idxs);

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