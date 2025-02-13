#ifndef ADE_EXT_IDPP_H
#define ADE_EXT_IDPP_H

#include <vector>
#include "arrayhelper.hpp"

namespace autode
{

    /* A struct containing parameters used for IDPP */
    struct IdppParams {
        double k_spr;  // spring constant
        bool sequential;  // whether sequential IDPP or not
        bool debug;  // whether to print debug messages
        double rmsgtol; // RMS gradient tolerance for path
        int maxiter; // maxiter for path
        double add_img_maxgtol; // Max gradient tol. for adding img
        double add_img_maxiter; // maxiter for each image addition

        void check_validity() const;
    };


    class FIREMinimiser {
        /* A Fast Inertial Relaxation Engine (FIRE) minimiser */

    private:
        // tunable parameters: set to sensible defaults
        static constexpr double dt_init = 0.25;  // initial time step
        static constexpr double dt_max = 1.0;  // maximum time step
        static constexpr double dt_min = 0.001;  // minimum time step
        static constexpr double alpha_start = 0.1;  // initial alpha
        static constexpr double atom_mass = 4.0;  // assumed mass of each atom
        static constexpr double f_alpha = 0.99;  // alpha damping factor
        static constexpr double f_inc = 1.1;  // factor to increase dt
        static constexpr double f_dec = 0.5;  // factor to decrease dt
        static constexpr int n_delay = 5;  // initial delay before increasing dt

        // variables
        int iter = 0;  // current number of iterations
        double dt = dt_init;  // current time step
        double alpha = alpha_start;  // current alpha

        int N_plus = 0;  // number of steps with positive dot product
        int N_minus = 0;  // number of steps with negative dot product

        arrx::array1d vels;  // velocities

    public:

        FIREMinimiser() = default;

        void take_step(arrx::array1d* coords, const arrx::array1d& grad);

    };

    class Image {
        /* A NEB image, holding coordinates, energy and gradients */

    private:
        int n_atoms; // number of atoms
        FIREMinimiser opt;  // minimiser for this image

    public:
        arrx::array1d coords;  // coordinates
        arrx::array1d grad;  // gradients
        double en = 0.0;  // energy
        double k_m1, k_p1;  // force constants to left and right images
        bool active = false;  // whether this image is active

        Image() = default;

        explicit Image(int num_atoms, double k);

        double get_tau_k_fac(arrx::array1d* tau,
                             const Image& img_m1,
                             const Image& img_p1,
                             const bool force_lc) const;

        void update_neb_grad(const Image& img_m1,
                             const Image& img_p1,
                             bool force_lc);

        double max_g() const;

        void min_step();
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
        /* A NEB calculation for interpolation, holding images */

    public:
        int n_images;  // total number of images
        int n_atoms;  // total number of atoms

        struct frontier_pair {
            int left, right;
        } frontier;  // frontier image indices
        bool images_prepared = false; // are images filled in?

        std::vector<Image> images;  // Set of images
        double k_spr;  // base spring constant

        NEB(arrx::array1d init_coords,
            arrx::array1d final_coords,
            double k_spr,
            int num_images);

        void fill_linear_interp();

        void fill_sequentially(const IDPPPotential& pot,
                               const int add_maxiter,
                               const double add_maxgtol);

        double get_d_id() const;

        double get_k_mid() const;

        void reset_k_spr();

        void get_engrad(double& en, arrx::array1d& grad) const;

        void get_frontier_engrad(double& en, arrx::array1d& grad) const;

        void get_coords(arrx::array1d& coords) const;

        void get_frontier_coords(arrx::array1d& coords) const;

        void set_coords(const arrx::array1d& coords);

        void set_frontier_coords(const arrx::array1d& coords);

        void add_first_two_images();

        void add_image_next_to(const int idx);

    };

    /* Barzilai-Borwein algorithm for minimisation */
    class BBMinimiser {

    private:
        int iter = 0;  // current number of iterations
        int maxiter;  // maximum number of iterations
        int n_backtrack = 0;  // number of backtracks
        double gtol; // gradient tolerance criteria

        const double bb_maxstep = 0.02; // max step size for BB
        const double sd_maxstep = 0.01; // max step size for SD

    public:
        arrx::array1d coords, last_coords;  // current & last coordinates
        arrx::array1d grad, last_grad;  // current & last gradients
        double en, last_en;  // current & last energies
        arrx::array1d step;

        explicit BBMinimiser(int max_iter, double tol);

        void calc_bb_step();

        void calc_sd_step();

        void backtrack();

        void take_step();

        int min_frontier(NEB& neb,
                         const NEB::frontier_pair idxs,
                         const IDPPPotential& pot);

        void minimise_neb(NEB& neb, const IDPPPotential& pot);
    };

    NEB calculate_neb(arrx::array1d init_coords,
                      arrx::array1d final_coords,
                      const int num_images,
                      const IdppParams& params,
                      const arrx::array1d& interm_coords,
                      const bool load_from_interm);

    void calculate_idpp_path(double* init_coords_ptr,
                             double* final_coords_ptr,
                             int coords_len,
                             int n_images,
                             double* all_coords_ptr,
                             const IdppParams& params);

    double get_path_length(double* init_coords_ptr,
                         double* final_coords_ptr,
                         int coords_len,
                         int n_images,
                         const IdppParams& params);

    void relax_path(double* all_coords_ptr,
                    int coords_len,
                    int n_images,
                    const IdppParams& params);
}

#endif // ADE_EXT_IDPP_H