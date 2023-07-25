#include <random>
#include <cmath>
#include <stdexcept>
#include "points.h"


namespace autode {

    CubePointGenerator::CubePointGenerator() = default;

    CubePointGenerator::CubePointGenerator(int n_points,
                                           int dimension,
                                           double min_val,
                                           double max_val) {
        /* Generate a set of n points evenly spaced in a (hyper)cube with
         * dimension d, with side length
         *
         *      l = max_val - min_val
         *
         * Arguments:
         *     n: Number of points to generate
         *
         *     dim: Dimension of the space to generate the points in
         *
         *     min_val: Minimum value in the box (single dimension)
         *
         *     max_val:
         */
        if (n_points < 2){
            throw std::runtime_error("Must have at least 2 points to generate "
                                     "a point set");
        }

        this->n = n_points;
        this->dim = dimension;

        this->min_val = min_val;
        this->max_val = max_val;

        this->box_length = (max_val - min_val);
        this->half_box_length = (max_val - min_val) / 2.0;

        if (box_length < 0){
            throw std::runtime_error("Must have a positive side length. i.e. "
                                     "min_val < max_val");
        }

	// Gradient with respect to point displacement
	this->s_grad = std::vector<std::vector<double>>(n_points,
	                                             std::vector<double>(dimension, 0.0));

    // ∆X_ij = {(x_i - x_j), (y_i - y_j), ...}
    this->delta_point = std::vector<double>(dimension, 0.0);

    set_init_random_points();
    }


    void CubePointGenerator::set_init_random_points(){
        /* Set the set of points using random uniform distribution within a
         * box, centred at the origin (for more simple periodic
         * boundary conditions)
         */
        points.clear();
        s_grad.clear();

        std::random_device rand_device;

        std::uniform_real_distribution<double> unif_distro(-half_box_length,
                                                           half_box_length);
        std::default_random_engine rand_generator(rand_device());

        // Initialise all the points randomly in the space
        for (int i=0; i < n; i++){
            std::vector<double> point;
            point.reserve(dim);

            // with dim members per point
            for (int j=0; j < dim; j++){
                point.push_back(unif_distro(rand_generator));
            }

            points.push_back(point);

            // Initialise a zero gradient initially
            s_grad.emplace_back(point.size(), 0.0);
        }
        set_grad();

        // Prevent very close initial geometries -> large gradients
        if (norm_grad() > 100){
            set_init_random_points();
        }
    }


    double CubePointGenerator::norm_grad() {
        /* Calculate the norm of the gradient vector
         */
        double norm = 0.0;

        for (auto &grad : s_grad){
            for (auto &component : grad){
                norm += component * component;
            }
        }

        return sqrt(norm);
    }


    void CubePointGenerator::set_delta_point_pbc(int i, int j){
        /* Calculate the components of the ∆X_ij vector, in 2D
         *
         *   ∆X_ij = {(x_i - x_j), (y_i - y_j)}
         *
         *  with periodic boundary conditions, such that in each direction
         *  the nearest atom is, at most, half a box length away.
         *
         *  Arguments:
         *      i: Index of one point
         *
         *      j: Index of another point
         */

        for (int k = 0; k < dim; k++){
            delta_point[k] = points[i][k] - points[j][k];

            // Apply the nearest image convention in all directions
           if (delta_point[k] > half_box_length){
               delta_point[k] -= box_length;
           }
           else if (delta_point[k] < -half_box_length){
               delta_point[k] += box_length;
           }
        } // k
    }


    void CubePointGenerator::shift_box_centre(){
        /* Shift the box back such that the center is between min_val, max_val
         * in all dimensions
         */
        for (auto &point : points) {
            for (auto &component: point) {
                component += (max_val - min_val) / 2.0;
            } // k
        }
    }


    double CubePointGenerator::norm_squared_delta_point(){
        /*
         *
         */
        double norm_squared = 0.0;

        for (auto &component : delta_point){
            norm_squared += component * component;
        }

        return norm_squared;
    }


    void CubePointGenerator::set_grad(){
        /* Calculate the gradient with respect to the points
         *
         *  E = Σ'_ij 1 / |x_i - x_j|
         *
         *  where x_i is a vector in dim-dimensional space with a distance
         *
         *  |x_i - x_j| = √[(x_i0 x_j0)^2 +  (x_i1 x_j1)^2  + ... ]
         *
         */

        for (int i=0; i < n; i++) {
            // Zero the gradient of all (x, y, z, ..) components
            std::fill(s_grad[i].begin(), s_grad[i].end(), 0.0);

            // Should loop for all i, j and j, i but not i = j
            for (int j = 0; j < n; j++) {

                if (i == j) continue;

                set_delta_point_pbc(i, j);
                auto rep_ftr = -1.0 / norm_squared_delta_point();

                for (int k = 0; k < dim; k++){
                    s_grad[i][k] += rep_ftr * delta_point[k];
                }

            } // j
        } // i
    }


    void CubePointGenerator::run(double grad_diff_tol = 1E-4,
                                 double step_size = 0.1,
                                 int max_iterations = 100) {
        /* Generate a set of n points evenly spaced in a dimension d. Sets
         * CubePointGenerator.points. Will minimise the Coulomb energy between
         * the points (as J. J. Thomson in 1904 in 3D) from a random starting
         * point
         *
         *  Arguments:
         *      grad_diff_tol: Tolerance on the absolute difference between
         *                     sequential gradient evaluations |∇V_i - ∇V_i+1|
         *
         *      step size: Fixed step size to take in the steepest decent
         *
	     *      max_iterations:
         */
        set_grad();
        int iteration = 0;

        double _norm_grad = 0.0;
        double _norm_grad_prev = 2*grad_diff_tol;

        while (fabs(_norm_grad - _norm_grad_prev) > grad_diff_tol && iteration < max_iterations) {

            _norm_grad_prev = _norm_grad;

            for (int point_idx = 0; point_idx < n; point_idx++) {
                // Do a steepest decent step

                for (int k = 0; k < dim; k++) {

                    // Ensure the translation is not more than the whole
                    // box length
                    points[point_idx][k] -= step_size * s_grad[point_idx][k];

                    if (points[point_idx][k] > half_box_length) {
                        points[point_idx][k] -= box_length;
                    } else if (points[point_idx][k] < -half_box_length) {
                        points[point_idx][k] += box_length;
                    }


                } // k
            }// point_idx

            set_grad();
            _norm_grad = norm_grad();

            iteration++;
        }

    shift_box_centre();
    }


}
