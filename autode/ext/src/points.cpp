#include <random>
#include <cmath>
#include "points.h"

namespace autode {

    PointGenerator::PointGenerator(int n_points,
                                   int dimension,
                                   double min_val = -3.145,
                                   double max_val = 3.145) {
        /* Generate a set of n points evenly spaced in a cube with dimension d,
         * with dimensions l = max_val - min_val
         *
         * Arguments:
         *     n: Number of points to generate
         *
         *     dim: Dimension of the space to generate the points in
         *
         *     min_val: Minimum value of the
         */
        this->n = n_points;
        this->dim = dimension;

        this->min_val = min_val;
        this->max_val = max_val;

        set_init_random_points();
    }


    void PointGenerator::set_init_random_points(){
        /* Set the set of points using random uniform distribution
         */

        std::random_device rand_device;

        std::uniform_real_distribution<double> unif_distro(min_val, max_val);
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
    }


    double PointGenerator::norm_grad() {
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


    void PointGenerator::set_grad(){
        /* Calculate the gradient with respect to the points
         *
         *  E = Σ'_ij 1 / |x_i - x_j|
         *
         *  where x_i is a vector in dim-dimensional space with a distance
         *
         *  |x_i - x_j| = √[(x_i0 x_j0)^2 +  (x_i1 x_j1)^2  + ... ]
         *
         *  TODO: Periodic boundaries
         *
         */

        for (int i=0; i < n; i++) {
            // Zero the gradient of all (x, y, z, ..) components
            std::fill(s_grad[i].begin(), s_grad[i].end(), 0.0);

            for (int j = 0; j < n; j++) {

                // Only loop over non identical pairs
                if (i == j) {
                    continue;
                }


                double r_sq = 0.0;

                for (int k = 0; k < dim; k++){
                    double d_k = points[i][k] - points[j][k];
                    r_sq += d_k * d_k;
                }

                // Repulsion factor
                auto rep_ftr = -1.0 / r_sq;

                for (int k = 0; k < dim; k++){
                    double d_k = points[i][k] - points[j][k];
                    s_grad[i][k] += rep_ftr * d_k;
                }

            } // j
        } // i
    };


    void PointGenerator::run(double grad_tol = 1E-4,
                             double step_size = 0.1,
                             int max_iterations = 100) {
        /* Generate a set of n points evenly spaced in a dimension d. Sets
         * PointGenerator.points. Will minimise the Coulomb energy between the
         * points (as J. J. Thomson in 1904 in 3D) from a random starting point
         *
         *  Arguments:
         *      grad_tol:
         *
         */
        set_grad();
        int iteration = 0;

        while (norm_grad() > grad_tol && iteration < max_iterations){
            for (int point_idx = 0; point_idx < n; point_idx++){
                // Do a steepest decent step

                for (int k = 0; k < dim; k ++) {
                    points[point_idx][k] -= step_size * s_grad[point_idx][k];
                } // k
            }

            set_grad();
            iteration++;
        }
    }


}
