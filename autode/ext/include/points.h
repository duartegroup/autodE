#ifndef ADE_EXT_POINTS_H
#define ADE_EXT_POINTS_H
#include "vector"

namespace autode {

    class CubePointGenerator {

        private:
            // Attributes
            std::vector<std::vector<double>> s_grad;
            std::vector<double> delta_point;

            // Constructor helpers
            void set_init_random_points();
            void shift_box_centre();

            // Distance functions
            void set_delta_point_pbc(int i, int j);
            double norm_squared_delta_point();

        public:
            int dim = 0;
            int n = 0;

            double min_val;
            double max_val;
            double box_length;
            double half_box_length;
         
            // Gradient functions
            double norm_grad();

            std::vector<std::vector<double>> points;

            explicit CubePointGenerator();
            explicit CubePointGenerator(int n_points,
                                        int dimension,
                                        double min_val = -3.145,
                                        double max_val = 3.145);

            void set_grad();
            void run(double grad_tol, double step_size, int max_iterations);
    };
}

#endif //ADE_EXT_POINTS_H
