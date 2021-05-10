#ifndef ADE_EXT_POINTS_H
#define ADE_EXT_POINTS_H
#include "vector"

namespace autode {

    class PointGenerator {

        private:
            double s_energy = 0.0;
            std::vector<std::vector<double>> s_grad;

            void set_init_random_points();
            double norm_grad();

        public:
            int dim = 0;
            int n = 0;
            double min_val;
            double max_val;

            std::vector<std::vector<double>> points;

            explicit PointGenerator(int n_points,
                                    int dimension,
                                    double min_val,
                                    double max_val);

            void set_grad();
            void run(double grad_tol, double step_size, int max_iterations);
    };
}

#endif //ADE_EXT_POINTS_H
