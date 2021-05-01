#include "utils.h"
#include <cmath>


namespace autode{

    namespace utils{

        int fpowi(int value, int root){
            /* Floored n-th root of an integer. e.g.
             *
             *  fpowi(10, 2) ---> 3
             *  fpowi(3, 1) ---> 3
             *
             *  Arguments:
             *
             *     value:
             *
             *     root:
             *
             *  Returns:
             *     |_value^(1/root)_|
             */

            auto root_value = std::pow(static_cast<double>(value),
                                       1.0 / static_cast<double>(root));

            return static_cast<int>(std::floor(root_value));
        }

        int powi(int value, int exponent){
            /* Integer n-th power of an integer. e.g.
             *
             *  powi(10, 2) ---> 100
             *  powi(3, 1) ---> 3
             *
             *  Arguments:
             *
             *     value:
             *
             *     root:
             *
             *  Returns:
             *     value^exponent
             */

            auto result = std::pow(static_cast<double>(value),
                                   static_cast<double>(exponent));

            return static_cast<int>(result);
        }

    }
}

