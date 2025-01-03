#ifndef ADE_EXT_UTILS_H
#define ADE_EXT_UTILS_H

#include <string>


namespace autode{

    namespace utils{

        int fpowi(int value, int root);

        int powi(int value, int exponent);

        void assert_exc(const bool condition, const std::string& message);

    }

}


#endif //ADE_EXT_UTILS_H
