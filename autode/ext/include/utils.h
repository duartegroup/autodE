#ifndef ADE_EXT_UTILS_H
#define ADE_EXT_UTILS_H

#include <string>
#include <stdexcept>


namespace autode{

    namespace utils{

        int fpowi(int value, int root);

        int powi(int value, int exponent);

        inline void ensure(const bool condition, const char* message) {
            /* Check an assertion and raise an exception if it is not true
             *
             *  Arguments:
             *
             *    condition: The condition to check
             *
             *    message: The message in the exception if condition is false
             */
            if (! condition) throw std::runtime_error(message);
        }

    }

}


#endif //ADE_EXT_UTILS_H
