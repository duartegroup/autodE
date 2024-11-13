#ifndef ADE_EXT_UTILS_H
#define ADE_EXT_UTILS_H
#include <string>
#include "xtensor/xtensor.hpp"

namespace autode{

    namespace utils{

        int fpowi(int value, int root);

        int powi(int value, int exponent);

        void assert_exc(const bool condition, const std::string& message);

        double dot(const xt::xtensor<double, 1>& arr1, const xt::xtensor<double, 1>& arr2);

        double rms(const xt::xtensor<double, 1>& arr);

        template<class T>
        class Deque_Py {
            /* A memory efficient deque class similar to Python - old items are
               overwritten when more than maxlen items are put into the class */
        private:
            int maxlen;  // max number of items
            std::vector<T> data;  // holds the actual data
            std::vector<T*> ptrs;  // pointers to data items

        public:
            Deque_Py(int maxlen);

            int size();

            void reset_ptrs();

            void append(const T& item);

            T& operator[](int idx);
        };
    }

}


#endif //ADE_EXT_UTILS_H
