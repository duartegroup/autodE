#include "utils.h"
#include <cmath>
#include <stdexcept>
#include "xtensor/xtensor.hpp"
#include "xtensor/xmath.hpp"


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

        void assert_exc(const bool condition, const std::string& message) {
            /* Check an assertion and raise an exception if it is not true
             *
             *  Arguments:
             *
             *    condition: The condition to check
             *
             *    message: The message in the exception if condition is false
             */
            if (!condition) {
                throw std::runtime_error(message);
            }
        }

        double dot(const xt::xtensor<double, 1>& arr1,
                   const xt::xtensor<double, 1>& arr2) {
            /* Dot product of two 1D vectors
            *
            *  Arguments:
            *
            *    arr1: The first vector
            *
            *    arr2: The second vector
            *
            * Returns:
            *    arr1 . arr2
            */
            return xt::sum(arr1 * arr2)();
        }

        double rms(const xt::xtensor<double, 1>& arr) {
            /* Root mean square of a 1D vector
            *
            *  Arguments:
            *    arr: The vector
            *
            * Returns:
            *   sqrt(mean(arr^2))
            */
            return std::sqrt(xt::mean(arr * arr)());
        }

        template<class T>
        Deque_Py<T>::Deque_Py(int maxlen) : maxlen(maxlen) {
            /* Create a new deque with size maxlen */
            this->data.clear();
            this->ptrs.clear();
            this->data.reserve(maxlen);
            this->ptrs.reserve(maxlen);
        }

        template<class T>
        int Deque_Py<T>::size() {
            /* The current size of the deque */
            return data.size();
        }

        template<class T>
        void Deque_Py<T>::reset_ptrs() {
            /* Regenerate pointers after new item is added to data */
            ptrs.clear();
            for (size_t i = 0; i < this->size(); i++) {
                ptrs.push_back(&data[i]);
            }
        }

        template<class T>
        void Deque_Py<T>::append(const T& item) {
            /* Append a new item to this deque */
            if (this->size() < maxlen) {
                data.push_back(item);
                this->reset_ptrs();
            } else {
                // if deque is full, rotate pointers and assign to last
                std::rotate(ptrs.begin(), ptrs.begin() + 1, ptrs.end());
                (*this)[this->size() - 1] = item;
            }
        }

        template<class T>
        T& Deque_Py<T>::operator[](int idx) {
            /* Return the item at the given index as a reference */
            assert_exc(idx < this->size(), "Array index out of bounds");
            return *(ptrs[idx]);
        }

        template class Deque_Py<xt::xtensor<double, 1>>;

    }
}

