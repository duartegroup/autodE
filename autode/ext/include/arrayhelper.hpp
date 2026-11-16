#ifndef ADE_ARRX_HELPER_HPP
#define ADE_ARRX_HELPER_HPP

#include <vector>
#include <utility>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace arrx {

    inline void match_size(const size_t& s_1, const size_t& s_2) {
        if (s_1 != s_2) {
            throw std::runtime_error("Sizes of operands do not match!");
        }
    }

    enum class operation {
        add,
        subtract,
        multiply,
        divide,
        negate,
    };

    template <operation op>
    struct array_func {};

    template <>
    struct array_func<operation::add> {
        inline static double apply(double a, double b) {return a + b;}
    };

    template <>
    struct array_func<operation::subtract> {
        inline static double apply(double a, double b) {return a - b;}
    };

    template <>
    struct array_func<operation::multiply> {
        inline static double apply(double a, double b) {return a * b;}
    };

    template <>
    struct array_func<operation::divide> {
        inline static double apply(double a, double b) {return a / b;}
    };

    template <>
    struct array_func<operation::negate> {
        inline static double apply(double a) {return -a;}
    };

    template <typename T1, typename T2, operation op>
    class array_expr {
        /* A general template for an array-like from operations with two expressions */
    public:
        T1 var1;  // first expression
        T2 var2;  // second expression

        array_expr(T1&& v1, T2&& v2)
        : var1{std::forward<T1>(v1)}, var2{std::forward<T2>(v2)} {
            /* General constructor */
            match_size(var1.size(), var2.size());
        }

        size_t size() const {
            /* Return the size of the array-like expression */
            return var1.size();
        }

        inline double operator[] (size_t idx) const {
            /* Access elements by index */
            return array_func<op>::apply(var1[idx], var2[idx]);
        }
    };


    template <typename T1, operation op>
    class array_expr<T1, double, op> {
        /* Specialised template for array-double operations */
    public:
        T1 var;  // expression
        const double scalar;  // scalar value

        array_expr(T1&& v, const double& s)
        : var{std::forward<T1>(v)}, scalar{s} {}

        size_t size() const {
            /* Return the size of the array-like expression */
            return var.size();
        }

        inline double operator[] (size_t idx) const {
            /* Access elements by index */
            return array_func<op>::apply(var[idx], scalar);
        }
    };


    template <typename T2, operation op>
    class array_expr<double, T2, op> {
        /* Specialised template for double-array operations */
    public:
        const double scalar;  // scalar value
        T2 var;  // expression

        array_expr(const double& s, T2&& v)
        : scalar{s}, var{std::forward<T2>(v)} {}

        size_t size() const {
            /* Return the size of the array-like expression */
            return var.size();
        }

        inline double operator[] (size_t idx) const {
            /* Access elements by index */
            return array_func<op>::apply(scalar, var[idx]);
        }
    };


    template <typename T1, operation op>
    class array_expr<T1, void, op> {
        /* Specialised template for unary array operations */
    public:
        T1 var;  // expression

        array_expr(T1&& v) : var{std::forward<T1>(v)} {}

        size_t size() const {
            /* Return the size of the array-like expression */
            return var.size();
        }

        inline double operator[] (size_t idx) const {
            /* Access elements by index */
            return array_func<op>::apply(var[idx]);
        }
    };


    class array1d {
        /* An array class for floating point (double) data points */

        using array_iter = std::vector<double>::iterator;
        using const_array_iter = std::vector<double>::const_iterator;

    private:
        std::vector<double> data;  // data array

    public:

        array1d() = default;

        explicit array1d(const double* data, size_t size) {
            /* Construct from a C-style array */
            this->data = std::vector<double>(data, data + size);
        }

        explicit array1d(std::vector<double> data) {
            /* Construct from a C++ vector */
            this->data = data;
        }

        template <typename X, typename Y, operation op>
        array1d(const array_expr<X, Y, op>& arr_expr) {
            /* Construct from an array-expression */
            const size_t dim = arr_expr.size();
            data.reserve(dim);
            for (size_t i = 0; i < dim; i++) {
                data.push_back(arr_expr[i]);
            }
        }

        size_t size() const {
            /* Return the size of the array */
            return data.size();
        }

        void resize(size_t dim) {
            /* Resize the array, filling new members to 0 */
            data.resize(dim, 0.0);
        }

        void fill(const double& var) {
            /* Fill the array with a specific value */
            for (auto& x : data) x = var;
        }

        double& operator[](size_t idx) {
            /* Overloaded operator to access elements by index */
            return data[idx];
        }

        const double& operator[](size_t idx) const {
            /* Overloaded operator to access elements by index (const) */
            return data[idx];
        }

        array_iter begin() {
            /* Return an iterator to the beginning of the array */
            return data.begin();
        }

        array_iter end() {
            /* Return an iterator to the end of the array */
            return data.end();
        }

        const_array_iter begin() const {
            /* Return a const iterator to the beginning of the array */
            return data.begin();
        }

        const_array_iter end() const {
            /* Return a const iterator to the end of the array */
            return data.end();
        }

        /* In-place mathematical operations with scalars */
        array1d& operator+=(const double& val) {
            for (auto& x: data) x += val;
            return *this;
        }

        array1d& operator-=(const double& val) {
            for (auto& x: data) x -= val;
            return *this;
        }

        array1d& operator*=(const double& val) {
            for (auto& x: data) x *= val;
            return *this;
        }

        array1d& operator/=(const double& val) {
            for (auto& x: data) x /= val;
            return *this;
        }
    };

    inline array1d zeros(size_t len) {
        /* Create a zero array of len size */
        array1d arr;
        arr.resize(len);
        return arr;
    }


    /* Templates defined to help with creating correct classes and overloads */

    // return the type after removing const, volatile and reference
    template <typename T>
    using remove_cv_ref_t = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

    // is type array1d, ignoring const and reference?
    template <typename T>
    struct is_1d_array : std::integral_constant<
                            bool,
                            std::is_same<remove_cv_ref_t<T>, array1d>::value
                        > {};

    template <typename>
    struct derived_array_expr : std::false_type {};

    template <typename T1, typename T2, operation op>
    struct derived_array_expr<array_expr<T1, T2, op>> : std::true_type {};


    /* Implement array slicing */

    template <typename T>
    class slice_wrapper {

    private:
        T arr;
        size_t begin;
        size_t len;

        static_assert(is_1d_array<T>::value, "Only arrays can be sliced!");
        static_assert(!std::is_same<T, array1d>::value, "Temporary arrays can not be sliced");

    public:
        slice_wrapper() = delete;

        explicit slice_wrapper(T&& arr, size_t start, size_t end)
        : arr(std::forward<T>(arr)), begin(start) {
            if (start < 0) throw std::invalid_argument("Index less than 0");
            if (end > arr.size()) throw std::invalid_argument("Index out of bounds");
            if (end < start) throw std::invalid_argument("Slice end smaller than start");
            this->len = end - start;
        }

        const double& operator[](size_t idx) const {
            return arr[begin + idx];
        }

        size_t size() const {
            /* Size of the slice */
            return len;
        }

        void operator=(const array1d& other) {
            /* Set from an other array */
            match_size(len, other.size());
            for (size_t i = 0; i < len; i++) arr[begin+i] = other[i];
        }

        void operator+=(const array1d& other) {
            /* In-place add from another array */
            match_size(len, other.size());
            for (size_t i = 0; i < len; i++) arr[begin+i] += other[i];
        }

        void operator-=(const array1d& other) {
            /* In-place subtract */
            match_size(len, other.size());
            for (size_t i = 0; i < len; i++) arr[begin+i] -= other[i];
        }

        void operator*=(const array1d& other) {
            /* In-place multiply */
            match_size(len, other.size());
            for (size_t i = 0; i < len; i++) arr[begin+i] *= other[i];
        }

        void operator/=(const array1d& other) {
            /* In-place divide */
            match_size(len, other.size());
            for (size_t i = 0; i < len; i++) arr[begin+i] /= other[i];
        }

        operator array1d() const {
            /* Cast into an array */
            return array1d(&arr[begin], len);
        }
    };


    template<typename T, typename std::enable_if<is_1d_array<T>::value, int>::type = 0>
    inline slice_wrapper<T> slice(T&& arr, size_t start, size_t end) {
        return slice_wrapper<T>(std::forward<T>(arr), start, end);
    }

    // type derived from array slice
    template <typename>
    struct derived_array_slice : std::false_type {};

    // is type array slice
    template <typename T>
    struct derived_array_slice<slice_wrapper<T>> : std::true_type {};

    // is type array, array-expr, or array slice?
    template <typename T>
    struct is_array_like : std::integral_constant<
                bool,
                is_1d_array<T>::value ||
                derived_array_expr<remove_cv_ref_t<T>>::value ||
                derived_array_slice<remove_cv_ref_t<T>>::value
            > {};


    template<typename T1, typename T2>
    inline typename std::enable_if<
        is_array_like<T1>::value && is_array_like<T2>::value, double
    >::type dot(const T1& vec1, const T2& vec2) {
        /* Calculate the vector dot product */
        size_t dim = vec1.size();
        match_size(dim, vec2.size());
        double sum = 0.0;
        for (size_t i = 0; i < dim; i++) {
            sum += vec1[i] * vec2[i];
        }
        return sum;
    }

    template<typename T>
    inline typename std::enable_if<
        is_array_like<T>::value, double
    >::type norm_l2(const T& vec) {
        /* Calculate the vector L2 norm */
        size_t dim = vec.size();
        double sum_sq = 0.0;
        for (size_t i = 0; i < dim; i++) {
            sum_sq += vec[i] * vec[i];
        }
        return std::sqrt(sum_sq);
    }

    template<typename T>
    inline typename std::enable_if<
        is_array_like<T>::value, double
    >::type rms_v(const T& vec) {
        /* Calculate the RMS of vector */
        double sqrt_dim = std::sqrt(static_cast<double>(vec.size()));
        return norm_l2(vec) / sqrt_dim;
    }

    template<typename T>
    inline typename std::enable_if<
        is_array_like<T>::value, double
    >::type abs_max(const T& vec) {
        /* Calculate the maximum absolute value of vector */
        size_t dim = vec.size();
        if (dim < 1) throw std::invalid_argument("Empty vector supplied");
        double absmax_val = std::abs(vec[0]);
        for (size_t i = 0; i < dim; i++) {
            auto tmp = std::abs(vec[i]);
            if (tmp > absmax_val) absmax_val = tmp;
        }
        return absmax_val;
    }


    /* Force no-alias assignment on an array if requested */
    class noalias_wrapper {

    private:
        array1d& arr;

    public:

        noalias_wrapper() = delete;

        explicit noalias_wrapper(array1d& array) : arr(array) {}

        template <typename T,
                  typename std::enable_if<is_array_like<T>::value, int>::type = 0>
        void operator=(const T& expr) {
            /* Assign assuming no aliasing */
            const size_t dim = expr.size();
            if (arr.size() != dim) arr.resize(dim);
            for (size_t i = 0; i < dim; i++) arr[i] = expr[i];
        }

    };

    inline noalias_wrapper noalias(array1d& arr) {
        return noalias_wrapper(arr);
    }

    // is type double, ignoring const and ref?
    template <typename T>
    struct is_double : std::integral_constant<
                        bool,
                        std::is_same<double, remove_cv_ref_t<T>>::value
                    > {};

    // are operand types allowed for overloaded operators?
    template <typename T1, typename T2>
    struct allowed_operands : std::integral_constant<
                    bool,
                    (is_array_like<T1>::value && is_array_like<T2>::value) ||
                    (is_array_like<T1>::value && is_double<T2>::value) ||
                    (is_double<T1>::value && is_array_like<T2>::value)
                > {};

    // strip const and ref from double and leave other types unchanged
    template <typename T>
    using operand_transform_t = typename std::conditional<
                                        is_double<T>::value, double, T
                                    >::type;

    // enable template only if operands are valid
    template <typename X, typename Y>
    using operator_enable_t = typename std::enable_if<allowed_operands<X,Y>::value, int>::type;

    /* Binary operators */

    template <typename X, typename Y, operator_enable_t<X, Y> = 0>
    inline array_expr<operand_transform_t<X>, operand_transform_t<Y>, operation::add>
    operator+(X&& v1, Y&& v2) {
        return array_expr<operand_transform_t<X>, operand_transform_t<Y>, operation::add>(
            std::forward<X>(v1), std::forward<Y>(v2)
        );
    }

    template <typename X, typename Y, operator_enable_t<X, Y> = 0>
    inline array_expr<operand_transform_t<X>, operand_transform_t<Y>, operation::subtract>
    operator-(X&& v1, Y&& v2) {
        return array_expr<operand_transform_t<X>, operand_transform_t<Y>, operation::subtract>(
            std::forward<X>(v1), std::forward<Y>(v2)
        );
    }

    template <typename X, typename Y, operator_enable_t<X, Y> = 0>
    inline array_expr<operand_transform_t<X>, operand_transform_t<Y>, operation::multiply>
    operator*(X&& v1, Y&& v2) {
        return array_expr<operand_transform_t<X>, operand_transform_t<Y>, operation::multiply>(
            std::forward<X>(v1), std::forward<Y>(v2)
        );
    }

    template <typename X, typename Y, operator_enable_t<X, Y> = 0>
    inline array_expr<operand_transform_t<X>, operand_transform_t<Y>, operation::divide>
    operator/(X&& v1, Y&& v2) {
        return array_expr<operand_transform_t<X>, operand_transform_t<Y>, operation::divide>(
            std::forward<X>(v1), std::forward<Y>(v2)
        );
    }

    /* Unary operators */
    template <typename X, typename std::enable_if<is_array_like<X>::value, int>::type = 0>
    inline array_expr<X, void, operation::negate> operator-(X&& var) {
        return array_expr<X, void, operation::negate>(std::forward<X>(var));
    }

}

#endif // ADE_ARRX_HELPER_HPP
