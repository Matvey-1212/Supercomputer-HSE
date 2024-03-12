#include <iostream>
#include <memory>
#include <cstdint>
#include <iomanip>
#include <random>
#include "array_types.hpp"

#include <omp.h>

using ptrdiff_t = std::ptrdiff_t;
using size_t = std::size_t;

template <class T>
void fill_random(vec<T> x, T xmin, T xmax, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(xmin, xmax);
    for (ptrdiff_t i = 0; i < x.length(); i++)
    {
        x(i) = dist(rng);
    }
}

// compute q = z' * A * z, where A = I + 0.5 * (x * y' + y * x')
// I is a unit matrix
template <class T>
T quadratic_form(vec<T> x, vec<T> y, vec<T> z)
{
    T Q = 0;
    #pragma omp parallel for schedule(dynamic, 32) reduction(+: Q)
    for (int i = 0; i < z.length(); i++)
    {
        Q += z(i) * z(i) * (1 + x(i) * y(i));
        T x_i = x(i), y_i = y(i);
        for (int j = i+1; j < z.length(); j++)
        {
            Q += z(i) * z(j) * (x_i * y(j) + x(j) * y_i);
        }
    }
    return Q;
}

int main(int argc, char* argv[])
{
    ptrdiff_t n;

    std::cin >> n;
    vec<double> x(n);
    vec<double> y(n);
    vec<double> z(n);

    fill_random(x, -1.0, 1.0, 9876);
    fill_random(y, -1.0, 1.0, 9877);
    fill_random(z, -10.0, 10.0, 9878);

    double t0 = omp_get_wtime();

    double q = quadratic_form(x, y, z);

    double t1 = omp_get_wtime();

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
              << "Timing: " << t1 - t0 << " sec\n"
              << "Answer = " << q
              << std::endl;
    return 0;
}
