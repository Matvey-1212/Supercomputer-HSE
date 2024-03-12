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

template <class T>
void fill_random(matrix<T> x, T xmin, T xmax, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(xmin, xmax);
    for (ptrdiff_t i = 0; i < x.length(); i++)
    {
        x(i) = dist(rng);
    }
}

template <class T>
matrix<T> matmul_ikj(matrix<T> a, matrix<T> b)
{
    ptrdiff_t rowa = a.nrows();
    ptrdiff_t cola = a.ncols();
    ptrdiff_t colb = b.ncols();
    ptrdiff_t i, j, k;

    matrix<T> c(rowa, colb);
    for (i=0; i < c.length(); i++)
    {
        c(i) = 0;
    }
    for (i=0; i<rowa; i++)
    {
        for (k=0; k<cola; k++)
        {
            T a_ik = a(i, k);
            for (j=0; j<colb; j++)
            {
                c(i, j) += a_ik * b(k,j);
            }
        }
    }
    return c;
}

int main(int argc, char* argv[])
{
    ptrdiff_t n;

    std::cin >> n;
    matrix<double> a(n, n);
    matrix<double> b(n, n);

    fill_random(a, -1.0, 1.0, 9876);
    fill_random(b, -1.0, 1.0, 9877);

    double t0 = omp_get_wtime();

    matrix<double> c = matmul_ikj(a, b);

    double t1 = omp_get_wtime();

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
              << "Timing: " << t1 - t0 << " sec\n"
              << "Answer[0, 0] = " << c(0, 0)
              << std::endl;
    return 0;
}
