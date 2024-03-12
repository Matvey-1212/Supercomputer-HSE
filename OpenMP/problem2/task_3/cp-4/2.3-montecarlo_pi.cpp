#include <iomanip>
#include <iostream>
#include <cmath>
#include <cstddef>
#include <random>

#include <omp.h>

#include "array_types.hpp"

using std::cin;
using std::cout;
using ptrdiff_t = std::ptrdiff_t;
using size_t = std::size_t;

const double PI = 3.141592653589793;



double mc_pi(ptrdiff_t niter, size_t seed)
{
    ptrdiff_t pts_inside_circle = 0;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    #pragma omp parallel
    {
        std::mt19937_64 rng(seed + omp_get_thread_num());
        

        #pragma omp for reduction(+: pts_inside_circle)
        for (ptrdiff_t i = 0; i < niter; ++i)
        {
            double x = dist(rng), y = dist(rng);
            pts_inside_circle += (x * x + y * y <= 1);
        }
    }
    double pi_est = ((double) 4.0) * pts_inside_circle / niter;
    return pi_est;
}

int main(int argc, char** argv)
{
    ptrdiff_t niter;

    cin >> niter;

    double t1 = omp_get_wtime();

    double pi_est = mc_pi(niter, 4321);

    double t2 = omp_get_wtime();
    
    cout << "Computed average: " << std::setprecision(16) << pi_est << std::endl;
    cout << "Exact average: " << std::setprecision(16) << PI << std::endl;
    cout << "Time: " << t2 - t1 << std::endl;
    cout << "niter: " << niter << std::endl;
    return 0;
}