#include <iomanip>
#include <iostream>
#include <cmath>
#include <cstddef>

#include <omp.h>

using std::cin;
using std::cout;
using ptrdiff_t = std::ptrdiff_t;
using size_t = std::size_t;

const double PI = 3.141592653589793;

// integrate exp(-x^2 / 2) on (-10, 10), must be approx. sqrt(2 * pi)
double pi_integ(ptrdiff_t npoints)
{
    double xmin = -10, xmax = 10;
    double h = (xmax - xmin) / npoints;
    double integral = 0;

    #pragma omp parallel for reduction(+: integral)
    for (ptrdiff_t i=0; i<npoints; i++)
    {
        double x_i = xmin + (i + 0.5) * h;
        double func_i = exp(- x_i * x_i / 2);
        
        integral += func_i;
        
    }
    integral *= h;
    double pi_est = integral * integral / 2;
    return pi_est;
}

int main(int argc, char** argv)
{
    ptrdiff_t npoints;

    cin >> npoints;

    double t1 = omp_get_wtime();

    double pi_est = pi_integ(npoints);

    double t2 = omp_get_wtime();
    
    cout << "Computed average: " << std::setprecision(16) << pi_est << std::endl;
    cout << "Exact average: " << std::setprecision(16) << PI << std::endl;
    cout << "Time: " << t2 - t1 << std::endl;

    return 0;
}

