#include <iostream>
#include <memory>
#include <cstdint>
#include <iomanip>
#include "array_types.hpp"

#include <functional>
#include <chrono>

using intptr_t = std::intptr_t;

template <class T>
struct benchresult {
    T result;
    double btime;
    double alltime;
};

template <class T, class input_type>
auto benchmark(std::function<T(input_type)> fn, input_type input, intptr_t nrepeat){
    T result;
    auto start = std::chrono::steady_clock::now();
    for (intptr_t i = 0; i < nrepeat; i++) {
        result = fn(input);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration_s = end - start;
    double ms_per_run = duration_s.count() * 1000 / nrepeat;
    return benchresult<T> {result, ms_per_run, duration_s.count() * 1000};
}

template <typename T> T funct(T x)
{
	return 1 / (1 + x * x);
}


template <class T>
T integral(ptrdiff_t num_points, double a0, double b0)
{    
	T h = (b0 - a0) / num_points;
    T sum = 0.5 * (funct(a0) + funct(b0));
    for (ptrdiff_t i = 1; i < num_points; ++i) {
        a0 += h;
        sum += funct(a0);
    }
    return sum * h;
}

int main(int argc, char* argv[])
{
    double a0, b0;    
    ptrdiff_t num_points;

    std::cin >> num_points >> a0 >> b0;
    
    std::function<double(int)> int_value = [=](int idx) {return integral<double>(num_points, a0, b0);};

    auto benchresult = benchmark(int_value, 0, 1000);

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
              << "Number of trapezoids: " << num_points << "\n"
              << "AVG Timing: " << benchresult.btime << " ms\n"
              << "Timing: " << benchresult.alltime << " ms\n"
              << "Answer = " << benchresult.result
              << std::endl << std::endl;
    return 0;
}
