import cython
import time

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double integral(int num, double a0, double b0):
    cdef int i
    cdef double h = (b0 - a0) / num
    cdef double integral_val = 0.5 * (1/(1 + a0 * a0) + 1/(1 + b0 * b0))

    for i in range(1, num):
        a0 += h
        integral_val += 1/(1 + a0 * a0)

    return integral_val * h

@cython.boundscheck(False)
@cython.wraparound(False)
def benchmark(int num, double a0, double b0, int nrepeat):
    cdef double start = time.time()
    cdef double answer = 0.0
    for i in range(nrepeat):
        answer = integral(num, a0, b0)
    cdef double end = time.time()
    cdef double duration_s = (end - start) * 1000
    cdef double ms_per_run = duration_s / nrepeat
    return {"answer": answer, "btime": ms_per_run, "alltime": duration_s}