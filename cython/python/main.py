import time

def funct(x):
    return 1 / (1 + x**2)

def integral(num_points, a0, b0):
    h = (b0 - a0) / num_points
    integral = 0.5 * (funct(a0) + funct(b0))

    for i in range(1, num_points):
        a0 += h
        integral += funct(a0)
    return integral * h

def benchmark(fn, num_points, a0, b0, nrepeat):
    start = time.time()
    for _ in range(nrepeat):
        result = fn(num_points, a0, b0)
    end = time.time()
    alltime = (end - start) * 1000
    btime = alltime / nrepeat
    return result, btime, alltime


input_list = list(map(int, input().split(" ")))

num_points = input_list[0]  
a0 = input_list[1]
b0 = input_list[2]
nrepeat = 1000  

result, avg_timing, timing = benchmark(integral, num_points, a0, b0, nrepeat)

print('Number of trapezoids: ' + str(num_points))
print('AVG Timing: ' + str(avg_timing) + ' ms')
print('Timing: ' + str(timing) + ' ms')
print('Answer = ' + str(result))
print()