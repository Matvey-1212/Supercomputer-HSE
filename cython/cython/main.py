import trapezoid_method

input_list = list(map(int, input().split(" ")))

num_points = input_list[0]  
a0 = input_list[1]
b0 = input_list[2]
nrepeat = 1000  


res = trapezoid_method.benchmark(num_points, a0, b0, nrepeat)

print('Number of trapezoids: ' + str(num_points))
print('AVG Timing: ' + str(res['btime']) + ' ms')
print('Timing: ' + str(res['alltime']) + ' ms')
print('Answer = ' + str(res['answer']))
print()