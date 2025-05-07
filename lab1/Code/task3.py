from math import sin, asin

def progressive_error_approx1(x):
    return abs(sin(x) - x)

def progressive_error_approx2(x):
    return abs(sin(x) - (x - x**3 / 6))

def backward_error_approx1(x):
    return abs(x - asin(x))

def backward_error_approx2(x):
    return abs(x - asin(x - x**3 / 6))

def underflow_level(beta, L):
    return beta**L

test_values = [0.1, 0.5, 1.0]
precision = 4

print("Approximation 1")

for value in test_values:
    print(f"{value} & {round(backward_error_approx1(value),precision)} & {round(progressive_error_approx1(value),precision)} \\ ")

print("Approximation 2")

for value in test_values:
    print(f"{value} & {round(backward_error_approx2(value),precision)} & {round(progressive_error_approx2(value),precision)} \\ ")
