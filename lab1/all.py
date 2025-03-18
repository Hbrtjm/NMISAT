import numpy as np
import math

def find_machine_epsilon():
    eps = 1.0
    while (1.0 + eps) > 1.0:
        eps /= 2
    return eps * 2

def absolute_error(x, h):
    return abs(math.sin(x + h) - math.sin(x))

def relative_error(x, h):
    if math.sin(x) == 0:
        return float('inf')  # unbounded error
    return abs(h * math.cos(x) / math.sin(x))

def condition_number(x):
    if math.sin(x) == 0:
        return float('inf')
    return abs(x * math.cos(x) / math.sin(x))

def progressive_error_approx1(x):
    return abs(math.sin(x) - x)

def progressive_error_approx2(x):
    return abs(math.sin(x) - (x - x**3 / 6))

def backward_error_approx1(x):
    return abs(x - math.asin(x))

def backward_error_approx2(x):
    return abs(x - math.asin(x - x**3 / 6))

def underflow_level(beta, L):
    return beta**L

test_values = [0.1, 0.5, 1.0]

precision = 4


for value in test_values:
    print(f"{value} & {round(backward_error_approx1(value),precision)} & {round(progressive_error_approx1(value),precision)} \\\\")

print("Approximation 2")

for value in test_values:
    print(f"{value} & {round(backward_error_approx2(value),precision)} & {round(progressive_error_approx2(value),precision)} \\\\")


def subtraction_result(x, y, UFL):
    result = x - y
    return result if result >= UFL else 0.0

# Obliczenia
machine_epsilon = find_machine_epsilon()
ufl = underflow_level(10, -98)
sub_result = subtraction_result(6.87e-97, 6.81e-97, ufl)

h = 1e-5  # małe zakłócenie

precision = 8

for x in test_values:
    abs_err = absolute_error(x, h)
    rel_err = relative_error(x, h)

    cond_num = condition_number(x)

    print(f"x = {x}")
    print(f"  Absolute error: {round(abs_err,precision)}")
    print(f"  Relative error: {round(rel_err,precision)}")
    print(f"  Condition number: {round(cond_num,precision)}")
    
# print(f"Machine epsilon: {machine_epsilon}")
# print(f"Underflow level: {ufl}")
# print(f"Subtraction result (6.87e-97 - 6.81e-97): {sub_result}")
