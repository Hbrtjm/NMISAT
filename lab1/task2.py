from math import sin, cos

def find_machine_epsilon():
    eps = 1.0
    while (1.0 + eps) > 1.0:
        eps /= 2
    return eps * 2

def absolute_error(x, h):
    return abs(sin(x + h) - sin(x))

def relative_error(x, h):
    if sin(x) == 0:
        return float('inf')  # unbounded error
    return abs(h * cos(x) / sin(x))

def condition_number(x):
    if sin(x) == 0:
        return float('inf')
    return abs(x * cos(x) / sin(x))

h = 1e-5  # małe zakłócenie

precision = 8

test_values = [0.1, 0.5, 1.0]

for x in test_values:
    abs_err = absolute_error(x, h)
    rel_err = relative_error(x, h)
    cond_num = condition_number(x)
    print(f"x = {x}")
    print(f"  Absolute error: {round(abs_err,precision)}")
    print(f"  Relative error: {round(rel_err,precision)}")
    print(f"  Condition number: {round(cond_num,precision)}")