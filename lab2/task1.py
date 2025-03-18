from math import exp

def facts(n):
    if n == 0:
        return [1]
    results = [1]
    fact = 1
    for i in range(1, n):
        fact *= i
        results.append(fact)
    return results

def maclaurin_exp(x, epsilon=1e-15):
    fact_size = 10
    acc = 1
    fact = facts(fact_size)
    i = 0
    result = 0
    if x == 0:
        return 1
    while abs(acc / fact[i]) > epsilon:
        if i + 1 == fact_size:
            fact_size *= 2
            fact = facts(fact_size)
        result += acc / fact[i]
        acc *= x
        i += 1
    return result

def maclaurin_exp_negative(x, epsilon=1e-15):
    return 1 / maclaurin_exp(-x, epsilon)

def horner_maclaurin_exp(x, n=100, epsilon=1e-15):
    fact = facts(n)
    result = 1 / fact[n - 1]  # Initialize last term
    for i in range(n - 2, -1, -1):
        result = 1 / fact[i] + x * result
    return result


values = [-10, -5, -1, 1, 5, 10]
epsilon = 1e-15

for value in values:
    my_exp_value = maclaurin_exp(value, epsilon) if value >= 0 else maclaurin_exp_negative(value, epsilon)
    lib_exp_value = exp(value)
    my_horn_exp_value = horner_maclaurin_exp(value,)
    print(f"x = {value}")
    print(f"My exp: {my_exp_value}")
    print(f"Library exp: {lib_exp_value}")
    print(f"Using horner_maclaurin: {my_horn_exp_value}")
    print(f"Error between my_exp and library exp: {my_exp_value - lib_exp_value}")
    print(f"Error between my_exp and my_horn_exp: {my_exp_value - my_horn_exp_value}")
    print(f"Error between mmy_horn_exp and library exp: {lib_exp_value - my_horn_exp_value}\n")
