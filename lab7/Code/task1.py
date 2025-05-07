from numpy import cos, sin, exp
import matplotlib.pyplot as plt
import numpy as np

def approx(f=lambda x: x, epsilon=1e-16):
    a = -10  # Starting interval lower bound
    b = 10   # Starting interval upper bound

    if f(a) * f(b) > 0:
        raise ValueError("Function must have opposite signs at interval endpoints")

    x = (a + b) / 2
    iterations = 0

    while abs(f(x)) > epsilon and iterations < 1000:
        if f(a) * f(x) < 0:
            b = x
        else:
            a = x
        x = (a + b) / 2
        iterations += 1

    return x, iterations

def newton_approx(f=lambda x: x, f_der=lambda x: 1, x0=1.0, epsilon=1e-16, max_iter=100):
    x = x0
    iterations = 0

    while abs(f(x)) > epsilon and iterations < max_iter:
        if abs(f_der(x)) < 1e-10:
            raise ValueError("Derivative too close to zero - divergent")
        x = x - f(x)/f_der(x)
        iterations += 1

    return x, iterations

def f1(x):
    return x * cos(x) - 1

def f1_der(x):
    return cos(x) - x * sin(x)

def f2(x):
    return x**3 - 5*x - 6

def f2_der(x):
    return 3 * x**2 - 5

def f3(x):
    return exp(-x) - x**2 + 1

def f3_der(x):
    return -exp(-x) - 2*x

def plot_function(f, x_range, title):
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = [f(xi) for xi in x]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.grid(True)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

def compare_methods(f, f_der, x0_values, epsilon_values, title):
    results = []

    for x0 in x0_values:
        for eps in epsilon_values:
            try:
                root, iterations = newton_approx(f, f_der, x0, eps)
                results.append({
                    'x0': x0,
                    'epsilon': eps,
                    'root': root,
                    'iterations': iterations,
                    'final_error': abs(f(root))
                })
            except ValueError as e:
                print(f"Error with x0={x0}, epsilon={eps}: {e}")

    # Print results in a table format
    print(f"\nResults for {title}:")
    print(f"{'x0':^10} | {'epsilon':^12} | {'root':^15} | {'iterations':^10} | {'final error':^12}")
    print("-" * 65)

    for r in results:
        print(f"{r['x0']:^10} | {r['epsilon']:^12.1e} | {r['root']:^15.10f} | {r['iterations']:^10} | {r['final_error']:^12.2e}")

    return results

def get_different_roots(results, epsilon=1e-4):
    roots = [] 
    for element in results:
        root = element['root']
        if len(roots) == 0:
            roots.append(root)
        for other in roots:
            if abs(root - other) > epsilon:
                roots.append(root)
    return roots


def main():
    epsilon_values = [1e-6, 1e-10, 1e-14]
    x0_values = [0.5, 1.0, 2.0]

    plot_function(f1, [-2, 5], "Function (a): x*cos(x) - 1")
    plot_function(f2, [-4, 4], "Function (b): x^3 - 5x - 6")
    plot_function(f3, [-2, 2], "Function (c): e^(-x) - x^2 + 1")

    print("\n=== NEWTON'S METHOD COMPARISONS ===")

    print("\nEquation (a): x*cos(x) = 1")
    results_a = compare_methods(f1, f1_der, x0_values, epsilon_values, "x*cos(x) = 1")

    print("\nEquation (b): x^3 - 5x - 6 = 0")
    results_b = compare_methods(f2, f2_der, x0_values, epsilon_values, "x^3 - 5x - 6 = 0")

    print("\nEquation (c): e^(-x) = x^2 - 1")
    results_c = compare_methods(f3, f3_der, x0_values, epsilon_values, "e^(-x) = x^2 - 1")

    print("\n=== FINAL SOLUTIONS ===") 
    # solution_a, iters_a = newton_approx(f1, f1_der, 1.0)
    solutions_a = get_different_roots(results_a)
    print(f"Solution for x*cos(x) = 1: x ≈ {solutions_a} (found in iterations)")

    solutions_b = get_different_roots(results_b)
    # solution_b, iters_b = newton_approx(f2, f2_der, 2.0)
    print(f"Solution for x^3 - 5x - 6 = 0: x ≈ {solutions_b} (found in iterations)")

    solutions_c  = get_different_roots(results_c)
    # solution_c, iters_c = newton_approx(f3, f3_der, 1.0)
    print(f"Solution for e^(-x) = x^2 - 1: x ≈ {solutions_c} (found in iterations)")

if __name__ == "__main__":
    main()
