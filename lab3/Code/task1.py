import numpy as np

def mononomial(values):
    n = len(values)
    A = np.array([ [values[j][0] ** i for i in range(n)] for j in range(n) ])
    b = np.array(list(zip(*values))[1])
    coefficients = np.linalg.solve(A,b)
    for i, coeff in enumerate(coefficients):
        x = f"x ^ {i}" if i > 0 else ''
        print(f"{round(coeff,3)} * {x} + ", end='')
    print()
    return np.poly1d(coefficients)


def Lagrange_polynomial(values):
    x_values, y_values = zip(*values)
    n = len(values)
    def L(i, x):
        nonlocal x_values
        numer = np.prod([(x - x_values[j]) for j in range(len(x_values)) if j != i])
        denom = np.prod([(x_values[i] - x_values[j]) for j in range(len(x_values)) if j != i])
        return numer / denom
    
    def P(x):
        return sum(y_values[i] * L(i, x) for i in range(len(x_values)))
    
    return P


def Newton_approximation(values):
    x_values, y_values = zip(*values)
    
    def divided_differences(x_values, y_values):
        n = len(x_values)
        coef = list(y_values)
        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                coef[i] = (coef[i] - coef[i - 1]) / (x_values[i] - x_values[i - j])
        return coef
    
    coeffs = divided_differences(x_values, y_values)
    
    def P(x):
        n = len(coeffs)
        result = coeffs[-1]
        for i in range(n - 2, -1, -1):
            result = result * (x - x_values[i]) + coeffs[i]
        return result
    
    return P

def test_interpolation(values):
    poly_mono = mononomial(values)
    poly_lagrange = Lagrange_polynomial(values)
    poly_newton = Newton_approximation(values)
    test_xs = [-3,-2,-1,0, 1, 2, 3]
    for x in test_xs:
        print(f"{x} & {poly_mono(x)} & {poly_lagrange(x)} & {poly_newton(x)} \\\\")

def main():
    interpolation_points = [(0.0, 0.0), (0.5, 1.6), (1.0, 2.0), (6.0, 2.0), (7.0, 1.5), (9.0, 0.0)]
    for point in interpolation_points:
        print(point)
    test_interpolation(interpolation_points)

if __name__ == "__main__":
    main()
