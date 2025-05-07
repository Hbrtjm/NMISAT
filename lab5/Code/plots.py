import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from numpy.polynomial.chebyshev import chebfit, chebval

# Experimental data
x_data = np.array([0.0, 0.5, 1.0, 6.0, 7.0, 9.0])
y_data = np.array([0.0, 1.6, 2.0, 2.0, 1.5, 0.0])

# Points to be ploted 
x_plot = np.linspace(0, 9, 1000)

def polynomial_interpolation(x, y):
    A = np.vstack([x**i for i in range(6)]).T
    coeffs = np.linalg.solve(A, y)
    
    def poly_func(x_new):
        result = np.zeros_like(x_new, dtype=float)
        for i, coeff in enumerate(coeffs):
            result += coeff * (x_new ** i)
        return result
    
    return poly_func, coeffs

def cubic_spline_interpolation(x, y):
    spline = CubicSpline(x, y)
    return spline

def chebyshev_interpolation(x, y, degree=5):
    x_scaled = 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1
    
    coeffs = chebfit(x_scaled, y, degree)
    def cheb_func(x_new):
        x_new_scaled = 2 * (x_new - np.min(x)) / (np.max(x) - np.min(x)) - 1
        return chebval(x_new_scaled, coeffs)
    
    return cheb_func, coeffs

poly_func, poly_coeffs = polynomial_interpolation(x_data, y_data)
spline_func = cubic_spline_interpolation(x_data, y_data)
cheb_func, cheb_coeffs = chebyshev_interpolation(x_data, y_data)

y_poly = poly_func(x_plot)
y_spline = spline_func(x_plot)
y_cheb = cheb_func(x_plot)

plt.figure(figsize=(15, 10))

# Polynomial interpolation
plt.subplot(2, 2, 1)
plt.plot(x_plot, y_poly, 'r-', label='Wielomian stopnia 5')
plt.plot(x_data, y_data, 'bo', label='Dane pomiarowe')
plt.title('(a) Interpolacja wielomianem stopnia 5')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()

# Spline
plt.subplot(2, 2, 2)
plt.plot(x_plot, y_spline, 'g-', label='Cubic Spline')
plt.plot(x_data, y_data, 'bo', label='Dane pomiarowe')
plt.title('(b) Interpolacja funkcją sklejaną stopnia 3')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()

# Chebyshev interpolation
plt.subplot(2, 2, 3)
plt.plot(x_plot, y_cheb, 'm-', label='Wielomiany Czebyszewa')
plt.plot(x_data, y_data, 'bo', label='Dane pomiarowe')
plt.title('(c) Interpolacja wielomianami Czebyszewa')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()

# Method comparison 
plt.subplot(2, 2, 4)
plt.plot(x_plot, y_poly, 'r-', label='Wielomian stopnia 5')
plt.plot(x_plot, y_spline, 'g-', label='Cubic Spline')
plt.plot(x_plot, y_cheb, 'm-', label='Wielomiany Czebyszewa')
plt.plot(x_data, y_data, 'bo', label='Dane pomiarowe')
plt.title('(d) Porównanie metod interpolacji')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print("Polynomial coefficients:")
for i, coeff in enumerate(poly_coeffs):
    x = f"x ^ {i}" if i > 0 else ''
    print(f"{round(coeff,10)} * {x} + ", end='')
print()
