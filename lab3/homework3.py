# homework3.py
from scipy.interpolate import CubicSpline
import numpy as np

# Zadanie 3: Interpolacja sklejanymi funkcjami sześciennymi
def cubic_spline_interpolation(x0, x1, x2, y0, y1, y2):
    x_vals = [x0, x1, x2]
    y_vals = [y0, y1, y2]
    # TODO
    spline = CubicSpline(x_vals, y_vals, bc_type='not-a-knot')
    return spline

# Przykładowe wartości testowe
x0, x1, x2 = 0, 1, 2
y0, y1, y2 = 1, 2, 0.5
spline_func = cubic_spline_interpolation(x0, x1, x2, y0, y1, y2)
test_xs = np.linspace(x0, x2, 50)
test_ys = spline_func(test_xs)

print(f"Interpolowane wartości dla testowych x: {test_ys}")
