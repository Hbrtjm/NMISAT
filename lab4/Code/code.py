import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

def point_approximation_second_degree(x_points, y_points):
    """
    Implement point approximation method using second-degree polynomials.
    
    Parameters:
    x_points (array-like): x-coordinates of data points
    y_points (array-like): y-coordinates of data points (function values)
    
    Returns:
    tuple: (a, b, c) coefficients for the polynomial a + bx + cx^2
    """
    n = len(x_points)
    
    # Build the coefficient matrix A
    A = np.zeros((3, 3))
    A[0, 0] = n
    A[0, 1] = np.sum(x_points)
    A[0, 2] = np.sum(x_points**2)
    A[1, 0] = np.sum(x_points)
    A[1, 1] = np.sum(x_points**2)
    A[1, 2] = np.sum(x_points**3)
    A[2, 0] = np.sum(x_points**2)
    A[2, 1] = np.sum(x_points**3)
    A[2, 2] = np.sum(x_points**4)
    
    # Build the right-hand side vector b
    b = np.zeros(3)
    b[0] = np.sum(y_points)
    b[1] = np.sum(x_points * y_points)
    b[2] = np.sum(x_points**2 * y_points)
    
    # Solve the system of equations
    coefficients = solve(A, b)
    
    return coefficients

def plot_approximation(x_points, y_points, coefficients, function=None):
    """
    Plot the data points and the approximation polynomial.
    
    Parameters:
    x_points (array-like): x-coordinates of data points
    y_points (array-like): y-coordinates of data points
    coefficients (array-like): (a, b, c) coefficients for polynomial a + bx + cx^2
    function (callable, optional): The original function to compare with
    """
    a, b, c = coefficients
    
    # Generate points for smooth curve
    x_smooth = np.linspace(min(x_points), max(x_points), 100)
    y_approx = a + b * x_smooth + c * x_smooth**2
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_points, y_points, color='red', label='Data Points')
    plt.plot(x_smooth, y_approx, color='blue', label=f'Approximation: {a:.4f} + {b:.4f}x + {c:.4f}x²')
    
    if function is not None:
        y_func = function(x_smooth)
        plt.plot(x_smooth, y_func, color='green', linestyle='--', label='Original Function')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Second-Degree Polynomial Approximation')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate and print the mean squared error
    y_fitted = a + b * np.array(x_points) + c * np.array(x_points)**2
    mse = np.mean((np.array(y_points) - y_fitted)**2)
    print(f"Mean Squared Error: {mse:.6f}")

# Example usage
if __name__ == "__main__":
    # Example data for f(x) = 1 + x^3
    x = np.linspace(0, 1, 10)
    f = lambda x: 1 + x**3
    y = f(x)
    
    coefficients = point_approximation_second_degree(x, y)
    print(f"Approximation: p(x) = {coefficients[0]:.4f} + {coefficients[1]:.4f}x + {coefficients[2]:.4f}x²")
    
    plot_approximation(x, y, coefficients, f)
