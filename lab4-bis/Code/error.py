import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
def absolute_value(x):
    return abs(x)

def approximation_function(x):
    #The approximation function.
    #Function: (2/(5*π)) + (24*x²/(5*π)) - (32*x⁴/(15*π))
    return (2/(5*np.pi)) + (24*x**2/(5*np.pi)) - (32*x**4/(15*np.pi))

def calculate_error(x):
    return abs(absolute_value(x) - approximation_function(x))

def generate_error_table(start, end, num_points):
    x_values = np.linspace(start, end, num_points)
    
    table_data = []
    for x in x_values:
        abs_val = absolute_value(x)
        approx = approximation_function(x)
        error = calculate_error(x)
        
        table_data.append([x, abs_val, approx, error])
    
    headers = ["x", "|x|", "Approximation", "Error"]
    return tabulate(table_data, headers, floatfmt=".6f")

def plot_functions(start, end, num_points):
    x_values = np.linspace(start, end, num_points)
    abs_values = [absolute_value(x) for x in x_values]
    approx_values = [approximation_function(x) for x in x_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, abs_values, 'b-', label='|x|')
    plt.plot(x_values, approx_values, 'r--', label='Approximation')
    plt.grid(True)
    plt.legend()
    plt.title('Absolute Value vs. Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('function_comparison.png')
    plt.close()
    
    # Plot the error
    error_values = [calculate_error(x) for x in x_values]
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, error_values, 'g-')
    plt.grid(True)
    plt.title('Error: |x| - Approximation')
    plt.xlabel('x')
    plt.ylabel('Error')
    plt.savefig('error_plot.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    start = -0.8
    end = 1.0
    num_points = 10    
    error_table = generate_error_table(start, end, num_points)
    print("\nError Table:")
    print(error_table)
    
    plot_functions(start, end, 200) 
    print("\nPlots saved as 'function_comparison.png' and 'error_plot.png'")
    x_values = np.linspace(start, end, 1000)
    errors = [calculate_error(x) for x in x_values]
    max_error = max(errors)
    max_error_x = x_values[errors.index(max_error)]
    
    print(f"\nMaximum error: {max_error:.8f} at x = {max_error_x:.6f}")
