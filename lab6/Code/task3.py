import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

def function(x):
    return 1 / (1 + x**2)

def rectangular_integration(function, a, b, n):
    """Simple rectangular integration with n subdivisions"""
    dx = (b - a) / n
    return sum([dx * function(a + dx * (i + 0.5)) for i in range(n)])

def simpson_rule(function, a, b):
    """Simple Simpson's rule for a single interval"""
    mid = (a + b) / 2
    h = (b - a) / 2
    return (h / 3) * (function(a) + 4 * function(mid) + function(b))

def trapezoidal_rule(function, a, b):
    """Simple trapezoidal rule for a single interval"""
    h = b - a
    return (h / 2) * (function(a) + function(b))

def adaptive_simpson(function, a, b, eps, level=0, max_level=50):
    """Adaptive Simpson's rule implementation"""
    # Compute Simpson's rule for the entire interval
    whole_interval = simpson_rule(function, a, b)
    
    # Compute Simpson's rule for the two halves
    mid = (a + b) / 2
    left_interval = simpson_rule(function, a, mid)
    right_interval = simpson_rule(function, mid, b)
    
    # Combine the two half intervals
    both_halves = left_interval + right_interval
    
    # Estimate the error
    error = abs(whole_interval - both_halves) / 15
    
    # If error is small enough or max recursion level reached, return result
    if (error < eps) or (level >= max_level):
        return both_halves + (both_halves - whole_interval) / 15
    
    # Otherwise, recurse on the two halves with half the error tolerance
    else:
        left_result = adaptive_simpson(function, a, mid, eps/2, level+1, max_level)
        right_result = adaptive_simpson(function, mid, b, eps/2, level+1, max_level)
        return left_result + right_result

def adaptive_integral(function, a, b, eps, level=0, max_level=50, intervals=None):
    """Adaptive integration using trapezoidal rule with visualization data"""
    if intervals is not None and level < max_level:
        intervals.append((a, b, level))
    
    # Compute trapezoidal rule for the entire interval
    whole_interval = trapezoidal_rule(function, a, b)
    
    # Compute trapezoidal rule for the two halves
    mid = (a + b) / 2
    left_interval = trapezoidal_rule(function, a, mid)
    right_interval = trapezoidal_rule(function, mid, b)
    
    # Combine the two half intervals
    both_halves = left_interval + right_interval
    
    # Estimate the error
    error = abs(whole_interval - both_halves)
    
    # If error is small enough or max recursion level reached, return result
    if (error < eps) or (level >= max_level):
        return both_halves
    
    # Otherwise, recurse on the two halves with half the error tolerance
    else:
        left_result = adaptive_integral(function, a, mid, eps/2, level+1, max_level, intervals)
        right_result = adaptive_integral(function, mid, b, eps/2, level+1, max_level, intervals)
        return left_result + right_result

def exact_integral():
    """The exact value of ∫[0,1] 1/(1+x²) dx is arctan(1) = π/4"""
    return np.arctan(1)

def visualize_rectangular(ax, function, a, b, n):
    """Visualize rectangular integration"""
    x = np.linspace(a, b, 1000)
    y = [function(i) for i in x]
    
    dx = (b - a) / n
    
    # Plot the function
    ax.plot(x, y, 'b-', lw=2, label='f(x) = 1/(1+x²)')
    
    # Plot the rectangles
    for i in range(n):
        left = a + i * dx
        mid = left + dx / 2
        height = function(mid)
        rect = Rectangle((left, 0), dx, height, facecolor='r', alpha=0.3)
        ax.add_patch(rect)
    
    ax.set_title(f'Rectangular Integration (n={n})')
    ax.legend()
    ax.grid(True)

def visualize_adaptive(ax, function, a, b, eps):
    """Visualize adaptive integration"""
    x = np.linspace(a, b, 1000)
    y = [function(i) for i in x]
    
    # Plot the function
    ax.plot(x, y, 'b-', lw=2, label='f(x) = 1/(1+x²)')
    
    # Collect intervals during adaptive integration
    intervals = []
    result = adaptive_integral(function, a, b, eps, intervals=intervals)
    
    # Sort intervals by level for better visualization
    intervals.sort(key=lambda interval: interval[2])
    
    # Create a colormap for different recursion levels
    cmap = plt.cm.get_cmap('viridis', max(interval[2] for interval in intervals) + 1)
    
    # Plot the intervals
    for a_i, b_i, level in intervals:
        mid = (a_i + b_i) / 2
        height = function(mid)
        width = b_i - a_i
        rect = Rectangle((a_i, 0), width, height, facecolor=cmap(level), alpha=0.5)
        ax.add_patch(rect)
    
    # Add a colorbar to show recursion level
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(interval[2] for interval in intervals)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Recursion Level')
    
    ax.set_title(f'Adaptive Integration (ε={eps})')
    ax.legend()
    ax.grid(True)

def performance_comparison(function, a, b, n_values, eps_values):
    """Compare performance and accuracy of different methods"""
    exact = exact_integral()
    
    # Regular rectangular method
    rect_results = []
    rect_errors = []
    rect_times = []
    
    for n in n_values:
        start_time = time.time()
        result = rectangular_integration(function, a, b, n)
        end_time = time.time()
        
        rect_results.append(result)
        rect_errors.append(abs(result - exact))
        rect_times.append(end_time - start_time)
    
    # Adaptive method
    adapt_results = []
    adapt_errors = []
    adapt_times = []
    
    for eps in eps_values:
        start_time = time.time()
        result = adaptive_integral(function, a, b, eps)
        end_time = time.time()
        
        adapt_results.append(result)
        adapt_errors.append(abs(result - exact))
        adapt_times.append(end_time - start_time)
    
    # Adaptive Simpson method
    adapt_simpson_results = []
    adapt_simpson_errors = []
    adapt_simpson_times = []
    
    for eps in eps_values:
        start_time = time.time()
        result = adaptive_simpson(function, a, b, eps)
        end_time = time.time()
        
        adapt_simpson_results.append(result)
        adapt_simpson_errors.append(abs(result - exact))
        adapt_simpson_times.append(end_time - start_time)
    
    return {
        'exact': exact,
        'rectangular': {
            'n_values': n_values,
            'results': rect_results,
            'errors': rect_errors,
            'times': rect_times
        },
        'adaptive': {
            'eps_values': eps_values,
            'results': adapt_results,
            'errors': adapt_errors,
            'times': adapt_times
        },
        'adaptive_simpson': {
            'eps_values': eps_values,
            'results': adapt_simpson_results,
            'errors': adapt_simpson_errors,
            'times': adapt_simpson_times
        }
    }

def visualize_performance(comparison_data):
    """Visualize performance comparison between methods"""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    rect_n = comparison_data['rectangular']['n_values']
    rect_errors = comparison_data['rectangular']['errors']
    rect_times = comparison_data['rectangular']['times']
    
    adapt_eps = comparison_data['adaptive']['eps_values']
    adapt_errors = comparison_data['adaptive']['errors']
    adapt_times = comparison_data['adaptive']['times']
    
    adapt_simpson_eps = comparison_data['adaptive_simpson']['eps_values']
    adapt_simpson_errors = comparison_data['adaptive_simpson']['errors']
    adapt_simpson_times = comparison_data['adaptive_simpson']['times']
    
    # Error comparison
    axs[0, 0].loglog(rect_n, rect_errors, 'rs-', label='Rectangular')
    axs[0, 0].loglog(1/np.array(adapt_eps), adapt_errors, 'go-', label='Adaptive')
    axs[0, 0].loglog(1/np.array(adapt_simpson_eps), adapt_simpson_errors, 'b^-', label='Adaptive Simpson')
    axs[0, 0].set_xlabel('n (Rectangular) or 1/ε (Adaptive)')
    axs[0, 0].set_ylabel('Absolute Error')
    axs[0, 0].set_title('Error Comparison')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Time comparison
    axs[0, 1].loglog(rect_n, rect_times, 'rs-', label='Rectangular')
    axs[0, 1].loglog(1/np.array(adapt_eps), adapt_times, 'go-', label='Adaptive')
    axs[0, 1].loglog(1/np.array(adapt_simpson_eps), adapt_simpson_times, 'b^-', label='Adaptive Simpson')
    axs[0, 1].set_xlabel('n (Rectangular) or 1/ε (Adaptive)')
    axs[0, 1].set_ylabel('Computation Time (s)')
    axs[0, 1].set_title('Time Comparison')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Efficiency comparison (error vs. time)
    axs[1, 0].loglog(rect_times, rect_errors, 'rs-', label='Rectangular')
    axs[1, 0].loglog(adapt_times, adapt_errors, 'go-', label='Adaptive')
    axs[1, 0].loglog(adapt_simpson_times, adapt_simpson_errors, 'b^-', label='Adaptive Simpson')
    axs[1, 0].set_xlabel('Computation Time (s)')
    axs[1, 0].set_ylabel('Absolute Error')
    axs[1, 0].set_title('Efficiency Comparison')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Results table
    axs[1, 1].axis('off')
    table_data = [['Method', 'Parameter', 'Result', 'Error', 'Time (s)']]
    
    # Add rectangular data
    for i, n in enumerate(rect_n):
        table_data.append([
            'Rectangular' if i == 0 else '',
            f'n={n}',
            f'{comparison_data["rectangular"]["results"][i]:.8f}',
            f'{rect_errors[i]:.8e}',
            f'{rect_times[i]:.6f}'
        ])
    
    # Add adaptive data
    for i, eps in enumerate(adapt_eps):
        table_data.append([
            'Adaptive' if i == 0 else '',
            f'ε={eps}',
            f'{comparison_data["adaptive"]["results"][i]:.8f}',
            f'{adapt_errors[i]:.8e}',
            f'{adapt_times[i]:.6f}'
        ])
    
    # Add adaptive Simpson data
    for i, eps in enumerate(adapt_simpson_eps):
        table_data.append([
            'Adaptive Simpson' if i == 0 else '',
            f'ε={eps}',
            f'{comparison_data["adaptive_simpson"]["results"][i]:.8f}',
            f'{adapt_simpson_errors[i]:.8e}',
            f'{adapt_simpson_times[i]:.6f}'
        ])
    
    # Add exact value
    table_data.append(['Exact', '', f'{comparison_data["exact"]:.8f}', '0', ''])
    
    table = axs[1, 1].table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    plt.savefig('integration_performance_comparison.png')
    plt.show()

if __name__ == "__main__":
    exact = exact_integral()
    print(f"Calculating the integral of f(x) = 1/(1+x²) from 0 to 1")
    print(f"Exact value: arctan(1) = π/4 = {exact:.10f}")
    
    # Rectangular method with h=0.1
    h = 0.1
    n = int(1 / h)
    rect_result = rectangular_integration(function, 0, 1, n)
    print(f"\nRectangular method with h={h} (n={n}):")
    print(f"Result: {rect_result:.10f}")
    print(f"Absolute error: {abs(rect_result - exact):.10e}")
    
    # Adaptive integration
    eps_values = [1e-3, 1e-6, 1e-9]
    print("\nAdaptive integration:")
    
    for eps in eps_values:
        adapt_result = adaptive_integral(function, 0, 1, eps)
        print(f"With ε={eps}:")
        print(f"Result: {adapt_result:.10f}")
        print(f"Absolute error: {abs(adapt_result - exact):.10e}")
    
    # Adaptive Simpson integration
    print("\nAdaptive Simpson integration:")
    
    for eps in eps_values:
        adapt_simpson_result = adaptive_simpson(function, 0, 1, eps)
        print(f"With ε={eps}:")
        print(f"Result: {adapt_simpson_result:.10f}")
        print(f"Absolute error: {abs(adapt_simpson_result - exact):.10e}")
    
    # Visualization
    plt.figure(figsize=(12, 10))
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Visualize rectangular integration with h=0.1
    visualize_rectangular(axs[0, 0], function, 0, 1, n)
    
    # Visualize adaptive integration with different epsilon values
    visualize_adaptive(axs[0, 1], function, 0, 1, 1e-3)
    visualize_adaptive(axs[1, 0], function, 0, 1, 1e-6)
    visualize_adaptive(axs[1, 1], function, 0, 1, 1e-9)
    
    plt.tight_layout()
    plt.savefig('integration_visualization.png')
    plt.show()
    
    # Performance comparison
    n_values = [10, 20, 50, 100, 200, 500, 1000]
    eps_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    
    comparison_data = performance_comparison(function, 0, 1, n_values, eps_values)
    visualize_performance(comparison_data)
