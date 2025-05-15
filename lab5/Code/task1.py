import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def function(x):
    return 1/(1+x)

def rectangular_integration(function, a, b, n):
    dx = (b - a)/n
    # Using the midpoint rule for better accuracy
    return sum([dx * function(a + dx * (i + 0.5)) for i in range(n)])

def trapezoid_integration(function, a, b, n):
    dx = (b - a)/n
    return dx * (function(a)/2 + sum([function(a + dx * i) for i in range(1, n)]) + function(b)/2)

def simpson_method_simple(function, a, b):
    h = (b - a)/2
    return (h/3) * (function(a) + 4*function(a + h) + function(b))

def simpson_method_composite(function, a, b, n):
    if n % 2 != 0:
        n += 1  # Ensure n is even for Simpson's rule
    
    dx = (b - a)/n
    result = function(a) + function(b)
    
    for i in range(1, n):
        x = a + i * dx
        if i % 2 == 0:
            result += 2 * function(x)
        else:
            result += 4 * function(x)
            
    return (dx/3) * result

def simpson_three_eighths_rule(function, a, b):
    h = (b - a)/3
    return (3*h/8) * (function(a) + 3*function(a + h) + 3*function(a + 2*h) + function(b))

def boole_rule(function, a, b):
    h = (b - a)/4
    return (2*h/45) * (7*function(a) + 32*function(a + h) + 12*function(a + 2*h) + 32*function(a + 3*h) + 7*function(b))

def exact_integral():
    # The exact value of ∫(1/(1+x))dx from 0 to 1 is ln(2)
    import math
    return math.log(2)

def plot_rectangular_method(function, a, b, n):
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(a, b, 1000)
    y = [function(i) for i in x]
    
    dx = (b - a)/n
    rectangles_x = []
    rectangles_y = []
    
    for i in range(n):
        mid_x = a + dx * (i + 0.5)
        height = function(mid_x)
        
        # Add the corners of the rectangle
        rectangles_x.extend([a + i*dx, a + i*dx, a + (i+1)*dx, a + (i+1)*dx])
        rectangles_y.extend([0, height, height, 0])
    
    ax.plot(x, y, 'b-', lw=2, label='f(x) = 1/(1+x)')
    ax.fill(rectangles_x, rectangles_y, 'r', alpha=0.3)
    ax.set_title(f'Rectangle Method (n={n})')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('rectangular.png')
    plt.close()

def plot_trapezoid_method(function, a, b, n):
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(a, b, 1000)
    y = [function(i) for i in x]
    
    dx = (b - a)/n
    trapezoids_x = []
    trapezoids_y = []
    
    for i in range(n):
        x1 = a + i*dx
        x2 = a + (i+1)*dx
        y1 = function(x1)
        y2 = function(x2)
        
        # Add the corners of the trapezoid
        trapezoids_x.extend([x1, x1, x2, x2])
        trapezoids_y.extend([0, y1, y2, 0])
    
    ax.plot(x, y, 'b-', lw=2, label='f(x) = 1/(1+x)')
    ax.fill(trapezoids_x, trapezoids_y, 'g', alpha=0.3)
    ax.set_title(f'Trapezoid Method (n={n})')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('trapezoid.png')
    plt.close()

def plot_simpson_method(function, a, b, n):
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(a, b, 1000)
    y = [function(i) for i in x]
    
    # Plot the function
    ax.plot(x, y, 'b-', lw=2, label='f(x) = 1/(1+x)')
    
    # Plot the Simpson parabolas
    dx = (b - a)/n
    for i in range(0, n, 2):
        if i+2 <= n:
            x0 = a + i*dx
            x1 = a + (i+1)*dx
            x2 = a + (i+2)*dx
            
            y0 = function(x0)
            y1 = function(x1)
            y2 = function(x2)
            
            # Create a finer grid for the parabola
            xs = np.linspace(x0, x2, 100)
            
            # Lagrange interpolation formula for parabola through 3 points
            ys = []
            for xx in xs:
                L0 = ((xx-x1)*(xx-x2))/((x0-x1)*(x0-x2))
                L1 = ((xx-x0)*(xx-x2))/((x1-x0)*(x1-x2))
                L2 = ((xx-x0)*(xx-x1))/((x2-x0)*(x2-x1))
                yy = y0*L0 + y1*L1 + y2*L2
                ys.append(yy)
            
            # Plot the parabola
            ax.plot(xs, ys, 'r-', lw=1.5, alpha=0.5)
            
            # Fill the area under the parabola
            verts = [(x, 0) for x in xs] + [(xs[-1], ys[-1])]
            for i in range(len(xs)-1, -1, -1):
                verts.append((xs[i], ys[i]))
            poly = Polygon(verts, facecolor='purple', alpha=0.3)
            ax.add_patch(poly)
    
    ax.set_title(f'Simpson Method (n={n})')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('simpson.png')
    plt.close()

def plot_error_convergence(function, a, b, exact):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Convergence plot for all methods
    ns = [2, 4, 8, 16, 32, 64]
    
    rect_errors = []
    trap_errors = []
    simp_errors = []
    
    for n in ns:
        rect_errors.append(abs(rectangular_integration(function, a, b, n) - exact))
        trap_errors.append(abs(trapezoid_integration(function, a, b, n) - exact))
        simp_errors.append(abs(simpson_method_composite(function, a, b, n) - exact))
    
    ax.loglog(ns, rect_errors, 'o-', label='Prostokątna')
    ax.loglog(ns, trap_errors, 's-', label='Trapezoidalna')
    ax.loglog(ns, simp_errors, '^-', label='Simpsona')
    ax.set_xlabel('Liczba podziałów (n)')
    ax.set_ylabel('Błąd bezwględny')
    ax.set_title('Zbieżność błędu')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('error_convergence.png')
    plt.close()

def plot_error_comparison(methods, ns, errors):
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', 'D', '^', '*']
    
    for i, method in enumerate(methods):
        plt.plot(ns, errors[i], label=method, marker=markers[i])
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Liczba węzłów (n)')
    plt.ylabel('Błąd bezwzględny')
    plt.title('Porównanie błędu dla różnych metod całkowania')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('error_comparison.png')
    plt.close()

def plot_interactive_visualization(function, a, b, n):
    plt.figure(figsize=(14, 6))
    
    # Plot the actual function
    x = np.linspace(a, b, 1000)
    y = [function(i) for i in x]
    plt.plot(x, y, 'b-', lw=2, label='f(x) = 1/(1+x)')
    
    dx = (b - a)/n
    
    # Rectangle method
    rect_x = []
    rect_y = []
    for i in range(n):
        mid_x = a + dx * (i + 0.5)
        height = function(mid_x)
        rect_x.extend([a + i*dx, a + i*dx, a + (i+1)*dx, a + (i+1)*dx])
        rect_y.extend([0, height, height, 0])
    
    plt.fill(rect_x, rect_y, 'r', alpha=0.2, label=f'Prostokąty (n={n})')
    
    # Trapezoid method
    trap_x = []
    trap_y = []
    for i in range(n):
        x1 = a + i*dx
        x2 = a + (i+1)*dx
        y1 = function(x1)
        y2 = function(x2)
        trap_x.extend([x1, x1, x2, x2])
        trap_y.extend([0, y1, y2, 0])
    
    plt.fill(trap_x, trap_y, 'g', alpha=0.2, label=f'Trapezoidy (n={n})')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Visualization of Integration Methods for f(x) = 1/(1+x), n={n}')
    plt.legend()
    plt.grid(True)
    plt.savefig('interactive_visualization.png')
    plt.close()

if __name__ == "__main__":
    exact = exact_integral()
    print("Numerical integration of ∫(1/(1+x))dx from 0 to 1")
    print(f"Exact value: ln(2) = {exact:.10f}")
    print("\n--- Simple methods ---")
    
    rect_result = rectangular_integration(function, 0, 1, 1)
    trap_result = trapezoid_integration(function, 0, 1, 1)
    simp_result = simpson_method_simple(function, 0, 1)
    three_eighths = simpson_three_eighths_rule(function, 0, 1)
    boole = boole_rule(function, 0, 1)
    
    print(f"Rectangle method (n=1): {rect_result:.10f}, Error: {abs(rect_result-exact):.10f}")
    print(f"Trapezoid method (n=1): {trap_result:.10f}, Error: {abs(trap_result-exact):.10f}")
    print(f"Simpson's 1/3 rule: {simp_result:.10f}, Error: {abs(simp_result-exact):.10f}")
    print(f"Simpson's 3/8 rule: {three_eighths:.10f}, Error: {abs(three_eighths-exact):.10f}")
    print(f"Boole's rule (n=5): {boole:.10f}, Error: {abs(boole-exact):.10f}")
    
    print("\n--- Composite methods ---")
    
    # Using composite methods with n=3 and n=5
    rect_result_3 = rectangular_integration(function, 0, 1, 3)
    trap_result_3 = trapezoid_integration(function, 0, 1, 3)
    simp_result_3 = simpson_method_composite(function, 0, 1, 3)
    
    rect_result_5 = rectangular_integration(function, 0, 1, 5)
    trap_result_5 = trapezoid_integration(function, 0, 1, 5)
    simp_result_5 = simpson_method_composite(function, 0, 1, 5)
    
    print(f"Composite Rectangle method (n=3): {rect_result_3:.10f}, Error: {abs(rect_result_3-exact):.10f}")
    print(f"Composite Trapezoid method (n=3): {trap_result_3:.10f}, Error: {abs(trap_result_3-exact):.10f}")
    print(f"Composite Simpson's method (n=3): {simp_result_3:.10f}, Error: {abs(simp_result_3-exact):.10f}")
    
    print(f"Composite Rectangle method (n=5): {rect_result_5:.10f}, Error: {abs(rect_result_5-exact):.10f}")
    print(f"Composite Trapezoid method (n=5): {trap_result_5:.10f}, Error: {abs(trap_result_5-exact):.10f}")
    print(f"Composite Simpson's method (n=5): {simp_result_5:.10f}, Error: {abs(simp_result_5-exact):.10f}")
    
    # Create visualizations - each as a separate figure
    plot_rectangular_method(function, 0, 1, 5)
    plot_trapezoid_method(function, 0, 1, 5)
    plot_simpson_method(function, 0, 1, 4)
    plot_error_convergence(function, 0, 1, exact)
    
    # Comprehensive error comparison across multiple values of n
    methods = ['Prostokątna', 'Trapezoidowa', 'Simpsona']
    ns_ext = [1, 2, 4, 8, 16, 32, 64, 128]
    
    rect_errors_ext = []
    trap_errors_ext = []
    simp_errors_ext = []
    
    for n in ns_ext:
        rect_errors_ext.append(abs(rectangular_integration(function, 0, 1, n) - exact))
        trap_errors_ext.append(abs(trapezoid_integration(function, 0, 1, n) - exact))
        simp_errors_ext.append(abs(simpson_method_composite(function, 0, 1, n if n > 1 else 2) - exact))
    
    plot_error_comparison(methods, ns_ext, [rect_errors_ext, trap_errors_ext, simp_errors_ext])
    
    # Create an interactive visualization for different n values
    plot_interactive_visualization(function, 0, 1, 3)
