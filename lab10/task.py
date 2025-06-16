import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from time import time
from random import random

def function1(x):
    return x ** 2 + x + 1

def function2(x):
    return sqrt(1 - x**2)

def function3(x):
    return 1/sqrt(x)

def hit_and_miss(func, n=1000, a=0, b=1, **kwargs):
    """
    Monte Carlo integration using hit-and-miss method.
    Note: This method works poorly for function3 (1/sqrt(x)) due to unbounded behavior at x=0
    """
    hit = 0
    
    # Find the maximum value of the function in the interval for bounding rectangle
    if func == function1:
        max_val = max(func(a), func(b))  # monotonic function
    elif func == function2:
        max_val = func(0)  # maximum at x=0
    elif func == function3:
        # This function goes to infinity as x approaches 0
        # We'll use a large value, but this makes the method inefficient
        max_val = func(0.001)  # Very large value near 0
    else:
        # General case: sample points to find approximate maximum
        x_vals = [a + (b-a)*i/100 for i in range(101)]
        max_val = max(func(x) for x in x_vals if x > 0)
    
    for _ in range(int(n)):
        rand_x = a + (b - a) * random()
        rand_y = max_val * random()
        
        if rand_x > 0 and rand_y <= func(rand_x):
            hit += 1
    
    # Area of bounding rectangle times ratio of hits
    return (b - a) * max_val * hit / n

def monte_carlo_mean(func, n=1000, a=0, b=1, **kwargs):
    """
    Monte Carlo integration using mean value method.
    More efficient and works better for all functions.
    """
    total = 0
    for _ in range(int(n)):
        rand_x = a + (b - a) * random()
        # Avoid x=0 for function3 to prevent division by zero
        if func == function3 and rand_x < 1e-10:
            rand_x = 1e-10
        total += func(rand_x)
    
    return (b - a) * total / n

def rectangle_rule(func, eps=1e-3, a=0, b=1, **kwargs):
    """
    Numerical integration using rectangle rule with specified accuracy.
    """
    n = 1
    prev_integral = 0
    
    while True:
        h = (b - a) / n
        integral = 0
        
        for i in range(n):
            x = a + (i + 0.5) * h  # Midpoint rule
            # Avoid x=0 for function3
            if func == function3 and x < 1e-10:
                x = 1e-10
            integral += func(x)
        
        integral *= h
        
        if n > 1 and abs(integral - prev_integral) < eps:
            return integral
        
        prev_integral = integral
        n *= 2
        
        # Prevent infinite loop
        if n > 1e8:
            break
    
    return integral

def compare_methods(func, func_name, n_trials_list, accuracies):
    """
    Compare Monte Carlo methods and rectangle rule for different parameters.
    """
    print(f"\n=== Analysis for {func_name} ===")
    
    # Analytical solutions for comparison
    if func == function1:
        analytical = 11/6  # ∫(x² + x + 1)dx from 0 to 1 = [x³/3 + x²/2 + x]₀¹
    elif func == function2:
        analytical = np.pi/4  # ∫√(1-x²)dx from 0 to 1 = π/4 (quarter circle)
    elif func == function3:
        analytical = 2  # ∫1/√x dx from 0 to 1 = [2√x]₀¹ = 2
    
    print(f"Analytical solution: {analytical:.6f}")
    
    # Test Monte Carlo convergence
    print("\nMonte Carlo convergence (mean value method):")
    print("N\t\tResult\t\tError\t\tTime(s)")
    mc_times = []
    mc_errors = []
    
    for n in n_trials_list:
        start = time()
        result = monte_carlo_mean(func, n)
        elapsed = time() - start
        error = abs(result - analytical)
        
        print(f"{n}\t\t{result:.6f}\t{error:.6f}\t{elapsed:.6f}")
        mc_times.append(elapsed)
        mc_errors.append(error)
    
    # Test rectangle rule for different accuracies
    print(f"\nRectangle rule comparison:")
    print("Accuracy\tResult\t\tError\t\tTime(s)")
    rect_times = []
    rect_errors = []
    
    for eps in accuracies:
        start = time()
        result = rectangle_rule(func, eps)
        elapsed = time() - start
        error = abs(result - analytical)
        
        print(f"{eps}\t\t{result:.6f}\t{error:.6f}\t{elapsed:.6f}")
        rect_times.append(elapsed)
        rect_errors.append(error)
    
    # Test hit-and-miss method
    print(f"\nHit-and-miss method:")
    if func == function3:
        print("Warning: Hit-and-miss method is inefficient for 1/√x due to unbounded behavior!")
    
    start = time()
    result_hm = hit_and_miss(func, 10000)
    elapsed_hm = time() - start
    error_hm = abs(result_hm - analytical)
    print(f"Result: {result_hm:.6f}, Error: {error_hm:.6f}, Time: {elapsed_hm:.6f}s")
    
    return mc_times, mc_errors, rect_times, rect_errors

def plot_comparison(n_trials_list, mc_errors_list, mc_times_list, rect_times_list, rect_errors_list, func_names, accuracies):
    """
    Plot error convergence and time comparison for Monte Carlo vs Rectangle rule.
    """
    plt.figure(figsize=(18, 12))
    
    # Combined error convergence plot: Monte Carlo vs Rectangle Rule
    plt.subplot(2, 3, 1)
    colors = ['red', 'blue', 'green']
    markers_mc = ['o', 's', '^']
    markers_rect = ['x', '+', 'D']
    
    for i, (mc_errors, func_name) in enumerate(zip(mc_errors_list, func_names)):
        # Monte Carlo errors vs trials
        plt.loglog(n_trials_list, mc_errors, marker=markers_mc[i], linestyle='-', 
                  color=colors[i], label=f'MC: {func_name}', markersize=6)
        
        # Rectangle rule errors vs accuracy target (convert to equivalent "trials")
        # For visualization, we'll plot rectangle errors at positions corresponding to similar computational effort
        rect_equiv_trials = []
        for j, error in enumerate(rect_errors_list[i]):
            # Estimate equivalent trials based on computational time ratio
            if rect_times_list[i][j] > 0 and mc_times_list[i][0] > 0:
                time_ratio = rect_times_list[i][j] / mc_times_list[i][0]
                equiv_trials = n_trials_list[0] * time_ratio
                rect_equiv_trials.append(max(equiv_trials, n_trials_list[0]))
            else:
                rect_equiv_trials.append(n_trials_list[j] if j < len(n_trials_list) else n_trials_list[-1])
        
        plt.loglog(rect_equiv_trials, rect_errors_list[i], marker=markers_rect[i], 
                  linestyle='--', color=colors[i], label=f'Rect: {func_name}', markersize=6)
    
    plt.xlabel('Narzut obliczeniowy (ilość iteracji)')
    plt.ylabel('Błąd')
    plt.title('Zbieżność błędu: Monte Carlo i metody prostokątów')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Error vs Time comparison (most direct comparison)
    plt.subplot(2, 3, 2)
    for i, func_name in enumerate(zip(func_names)):
        # Monte Carlo: error vs time
        plt.loglog(mc_times_list[i], mc_errors_list[i], marker='o', linestyle='-', 
                  color=colors[i], label=f'MC: {func_name[0]}', markersize=6)
        
        # Rectangle rule: error vs time  
        plt.loglog(rect_times_list[i], rect_errors_list[i], marker='s', linestyle='--', 
                  color=colors[i], label=f'Rect: {func_name[0]}', markersize=6)
    
    plt.xlabel('Czas obliczeń [s]')
    plt.ylabel('Błąd')
    plt.title('Błąd po czasie obliczeń')
    plt.legend()
    plt.grid(True)
    
    # Theoretical 1/√n convergence for Monte Carlo
    plt.subplot(2, 3, 3)
    theoretical = [1/sqrt(n) for n in n_trials_list]
    # Normalize to match first data point
    if mc_errors_list[0][0] > 0:
        theoretical = [t * mc_errors_list[0][0] * sqrt(n_trials_list[0]) for t in theoretical]
    
    plt.loglog(n_trials_list, theoretical, 'k--', linewidth=2, label='Teoretyczne \frac{1}{\sqrt{n}}')
    for i, func_name in enumerate(func_names):
        plt.loglog(n_trials_list, mc_errors_list[i], marker=markers_mc[i], 
                  color=colors[i], label=f'MC: {func_name}')
    plt.xlabel('Liczba prób')
    plt.ylabel('Błąd')
    plt.title('Zbieżność metody Monte Carlo')
    plt.legend()
    plt.grid(True)
    

    
    # Combined execution time comparison
    plt.subplot(2, 3, 4)
    for i, func_name in enumerate(func_names):
        plt.loglog(n_trials_list, mc_times_list[i], marker='o', linestyle='-', 
                  color=colors[i], label=f'MC: {func_name}', markersize=6)
    
    # Add rectangle times (plotted against equivalent computational steps)
    for i, func_name in enumerate(func_names):
        # Estimate computational steps for rectangle method
        steps_estimate = [10**(3 + j*0.5) for j in range(len(rect_times_list[i]))]
        plt.loglog(steps_estimate, rect_times_list[i], marker='s', linestyle='--', 
                  color=colors[i], label=f'Rect: {func_name}', markersize=6)
    
    plt.xlabel('Liczba prób')
    plt.ylabel('Czas [s]')
    plt.title('Porównanie czasu wykonania')
    plt.legend()
    plt.grid(True)
    
    # Direct time comparison for similar accuracy
    plt.subplot(2, 3, 5)
    methods = ['Monte Carlo', 'Rectangle Rule']
    x_pos = np.arange(len(func_names))
    width = 0.35
    
    # Get times for similar accuracy (around 1e-4)
    target_accuracy = 1e-4
    mc_times_comp = []
    rect_times_comp = []
    
    for i in range(len(func_names)):
        # Find MC time for closest accuracy to target
        closest_mc_idx = min(range(len(mc_errors_list[i])), 
                           key=lambda x: abs(mc_errors_list[i][x] - target_accuracy))
        mc_times_comp.append(mc_times_list[i][closest_mc_idx])
        
        # Find Rectangle time for closest accuracy to target
        closest_rect_idx = min(range(len(rect_errors_list[i])), 
                             key=lambda x: abs(rect_errors_list[i][x] - target_accuracy))
        rect_times_comp.append(rect_times_list[i][closest_rect_idx])
    
    plt.bar(x_pos - width/2, mc_times_comp, width, label='Metoda Monte Carlo', alpha=0.7)
    plt.bar(x_pos + width/2, rect_times_comp, width, label='Metoda prostokątów', alpha=0.7)
    plt.xlabel('Funkcje')
    plt.ylabel('Czas [s]')
    plt.title(f'Porównanie czasowe dla ~{target_accuracy} Accuracy')
    plt.xticks(x_pos, func_names)
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # # Efficiency ratio plot
    # plt.subplot(2, 3, 6)
    # efficiency_ratios = [mc_times_comp[i] / rect_times_comp[i] for i in range(len(func_names))]
    # bars = plt.bar(func_names, efficiency_ratios, color=['red', 'blue', 'green'], alpha=0.7)
    # plt.ylabel(' (MC/Rectangle)')
    # plt.title('Efficiency Ratio: Monte Carlo / Rectangle Rule')
    # plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal performance')
    # plt.xticks(rotation=45)
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    # for bar, ratio in zip(bars, efficiency_ratios):
    #     plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
    #             f'{ratio:.1f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run all comparisons and analyses.
    """
    functions = [function1, function2, function3]
    func_names = ['x² + x + 1', '√(1-x²)', '1/√x']
    n_trials_list = [100, 500, 1000, 5000, 10000, 50000]
    accuracies = [1e-3, 1e-4, 1e-5, 1e-6]
    
    all_mc_errors = []
    all_mc_times = []
    all_rect_times = []
    all_rect_errors = []
    
    # Compare methods for each function
    for func, func_name in zip(functions, func_names):
        mc_times, mc_errors, rect_times, rect_errors = compare_methods(
            func, func_name, n_trials_list, accuracies
        )
        all_mc_errors.append(mc_errors)
        all_mc_times.append(mc_times)
        all_rect_times.append(rect_times)
        all_rect_errors.append(rect_errors)
    
    # Plot comprehensive comparison including time analysis
    plot_comparison(n_trials_list, all_mc_errors, all_mc_times, 
                   all_rect_times, all_rect_errors, func_names, accuracies)
    
    # Summary of findings
    print("\n=== SUMMARY ===")
    print("1. Hit-and-miss method:")
    print("   - Works well for bounded functions (function1, function2)")
    print("   - Poor performance for 1/√x due to unbounded behavior near x=0")
    print("   - Generally less efficient than mean value method")
    
    print("\n2. Monte Carlo convergence:")
    print("   - Error decreases as O(1/√n) - slow convergence")
    print("   - Requires many samples for high accuracy")
    print("   - Performance independent of dimension (advantage for high-D problems)")
    
    print("\n3. Rectangle rule vs Monte Carlo:")
    print("   - Rectangle rule: faster convergence for 1D problems")
    print("   - Monte Carlo: better for high-dimensional integrals")
    print("   - For 1D problems, rectangle rule is generally more efficient")

if __name__ == "__main__":
    main()