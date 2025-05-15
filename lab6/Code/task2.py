import numpy as np

def first_function(x, y):
    return x ** 2 + x * y ** 3 - 9 
        
def first_function_der_x(x, y):
    return 2 * x + y ** 3 

def first_function_der_y(x, y):
    return 3 * x * y ** 2

def second_function(x, y):
    return 3 * x ** 2 * y - y ** 3 - 4

def second_function_der_x(x, y):
    return 6 * x * y

def second_function_der_y(x, y):
    return 3 * x ** 2 - 3 * y ** 2

def make_jacobian(x, y):
    return np.array([
        [first_function_der_x(x, y), first_function_der_y(x, y)],
        [second_function_der_x(x, y), second_function_der_y(x, y)]
    ])

def make_vector(x, y):
    return np.array([first_function(x, y), second_function(x, y)])

def newton_method(x0, y0, tol=1e-16, max_iter=20):
    x, y = x0, y0
    print("\\begin{enumerate}")
    print(f"\\item Punkt wyjściowy: $x_0 = {x0:.6f}$, $y_0 = {y0:.6f}$")
    for iteration in range(max_iter):
        F = make_vector(x, y)
        f_norm = np.linalg.norm(F)
        J = make_jacobian(x, y)
        print(f"\\item Iteracja {iteration + 1}:")
        print(f"  \\begin{{itemize}}")
        print(f"    \\item Punkt: $(x_{iteration}, y_{iteration}) = ({x:.6f}, {y:.6f})$")
        print(f"    \\item Wartości funkcji: $F(x,y) = [{F[0]:.6f}, {F[1]:.6f}]^T$")
        print(f"    \\item $||F(x,y)|| = {f_norm:.6e}$")
        print(f"    \\item Macierz Jakobiego: $J = \\begin{{bmatrix}} {J[0,0]:.6f} & {J[0,1]:.6f} \\\\ {J[1,0]:.6f} & {J[1,1]:.6f} \\end{{bmatrix}}$")
        if f_norm < tol:
            print(f"    \\item Warunek zbieżności: $||F(x,y)|| < {tol}$")
            print(f"  \\end{{itemize}}")
            print("\\end{enumerate}")
            return x, y, iteration, True
        try:
            delta = np.linalg.solve(J, -F)
            print(f"    \\item $\\Delta = [{delta[0]:.6f}, {delta[1]:.6f}]^T$")
            x += delta[0]
            y += delta[1]
            print(f"    \\item Następny punkt: $(x_{iteration+1}, y_{iteration+1}) = ({x:.6f}, {y:.6f})$")
            print(f"  \\end{{itemize}}")
        except np.linalg.LinAlgError:
            print(f"    \\item Macierz z zerowym wyznacznikiem - rozbieżność")
            print(f"  \\end{{itemize}}")
            print("\\end{enumerate}")
            return x, y, iteration, False
    return x, y, max_iter, False

if __name__ == "__main__":
    x0, y0 = 1.5, 1.0
    x_sol, y_sol, iterations, converged = newton_method(x0, y0)
    if converged:
        print(f"\n% Summary")
        print(f"% Solution found after {iterations + 1} iterations:")
        print(f"% x = {x_sol:.10f}, y = {y_sol:.10f}")
        print(f"% Function values at solution:")
        print(f"% f1(x,y) = {first_function(x_sol, y_sol):.10e}")
        print(f"% f2(x,y) = {second_function(x_sol, y_sol):.10e}")
    else:
        print(f"\n% Failed to converge after {iterations + 1} iterations.")
        print(f"% Last values: x = {x_sol:.10f}, y = {y_sol:.10f}")
