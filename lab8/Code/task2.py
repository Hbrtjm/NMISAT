import numpy as np
from numpy.linalg import eigvals
from math import log, ceil

A = np.array([
    [10, -1, 2, -3],
    [1, 10, -1, 2],
    [2, 3, 20, -1],
    [3, 2, 1, 10]
], dtype=float)

b = np.array([0, 5, -10, 15], dtype=float)

D = np.diag(np.diag(A))
L_plus_U = A - D

D_inv = np.linalg.inv(D)
M_J = -D_inv @ L_plus_U
W_J = D_inv @ b

eigenvalues = eigvals(M_J)
rho = max(abs(eigenvalues))

print("Macierz M_J:")
print(np.round(M_J, 4))

print(f"\nPromień spektralny rho(M_J) = {rho:.6f}")

precisions = [1e-3, 1e-4, 1e-5]
p_values = [3, 4, 5]

print("\nSzacowana liczba iteracji Jacobiego dla różnych dokładności:")
for p in p_values:
    t_p = -p * log(10) / log(rho)
    print(f"Dokładność 10^-{p}: {ceil(t_p)} iteracji")

