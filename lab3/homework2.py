# homework2.py
from scipy.special import legendre
import numpy as np
import scipy.interpolate as spi
import scipy.integrate as spi_int

# Zadanie 2a: Sprawdzenie ortogonalności pierwszych 7 wielomianów Legendre’a
n = 7
def check_orthogonality(n):
    for i in range(n):
        for j in range(i, n):
            # TODOs
            integral = spi_int.quad(lambda x: legendre(i)(x) * legendre(j)(x), -1, 1)[0]
            print(f"Integral of P_{i} * P_{j}: {integral:.6f}")
check_orthogonality(n)

# Zadanie 2b: Sprawdzenie wzoru rekurencyjnego
for k in range(2, n):
    Pk = legendre(k)
    Pk_1 = legendre(k - 1)
    Pk_2 = legendre(k - 2)
    lhs = k * Pk
    # TODO
    rhs = (2 * k - 1) * np.poly1d([1, 0]) * Pk_1 - (k - 1) * Pk_2
    print(f"Rekurencja dla k={k}: {np.allclose(lhs.coeffs, rhs.coeffs)}")

# Zadanie 2c: Wyrażenie t^k jako liniowa kombinacja P0-P6
def express_monomials(n):
    basis = [legendre(i) for i in range(n)]
    for k in range(n):
        # TODO
        poly = np.poly1d([0] * (k + 1) + [1])
        coeffs = [spi_int.quad(lambda x: poly(x) * basis[i](x), -1, 1)[0] for i in range(n)]
        print(f"t^{k} = {coeffs}")
express_monomials(n)
