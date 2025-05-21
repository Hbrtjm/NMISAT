from random import choice
import math
import numpy as np

class Rational:
    def __init__(self, num, den):
        if den == 0:
            raise ZeroDivisionError("Denominator cannot be zero")
        if den < 0:
            num, den = -num, -den
        g = math.gcd(abs(num), abs(den))
        self.num = num // g
        self.den = den // g

    def __add__(self, other):
        if not isinstance(other, Rational):
            return NotImplemented
        g = math.gcd(self.den, other.den)
        b1 = self.den // g
        d1 = other.den // g
        new_num = self.num * d1 + other.num * b1
        new_den = b1 * other.den
        return Rational(new_num, new_den)

    def __sub__(self, other):
        if not isinstance(other, Rational):
            return NotImplemented
        g = math.gcd(self.den, other.den)
        b1 = self.den // g
        d1 = other.den // g
        new_num = self.num * d1 - other.num * b1
        new_den = b1 * other.den
        return Rational(new_num, new_den)

    def __mul__(self, other):
        if not isinstance(other, Rational):
            return NotImplemented
        g1 = math.gcd(abs(self.num), abs(other.den))
        g2 = math.gcd(abs(other.num), abs(self.den))
        n1 = (self.num // g1) * (other.num // g2)
        d1 = (self.den // g2) * (other.den // g1)
        return Rational(n1, d1)

    def __truediv__(self, other):
        if not isinstance(other, Rational):
            return NotImplemented
        return self.__mul__(Rational(other.den, other.num))

    def __repr__(self):
        return f"{self.num}/{self.den}" if self.den != 1 else f"{self.num}"


n = 10
A = np.empty((n, n), dtype=object)

for i in range(n):
    for j in range(n):
        A[i][j] = Rational(0, 1)

for i in range(n):
    if i > 0:
        A[i][i - 1] = Rational(1, i + 1)
    if i < n - 1:
        A[i][i + 1] = Rational(1, i + 2)
    if 0 < i < n - 1:
        A[i][i] = Rational(2, 1)
    elif i == 0 or i == n - 1:
        A[i][i] = Rational(1, 1)


def jacobi_with_divergence_handling(A, b, tol=1e-6, max_iter=1000, omega=0.8):
    n = len(b)
    x = [Rational(0, 1) for _ in range(n)]
    D_inv = []
    try:
        for i in range(n):
            if A[i][i].num == 0:
                raise ValueError(f"Zero on diagonal at position {i}")
            D_inv.append(Rational(A[i][i].den, A[i][i].num))

        def residual(x_vec):
            r = []
            for i in range(n):
                sum_term = Rational(0, 1)
                for j in range(n):
                    sum_term = sum_term + A[i][j] * x_vec[j]
                r.append(b[i] - sum_term)
            return max(abs(r_i.num / r_i.den) for r_i in r)

        r0 = residual(x)
        prev_res = r0

        for k in range(1, max_iter+1):
            x_new = []
            for i in range(n):
                sigma = Rational(0, 1)
                for j in range(n):
                    if j != i:
                        sigma = sigma + A[i][j] * x[j]
                y = (b[i] - sigma) * D_inv[i]
                x_new.append(y)
            
            res = residual(x_new)
            if res > prev_res:
                print(f"Metoda Jacobiego rozbiega się w iteracji {k}. Przełączam na tłumioną metodę Jacobiego (ω={omega}).")
                for m in range(k, max_iter+1):
                    x_damped = []
                    for i in range(n):
                        sigma = Rational(0, 1)
                        for j in range(n):
                            if j != i:
                                sigma = sigma + A[i][j] * x[j]
                        y = (b[i] - sigma) * D_inv[i]
                        omega_rational = Rational(int(omega * 1000), 1000) 
                        one_minus_omega = Rational(1000 - int(omega * 1000), 1000)
                        damped_val = (omega_rational * y) + (one_minus_omega * x[i])
                        x_damped.append(damped_val)
                    
                    x = x_damped
                    res_d = residual(x)
                    if res_d < tol:
                        print(f"Tłumiona metoda Jacobiego zbiega się po {m} iteracjach.")
                        return x
                    prev_res = res_d
                print("Tłumiona metoda Jacobiego nie zbiega się.")
                return x

            if res < tol:
                print(f"Metoda Jacobiego zbiega się po {k} iteracjach.")
                return x_new

            x = x_new
            prev_res = res

        print("Metoda Jacobiego osiągnęła maksymalną liczbę iteracji bez zbieżności lub rozbieżności.")
        return x
    except:
        print("Metoda Jacobiego nie zbiega się.")

def chebyshev(A, b, tol=1e-6, max_iter=1000):
    A_float = np.array([[a.num / a.den for a in row] for row in A])
    eigs = np.linalg.eigvals(A_float)
    lambda_min = min(abs(eigs))
    lambda_max = max(abs(eigs))

    x = np.zeros(n)
    r = b.astype(float) - A_float @ x
    d = r.copy()

    for k in range(1, max_iter + 1):
        alpha = 2.0 / (lambda_max + lambda_min)
        x_new = x + alpha * d
        r = b.astype(float) - A_float @ x_new

        if np.linalg.norm(r, np.inf) < tol:
            print(f"Metoda Czebyszewa zbiega się po {k} iteracjach.")
            return x_new

        beta = ((lambda_max - lambda_min) / (lambda_max + lambda_min)) ** 2
        d = r + beta * d
        x = x_new

    print("Metoda Czebyszewa nie zbiega się.")
    return x


for row in A:
    print("  ".join(str(x) for x in row))

x_true = np.array([choice([0, -1]) for _ in range(n)])

b = []
for i in range(n):
    sum_val = Rational(0, 1)
    for j in range(n):
        sum_val = sum_val + A[i][j] * Rational(x_true[j], 1)
    b.append(sum_val)

print("b =")
for bi in b:
    print(bi)

x_jacobi = jacobi_with_divergence_handling(A, b)
print("Rozwiązanie metodą Jacobiego:")
if x_jacobi is not None:
    for xi in x_jacobi:
        print(xi)

b_float = np.array([bi.num / bi.den for bi in b])
x_cheb = chebyshev(A, b_float)
print("Rozwiązanie metodą Czebyszewa:")
print(x_cheb)

print("Oryginalny x:")
print(x_true)

diff_cheb = x_cheb - x_true
diff_jacobi = (np.array([xi.num / xi.den for xi in x_jacobi]) if x_jacobi else 0) - x_true

print("Różnica między rozwiązaniem rzeczywistym a przybliżonym:")
print(f"Dla metody Czebyszewa: {diff_cheb}\nDla metody Jacobiego: {diff_jacobi}")
print(f"Bezwzględna różnica:\nCzebyszew: {abs(diff_cheb)}\nJacobi: {abs(diff_jacobi)}")
