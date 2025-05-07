import numpy as np

n = 8
pi = np.pi

# Obliczenie węzłów x_k dla kwadratury Czebyszewa I rodzaju
k = np.arange(1, n+1)
xk = np.cos((2*k - 1)*pi/(2*n))

# Funkcja g(x) = 1/(1+x^2)
g = 1 / (1 + xk**2)

weights_factor = np.sqrt(1 - xk**2)

# Wagi cząstkowe: w_k = pi/n dla wszystkich k
w = pi / n

# Obliczenie sumy kwadratury
I_approx = w * np.sum( g * weights_factor )
print(f"Przybliżona wartość całki I = {I_approx}")
print(f"Błąd całkowania względem analitycznej wartości: {abs(pi/2 - I_approx)}")
print(f"Błąd całkowania względem analitycznej wartości: {abs(pi/2 - I_approx)/(pi/2)}")
