# homework1.py
import numpy as np

# Zadanie 1: Kompromis między granicą błędu a zachowaniem wielomianu interpolacyjnego dla funkcji Rungego
def Rungegs(t):
    return 1 / (1 + 25 * t ** 2)

N = 100
start = -1
end = 1
x_values = np.linspace(start, end, N)
y_values = Rungegs(x_values)

# Interpolacja dla równoodległych węzłów
num_nodes = [5, 10, 15, 20]
interpolations = {}
for n in num_nodes:
    nodes = np.linspace(start, end, n)
    values = Rungegs(nodes)
    poly = np.polyfit(nodes, values, n - 1)
    interpolations[n] = np.poly1d(poly)

# Wyznaczenie błędu
errors = {}
for n in num_nodes:
    errors[n] = np.max(np.abs(interpolations[n](x_values) - y_values))
