import numpy as np
import time
import matplotlib.pyplot as plt
from random import randrange
import scipy.linalg

def gen_A(n, a=0, b=10):
    return [[randrange(a, b) for _ in range(n)] for _ in range(n)]

def gen_b(n, a=0, b=10):
    return [randrange(a, b) for _ in range(n)]

def solve_inverse_np(A, b):
    A_np = np.array(A)
    b_np = np.array(b)
    
    start = time.time()
    A_inv = np.linalg.inv(A_np)
    x = A_inv @ b_np
    end = time.time()
    
    I1 = A_np @ A_inv
    I2 = A_inv @ A_np
    
    identity = np.eye(len(A))
    is_I1 = np.allclose(I1, identity)
    is_I2 = np.allclose(I2, identity)
    
    print(f"A*A^(-1) = I: {is_I1}")
    print(f"A^(-1)*A = I: {is_I2}")
    
    return x, end - start

def solve_LU(A, b):
    A_np = np.array(A)
    b_np = np.array(b)
    
    start = time.time()
    P, L, U = scipy.linalg.lu(A_np)
    
    y = scipy.linalg.solve_triangular(L, P @ b_np, lower=True)
    
    x = scipy.linalg.solve_triangular(U, y, lower=False)
    end = time.time()
    
    return x, end - start

def solve_QR(A, b):
    A_np = np.array(A)
    b_np = np.array(b)
    
    start = time.time()
    Q, R = np.linalg.qr(A_np)
    
    x = np.linalg.solve(R, Q.T @ b_np)
    end = time.time()
    
    return x, end - start

def verify_solution(A, b, x):
    A_np = np.array(A)
    b_np = np.array(b)
    x_np = np.array(x)
    
    result = A_np @ x_np
    is_correct = np.allclose(result, b_np)
    
    return is_correct

def plot_times(sizes, trials = 100):
    times_lu = []
    times_inv = []
    times_qr = []
    
    for n in sizes:
        sum_time_lu = 0
        sum_time_inv = 0
        sum_time_qr = 0
        for k in range(trials):
            print(f"Calculating for size n={n}")
            A = gen_A(n)
            b = gen_b(n)
            
            _, time_lu = solve_LU(A, b)
            sum_time_lu += time_lu
            
            _, time_inv = solve_inverse_np(A, b)
            sum_time_inv += time_inv
            
            _, time_qr = solve_QR(A, b)
            sum_time_qr += time_qr

        times_lu.append(sum_time_lu/trials)
        times_inv.append(sum_time_inv/trials)
        times_qr.append(sum_time_qr/trials)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_lu, 'o-', label='LU')
    plt.plot(sizes, times_inv, 's-', label='Odwracanie macierzy')
    plt.plot(sizes, times_qr, '^-', label='QR')
    plt.xlabel('Wielkość macierzy(n)')
    plt.ylabel('Czas wykonania(s)')
    plt.title('Porównanie rozwiązywania układu równań różnymi metodami')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # n = int(input("Podaj liczbę n (rozmiar układu równań): "))
    n = 10
    A = gen_A(n)
    b = gen_b(n)
    
    print(f"Rozwiązywanie układu {n} równań...")
    
    x_lu, time_lu = solve_LU(A, b)
    is_correct_lu = verify_solution(A, b, x_lu)
    print(f"LU Decomposition: {time_lu:.6f} seconds, solution correct: {is_correct_lu}")
    
    x_inv, time_inv = solve_inverse_np(A, b)
    is_correct_inv = verify_solution(A, b, x_inv)
    print(f"Matrix Inversion: {time_inv:.6f} seconds, solution correct: {is_correct_inv}")
    
    x_qr, time_qr = solve_QR(A, b)
    is_correct_qr = verify_solution(A, b, x_qr)
    print(f"QR Decomposition: {time_qr:.6f} seconds, solution correct: {is_correct_qr}")
    
    print("\Porównanie rozwiązań:")
    print(f"LU i inwersja dają ten sam wynik: {np.allclose(x_lu, x_inv)}")
    print(f"LU i QR dają ten sam wynik: {np.allclose(x_lu, x_qr)}")
    print(f"Inwersja i QR dają ten sam wynik: {np.allclose(x_inv, x_qr)}")
    
    print(f"Generowanie wykresu...")

    sizes = [20 + 20 * i for i in range(5)]
    plot_times(sizes)

if __name__ == "__main__":
    main()
