import numpy as np
import time
import matplotlib.pyplot as plt
from random import randrange

def mmult(a, b):
    if len(a) < 1 or len(b) < 1 or len(b) != len(a[0]):
        return None
    result = [[0 for i in range(len(b[0]))] for j in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(a[0])):
                result[i][j] += a[i][k] * b[k][j]
    return result

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

def plot_times(sizes):
    times_lu = []
    times_inv = []
    times_qr = []
    
    for n in sizes:
        print(f"Calculating for size n={n}")
        A = gen_A(n)
        b = gen_b(n)
        
        _, time_lu = solve_LU(A, b)
        times_lu.append(time_lu)
        
        _, time_inv = solve_inverse_np(A, b)
        times_inv.append(time_inv)
        
        _, time_qr = solve_QR(A, b)
        times_qr.append(time_qr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_lu, 'o-', label='LU Decomposition')
    plt.plot(sizes, times_inv, 's-', label='Matrix Inversion')
    plt.plot(sizes, times_qr, '^-', label='QR Decomposition')
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Execution Time (s)')
    plt.title('Comparison of Linear System Solving Methods')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    n = int(input("Podaj liczbę n (rozmiar układu równań): "))
    
    A = gen_A(n)
    b = gen_b(n)
    
    print(f"Solving system of {n} equations...")
    
    x_lu, time_lu = solve_LU(A, b)
    is_correct_lu = verify_solution(A, b, x_lu)
    print(f"LU Decomposition: {time_lu:.6f} seconds, solution correct: {is_correct_lu}")
    
    x_inv, time_inv = solve_inverse_np(A, b)
    is_correct_inv = verify_solution(A, b, x_inv)
    print(f"Matrix Inversion: {time_inv:.6f} seconds, solution correct: {is_correct_inv}")
    
    x_qr, time_qr = solve_QR(A, b)
    is_correct_qr = verify_solution(A, b, x_qr)
    print(f"QR Decomposition: {time_qr:.6f} seconds, solution correct: {is_correct_qr}")
    
    print("\nComparison of solutions:")
    print(f"LU and Inv are equal: {np.allclose(x_lu, x_inv)}")
    print(f"LU and QR are equal: {np.allclose(x_lu, x_qr)}")
    print(f"Inv and QR are equal: {np.allclose(x_inv, x_qr)}")
    
    plot_homework = input("Do you want to generate the homework plot? (y/n): ")
    if plot_homework.lower() == 'y':
        sizes = [10, 30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000] # 7000, 10000]  # 5 values between 10 and 100
        plot_times(sizes)

if __name__ == "__main__":
    import scipy.linalg
    main()
