import math
import numpy as np

# ==========================================================
# PROGRAM: Demonstrasi Penyelesaian Sistem Nonlinear (M06)
# Nama : M. Azyan Naufan Rosada
# NIM  : 21120123140146
# ==========================================================

# Persamaan nonlinear
def f1(x, y):
    return x**2 + x*y - 10

def f2(x, y):
    return y + 3*x*y**2 - 57

# ==========================================================
# Fungsi Iterasi Titik Tetap (NIMx = 2 → g1B dan g2A)
# ==========================================================
def g1B(x, y):
    val = 10 - x*y
    return math.sqrt(val) if val >= 0 else float('nan')

def g2A(x, y):
    return 57 - 3*x*y**2

# ==========================================================
# Metode Iterasi Jacobi
# ==========================================================
def fixed_point_jacobi(x0, y0, tol, max_iter=200):
    print("\n=== Iterasi Titik Tetap - Metode Jacobi ===")
    x, y = x0, y0
    for k in range(max_iter):
        x_new = g1B(x, y)
        y_new = g2A(x, y)
        if any(map(math.isnan, [x_new, y_new])):
            print("Divergen pada iterasi ke-", k + 1)
            return None
        err = math.sqrt((x_new - x)**2 + (y_new - y)**2)
        print(f"Iter {k+1:3d}: x={x_new:.6f}, y={y_new:.6f}, error={err:.6e}")
        if err < tol:
            print(f"Konvergen pada iterasi ke-{k+1}")
            return x_new, y_new
        x, y = x_new, y_new
    print("Tidak konvergen dalam batas iterasi.")
    return None

# ==========================================================
# Metode Iterasi Seidel (VERSI PERBAIKAN)
# ==========================================================
def fixed_point_seidel(x0, y0, tol, max_iter=200):
    print("\n=== Iterasi Titik Tetap - Metode Seidel ===")
    x, y = x0, y0
    for k in range(max_iter):
        x_old, y_old = x, y  # 1. Simpan nilai lama

        x = g1B(x_old, y_old)
        # Periksa x sebelum digunakan untuk menghitung y
        if math.isnan(x):
            print(f"Iter {k+1:3d}: x=nan")
            print("Divergen (akar negatif) pada iterasi ke-", k + 1)
            return None

        y = g2A(x, y_old) # Gunakan x baru dan y lama

        # 2. Tambahkan pengaman overflow
        if abs(y) > 1e15: # Hentikan jika nilai y meledak
            print(f"Iter {k+1:3d}: x={x:.6f}, y={y:.6f}")
            print("Divergen (overflow numerik) pada iterasi ke-", k + 1)
            return None

        # 3. Hitung error dari nilai lama vs nilai baru
        err = math.sqrt((x - x_old)**2 + (y - y_old)**2)
        print(f"Iter {k+1:3d}: x={x:.6f}, y={y:.6f}, error={err:.6e}")

        if err < tol:
            print(f"Konvergen pada iterasi ke-{k+1}")
            return x, y
            
    print("Tidak konvergen dalam batas iterasi.")
    return None

# ==========================================================
# Metode Newton-Raphson
# ==========================================================
def newton_raphson(x0, y0, tol, max_iter=100):
    print("\n=== Metode Newton-Raphson ===")
    x, y = x0, y0
    for k in range(max_iter):
        J = np.array([
            [2*x + y, x],
            [3*y**2, 1 + 6*x*y]
        ])
        F = np.array([-f1(x, y), -f2(x, y)])
        delta = np.linalg.solve(J, F)
        x_new, y_new = x + delta[0], y + delta[1]
        err = math.sqrt(delta[0]**2 + delta[1]**2)
        print(f"Iter {k+1:2d}: x={x_new:.6f}, y={y_new:.6f}, error={err:.6e}")
        if err < tol:
            print(f"Konvergen pada iterasi ke-{k+1}")
            return x_new, y_new
        x, y = x_new, y_new
    print("Tidak konvergen.")
    return None

# ==========================================================
# Metode Secant (Broyden)
# ==========================================================
def broyden_method(x0, y0, tol, max_iter=100):
    print("\n=== Metode Secant (Broyden) ===")
    x, y = x0, y0
    B = np.eye(2)
    F = np.array([f1(x, y), f2(x, y)])
    for k in range(max_iter):
        delta = -np.linalg.solve(B, F)
        x_new, y_new = x + delta[0], y + delta[1]
        F_new = np.array([f1(x_new, y_new), f2(x_new, y_new)])
        yk = F_new - F
        B = B + np.outer((yk - B @ delta), delta) / (delta @ delta)
        err = np.linalg.norm(delta)
        print(f"Iter {k+1:2d}: x={x_new:.6f}, y={y_new:.6f}, error={err:.6e}")
        if err < tol:
            print(f"Konvergen pada iterasi ke-{k+1}")
            return x_new, y_new
        x, y, F = x_new, y_new, F_new
    print("Tidak konvergen.")
    return None

# ==========================================================
# EKSEKUSI PROGRAM
# ==========================================================
if __name__ == "__main__":
    x0, y0 = 1.5, 3.5
    tol = 1e-6

    fixed_point_jacobi(x0, y0, tol)
    fixed_point_seidel(x0, y0, tol)
    newton_raphson(x0, y0, tol)
    broyden_method(x0, y0, tol)
