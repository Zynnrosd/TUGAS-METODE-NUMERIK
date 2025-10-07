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
# Iterasi Titik Tetap - Jacobi
# ==========================================================
def fixed_point_jacobi(x0, y0, tol, max_iter=200):
    print("\n===================================================")
    print("METODE JACOBI (dengan g1B dan g2A)")
    print("Iter\t     x\t\t     y\t\t    |Δx|\t\t    |Δy|")
    print("---------------------------------------------------")

    x, y = x0, y0
    for k in range(max_iter):
        x_new = g1B(x, y)
        y_new = g2A(x, y)
        if any(map(math.isnan, [x_new, y_new])):
            print(f"❌ Terjadi error pada iterasi ke-{k+1}: math domain error")
            print("→ Solusi divergen atau keluar dari domain fungsi.")
            return None
        dx = abs(x_new - x)
        dy = abs(y_new - y)
        print(f"{k:2d}\t{x_new:10.6f}\t{y_new:10.6f}\t{dx:10.6f}\t{dy:10.6f}")
        if math.sqrt(dx**2 + dy**2) < tol:
            print(f"Konvergen pada iterasi ke-{k+1}")
            return x_new, y_new
        x, y = x_new, y_new

    print("Tidak konvergen dalam batas iterasi.")
    return None

# ==========================================================
# Iterasi Titik Tetap - Seidel
# ==========================================================
def fixed_point_seidel(x0, y0, tol, max_iter=200):
    print("\n===================================================")
    print("METODE GAUSS-SEIDEL (dengan g1B dan g2A)")
    print("Iter\t     x\t\t     y\t\t    |Δx|\t\t    |Δy|")
    print("---------------------------------------------------")

    x, y = x0, y0
    for k in range(max_iter):
        x_old, y_old = x, y
        x = g1B(x_old, y_old)
        if math.isnan(x):
            print(f"❌ Terjadi error pada iterasi ke-{k+1}: math domain error")
            print("→ Solusi divergen atau keluar dari domain fungsi.")
            return None

        y = g2A(x, y_old)
        dx = abs(x - x_old)
        dy = abs(y - y_old)

        if abs(y) > 1e15:
            print(f"❌ Divergen (overflow numerik) pada iterasi ke-{k+1}")
            return None

        print(f"{k:2d}\t{x:10.6f}\t{y:10.6f}\t{dx:10.6f}\t{dy:10.6f}")

        if math.sqrt(dx**2 + dy**2) < tol:
            print(f"Konvergen pada iterasi ke-{k+1}")
            return x, y

    print("Tidak konvergen dalam batas iterasi.")
    return None

# ==========================================================
# Metode Newton-Raphson (dengan deltaX dan deltaY)
# ==========================================================
def newton_raphson(x0, y0, tol, max_iter=100):
    print("\n===================================================")
    print("METODE NEWTON-RAPHSON")
    print("r\t     x\t\t     y\t\t deltaX\t\t deltaY")
    print("---------------------------------------------------")
    x, y = x0, y0
    for k in range(max_iter):
        J = np.array([
            [2*x + y, x],
            [3*y**2, 1 + 6*x*y]
        ])
        F = np.array([-f1(x, y), -f2(x, y)])
        delta = np.linalg.solve(J, F)
        dx, dy = delta[0], delta[1]
        x_new, y_new = x + dx, y + dy
        print(f"{k:2d}\t{x_new:10.6f}\t{y_new:10.6f}\t{dx:10.6f}\t{dy:10.6f}")
        if math.sqrt(dx**2 + dy**2) < tol:
            print(f"Konvergen pada iterasi ke-{k+1}")
            return x_new, y_new
        x, y = x_new, y_new
    print("Tidak konvergen.")
    return None

# ==========================================================
# Metode Secant (Broyden) - dengan deltaX dan deltaY
# ==========================================================
def broyden_method(x0, y0, tol, max_iter=100):
    print("\n===================================================")
    print("METODE SECANT (BROyDEN)")
    print("r\t     x\t\t     y\t\t deltaX\t\t deltaY")
    print("---------------------------------------------------")
    x, y = x0, y0
    B = np.eye(2)
    F = np.array([f1(x, y), f2(x, y)])
    for k in range(max_iter):
        delta = -np.linalg.solve(B, F)
        dx, dy = delta[0], delta[1]
        x_new, y_new = x + dx, y + dy
        F_new = np.array([f1(x_new, y_new), f2(x_new, y_new)])
        yk = F_new - F
        B = B + np.outer((yk - B @ delta), delta) / (delta @ delta)
        print(f"{k:2d}\t{x_new:10.6f}\t{y_new:10.6f}\t{dx:10.6f}\t{dy:10.6f}")
        if math.sqrt(dx**2 + dy**2) < tol:
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
    print("Tebakan awal: x0 = 1.5, y0 = 3.5")
    print("Toleransi: ε = 1e-06")
    print("NIMx = 2 → Kombinasi: g1B dan g2A")

    fixed_point_jacobi(x0, y0, tol)
    fixed_point_seidel(x0, y0, tol)
    newton_raphson(x0, y0, tol)
    broyden_method(x0, y0, tol)
