import numpy as np
import matplotlib.pyplot as plt 

# === PARAMÈTRES ===
L = 10
N = 100
F1, FN = 0, 0  # Conditions aux limites
C = 10
hx = L/N

# === CONSTRUCTION DU SYSTÈME ===
def matrice(N, C, hx, F1, FN):
    A, B = np.zeros((N, N)), np.zeros(N)
    
    # Conditions aux limites
    A[0, 0] = 1.0
    B[0] = F1
    A[N-1, N-1] = 1.0
    B[N-1] = FN
    
    # Points intérieurs (schéma différences finies)
    for i in range(1, N - 1):
        A[i, i-1] = 1/hx**2
        A[i, i]   = -2/hx**2
        A[i, i+1] = 1/hx**2
        B[i] = C
        
    return A, B

# === RÉSOLUTION ===
A, B = matrice(N, C, hx, F1, FN)
S = np.linalg.solve(A, B)

# === VISUALISATION ===
X = np.linspace(0, L, N)
plt.plot(X, S)
plt.title(f"Solution de $f''(x) = {C}$")
plt.xlabel("Position x (m)")
plt.ylabel("Valeur de f(x)")
plt.grid(True)
plt.show()