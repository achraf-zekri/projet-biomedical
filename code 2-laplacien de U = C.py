import numpy as np
import matplotlib.pyplot as plt 

# === PARAMÈTRES ===
n, m, L, l = 40, 80, 5.32e-2, 7.54e-2  
hx, hy = L/n, l/m 
C = -10
N = n * m

# === FONCTIONS UTILITAIRES ===
def be(i, j, n): return j * n + i  # Index 1D -> 2D
def re(k, n): return (k % n, k // n)  # Index 2D -> 1D

def matrixes(n, m, C, hx, hy):
    N = m * n
    A, B = np.zeros((N, N)), np.zeros(N)
    
    for K in range(N):
        i, j = re(K, n)

        # Conditions aux limites
        if j == 0 or j == m - 1:  # Bords haut/bas
            A[K, K] = 1.0
            B[K] = 0.0
        elif i == 0:  # Entrée gauche
            A[K, K] = 1.0
            A[K, be(i + 1, j, n)] = -1.0
        elif i == n - 1:  # Sortie droite
            A[K, K] = 1.0
            A[K, be(i - 1, j, n)] = -1.0
        else:  # Point intérieur
            A[K, K] = -2 * (1/hx**2 + 1/hy**2)
            A[K, be(i - 1, j, n)] = 1/hx**2
            A[K, be(i + 1, j, n)] = 1/hx**2
            A[K, be(i, j - 1, n)] = 1/hy**2
            A[K, be(i, j + 1, n)] = 1/hy**2
            B[K] = C
            
    return A, B    

# === CONSTRUCTION ET RÉSOLUTION ===
A, B = matrixes(n, m, C, hx, hy)
S = np.linalg.solve(A, B)

# === VISUALISATION ===
plt.figure(figsize=(10, 8))
plt.imshow(S.reshape((m, n)), extent=[0, L, 0, l], 
           origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Valeur de S')
plt.xlabel('Position x (m)')
plt.ylabel('Position y (m)')
plt.title(f'Solution de $\\Delta U = {C}$')
plt.show()