import numpy as np
import matplotlib.pyplot as plt 
import scipy.sparse as sp 
from scipy.sparse.linalg import spsolve

# === PARAMÈTRES PHYSIQUES ET NUMÉRIQUES ===
n, m, L, l = 100, 200, 5.32e-2, 7.54e-2  
hx, hy = L/n, l/m 
C = 1.8
P1, P2 = 400, 200

# === DIMENSIONS DES SYSTÈMES ===
Np, Nu, Nv = n*m, (n+1)*m, n*(m+1)
N = Np + Nu + Nv

# === FONCTIONS D'INDEXATION ===
def idx_P(i, j, n): return j * n + i 
def idx_U(i, j, n): return j * (n + 1) + i + Np
def idx_V(i, j, n, m): return j * n + i + Np + Nu

# === CONSTRUCTION DU SYSTÈME ===
def matrixes(n, m, C, hx, hy, P1, P2):
    Np, Nu, Nv = n*m, (n+1)*m, n*(m+1)
    N = Np + Nu + Nv
    
    A, B = sp.lil_matrix((N, N)), np.zeros(N)

    # Équation de continuité (Pression)
    for j in range(m):
        for i in range(n):
            K = idx_P(i, j, n)
            if i == 0:          # Entrée: P = P1
                A[K, K], B[K] = 1.0, P1
            elif i == n - 1:     # Sortie: P = P2
                A[K, K], B[K] = 1.0, P2
            elif j == 0:         # Mur inférieur: dP/dy = 0
                A[K, K], A[K, idx_P(i, j + 1, n)] = -1.0, 1.0
            elif j == m - 1:     # Mur supérieur: dP/dy = 0
                A[K, K], A[K, idx_P(i, j - 1, n)] = 1.0, -1.0
            else:                # Intérieur: div(u) = 0
                A[K, idx_U(i + 1, j, n)] = 1.0/hx
                A[K, idx_U(i, j, n)] = -1.0/hx
                A[K, idx_V(i, j + 1, n, m)] = 1.0/hy
                A[K, idx_V(i, j, n, m)] = -1.0/hy

    # Équation de quantité de mouvement (Vitesse U)
    for j in range(m):
        for i in range(n + 1):
            K = idx_U(i, j, n)
            if i == 0:          # Entrée: du/dx = 0
                A[K, K], A[K, idx_U(i + 1, j, n)] = -1.0, 1.0
            elif i == n:         # Sortie: du/dx = 0
                A[K, K], A[K, idx_U(i - 1, j, n)] = 1.0, -1.0
            elif j == 0 or j == m - 1:  # Murs: u = 0
                A[K, K], B[K] = 1.0, 0.0
            else:                # Intérieur: C*Δu - dP/dx = 0
                A[K, idx_P(i, j, n)] = -1.0/hx
                A[K, idx_P(i - 1, j, n)] = 1.0/hx
                A[K, K] = -2.0 * C * (1.0/hx**2 + 1.0/hy**2)
                A[K, idx_U(i + 1, j, n)] = C/hx**2
                A[K, idx_U(i - 1, j, n)] = C/hx**2
                A[K, idx_U(i, j + 1, n)] = C/hy**2
                A[K, idx_U(i, j - 1, n)] = C/hy**2

    # Équation de quantité de mouvement (Vitesse V)
    for j in range(m + 1):
        for i in range(n):
            K = idx_V(i, j, n, m)
            if j == 0 or j == m:        # Murs: v = 0
                A[K, K], B[K] = 1.0, 0.0
            elif i == 0:                # Entrée: v = 0
                A[K, K], B[K] = 1.0, 0.0
            elif i == n - 1:            # Sortie: dv/dx = 0
                A[K, K], A[K, idx_V(i - 1, j, n, m)] = 1.0, -1.0
            else:                       # Intérieur: C*Δv - dP/dy = 0
                A[K, idx_P(i, j, n)] = -1.0/hy
                A[K, idx_P(i, j - 1, n)] = 1.0/hy
                A[K, K] = -2.0 * C * (1.0/hx**2 + 1.0/hy**2)
                A[K, idx_V(i + 1, j, n, m)] = C/hx**2
                A[K, idx_V(i - 1, j, n, m)] = C/hx**2
                A[K, idx_V(i, j + 1, n, m)] = C/hy**2
                A[K, idx_V(i, j - 1, n, m)] = C/hy**2
                
    return A.tocsr(), B

# === RÉSOLUTION ===
A, B = matrixes(n, m, C, hx, hy, P1, P2)
S = spsolve(A, B)

# === POST-TRAITEMENT ===
# Extraction des champs
P_vecteur = S[:Np]
U_vecteur = S[Np:Np+Nu] 
V_vecteur = S[Np+Nu:]

# Remise en forme grilles décalées
P_grille_stag = P_vecteur.reshape((m, n))
U_grille_stag = U_vecteur.reshape((m, n + 1))
V_grille_stag = V_vecteur.reshape((m + 1, n))

# Interpolation aux centres
P_grille = P_grille_stag
Ux_grille = (U_grille_stag[:, :-1] + U_grille_stag[:, 1:]) / 2.0
Uy_grille = (V_grille_stag[:-1, :] + V_grille_stag[1:, :]) / 2.0

# === VISUALISATION ===
x_coords = np.linspace(hx/2.0, L - hx/2.0, n)
y_coords = np.linspace(hy/2.0, l - hy/2.0, m)
X, Y = np.meshgrid(x_coords, y_coords)
plot_extent, pas = [0, L, 0, l], 5

# Graphique 1: Pression + vecteurs vitesse
plt.figure(figsize=(10, 7))
plt.imshow(P_grille, extent=plot_extent, origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Pression (Pa)')
plt.quiver(X[::pas, ::pas], Y[::pas, ::pas], 
           Ux_grille[::pas, ::pas], Uy_grille[::pas, ::pas], color='black')
plt.xlabel('Position x (m)'); plt.ylabel('Position y (m)')
plt.title('Champ de Pression et Vecteurs Vitesse')
plt.show()

# Graphique 2: Vitesse horizontale
plt.figure(figsize=(10, 7))
plt.imshow(Ux_grille, extent=plot_extent, origin='lower', aspect='auto', cmap='coolwarm')
plt.colorbar(label='Vitesse Ux (m/s)')
plt.xlabel('Position x (m)'); plt.ylabel('Position y (m)')
plt.title('Champ de Vitesse Horizontale (Ux)')
plt.show()

# Graphique 3: Vitesse verticale  
plt.figure(figsize=(10, 7))
plt.imshow(Uy_grille, extent=plot_extent, origin='lower', aspect='auto', cmap='coolwarm')
plt.colorbar(label='Vitesse Uy (m/s)')
plt.xlabel('Position x (m)'); plt.ylabel('Position y (m)')
plt.title('Champ de Vitesse Verticale (Uy)')
plt.show()

# Graphique 4: Pression seule
plt.figure(figsize=(10, 7))
plt.imshow(P_grille, extent=plot_extent, origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Pression (Pa)')
plt.xlabel('Position x (m)'); plt.ylabel('Position y (m)')
plt.title('Champ de Pression (P)')
plt.show()