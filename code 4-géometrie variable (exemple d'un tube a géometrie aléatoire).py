import numpy as np
import matplotlib.pyplot as plt 
import scipy.sparse as sp 
from scipy.sparse.linalg import spsolve
import copy 

# === PARAMÈTRES PHYSIQUES ET NUMÉRIQUES ===
n, m, L, l = 100, 250, 5.32e-2, 7.54e-2 
hx, hy = L/n, l/m 
C, P1, P2 = 1.8, 400, 200


# === GÉOMÉTRIE ALÉATOIRE ===
y_centre = l / 2.0
rayon_moyen = 2.5e-2
amplitude_aleatoire = 2.5e-2
taille_lissage = 8

bruit_brut = (np.random.rand(n + 1) - 0.5) * 2 * amplitude_aleatoire
kernel = np.ones(taille_lissage) / taille_lissage
bruit_lisse = np.convolve(bruit_brut, kernel, mode='same')
R = np.clip(rayon_moyen + bruit_lisse, 0.01 * l, y_centre * 0.95)


# === DIMENSIONS DES SYSTÈMES ===
Np, Nu, Nv = n*m, (n+1)*m, n*(m+1)
N = Np + Nu + Nv

# === FONCTIONS D'INDEXATION ===
def idx_P(i, j, n): return j * n + i 
def idx_U(i, j, n): return j * (n + 1) + i + Np
def idx_V(i, j, n, m): return j * n + i + Np + Nu

# === CONSTRUCTION DU SYSTÈME ===
def matrixes(n, m, C, hx, hy, P1, P2, R, l):
    Np, Nu, Nv = n*m, (n+1)*m, n*(m+1)
    N = Np + Nu + Nv
    A, B = sp.lil_matrix((N, N)), np.zeros(N)
    y_centre = l / 2.0

    # Équation de continuité (Pression)
    for j in range(m):
        for i in range(n):
            K, y_phys = idx_P(i, j, n), (j + 0.5) * hy
            R_local, dist = (R[i] + R[i+1]) / 2.0, abs(y_phys - y_centre)

            if dist > R_local:  # Mur solide
                if y_phys < y_centre:  # Mur inférieur
                    A[K, K], A[K, idx_P(i, j + 1, n)] = -1.0, 1.0
                else:  # Mur supérieur
                    A[K, K], A[K, idx_P(i, j - 1, n)] = 1.0, -1.0
            elif i == 0:  # Entrée
                A[K, K], B[K] = 1.0, P1
            elif i == n - 1:  # Sortie
                A[K, K], B[K] = 1.0, P2
            else:  # Intérieur fluide
                A[K, idx_U(i + 1, j, n)] = 1.0/hx
                A[K, idx_U(i, j, n)] = -1.0/hx
                A[K, idx_V(i, j + 1, n, m)] = 1.0/hy
                A[K, idx_V(i, j, n, m)] = -1.0/hy

    # Équation de quantité de mouvement (Vitesse U)
    for j in range(m):
        for i in range(n + 1):
            K, y_phys = idx_U(i, j, n), (j + 0.5) * hy
            R_local, dist = R[i], abs(y_phys - y_centre)

            if dist > R_local:  # Mur solide
                A[K, K], B[K] = 1.0, 0.0
            elif i == 0:  # Entrée
                A[K, K], A[K, idx_U(i + 1, j, n)] = -1.0, 1.0
            elif i == n:  # Sortie
                A[K, K], A[K, idx_U(i - 1, j, n)] = 1.0, -1.0
            else:  # Intérieur fluide
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
            K, y_phys = idx_V(i, j, n, m), j * hy
            R_local, dist = (R[i] + R[i+1]) / 2.0, abs(y_phys - y_centre)

            if j == 0 or j == m or dist > R_local:  # Mur solide
                A[K, K], B[K] = 1.0, 0.0
            elif i == 0:  # Entrée
                A[K, K], B[K] = 1.0, 0.0
            elif i == n - 1:  # Sortie
                A[K, K], A[K, idx_V(i - 1, j, n, m)] = 1.0, -1.0
            else:  # Intérieur fluide
                A[K, idx_P(i, j, n)] = -1.0/hy
                A[K, idx_P(i, j - 1, n)] = 1.0/hy
                A[K, K] = -2.0 * C * (1.0/hx**2 + 1.0/hy**2)
                A[K, idx_V(i + 1, j, n, m)] = C/hx**2
                A[K, idx_V(i - 1, j, n, m)] = C/hx**2
                A[K, idx_V(i, j + 1, n, m)] = C/hy**2
                A[K, idx_V(i, j - 1, n, m)] = C/hy**2
                
    return A.tocsr(), B

# === RÉSOLUTION ===
print("Construction des matrices...")
A, B = matrixes(n, m, C, hx, hy, P1, P2, R, l)
print("Résolution du système creux...")
S = spsolve(A, B)
print("Résolution terminée.")

# === POST-TRAITEMENT ===
# Extraction des champs
P_vecteur = S[:Np]
U_vecteur = S[Np:Np+Nu] 
V_vecteur = S[Np+Nu:]

# Remise en forme et interpolation
P_grille = P_vecteur.reshape((m, n))
Ux_grille = (U_vecteur.reshape((m, n + 1))[:, :-1] + U_vecteur.reshape((m, n + 1))[:, 1:]) / 2.0
Uy_grille = (V_vecteur.reshape((m + 1, n))[:-1, :] + V_vecteur.reshape((m + 1, n))[1:, :]) / 2.0

# === MASQUAGE DES RÉGIONS SOLIDES ===
x_coords = np.linspace(hx/2.0, L - hx/2.0, n)
y_coords = np.linspace(hy/2.0, l - hy/2.0, m)
X, Y = np.meshgrid(x_coords, y_coords)

R_local_p = (R[:-1] + R[1:]) / 2.0 
dist_au_centre_p = abs(y_coords.reshape(m, 1) - l/2.0)
masque_mur = dist_au_centre_p > R_local_p.reshape(1, n)

P_grille_plot = np.where(masque_mur, np.nan, P_grille)
Ux_grille_plot = np.where(masque_mur, np.nan, Ux_grille)
Uy_grille_plot = np.where(masque_mur, np.nan, Uy_grille)

cmap_pression = copy.copy(plt.cm.viridis)
cmap_vitesse = copy.copy(plt.cm.coolwarm)
cmap_pression.set_bad('black')
cmap_vitesse.set_bad('black')

# === VISUALISATION ===
plot_extent, pas = [0, L, 0, l], 5

# Graphique 1: Pression + vecteurs vitesse
plt.figure(figsize=(10, 7))
plt.imshow(P_grille_plot, extent=plot_extent, origin='lower', aspect='auto', cmap=cmap_pression)
plt.colorbar(label='Pression (Pa)')
plt.quiver(X[::pas, ::pas], Y[::pas, ::pas], 
           Ux_grille_plot[::pas, ::pas], Uy_grille_plot[::pas, ::pas], color='black')
plt.xlabel('Position x (m)'); plt.ylabel('Position y (m)')
plt.title('Champ de Pression et Vecteurs Vitesse')
plt.show()

# Graphique 2: Vitesse horizontale
plt.figure(figsize=(10, 7))
plt.imshow(Ux_grille_plot, extent=plot_extent, origin='lower', aspect='auto', cmap=cmap_vitesse)
plt.colorbar(label='Vitesse Ux (m/s)')
plt.xlabel('Position x (m)'); plt.ylabel('Position y (m)')
plt.title('Champ de Vitesse Horizontale (Ux)')
plt.show()

# Graphique 3: Vitesse verticale  
plt.figure(figsize=(10, 7))
plt.imshow(Uy_grille_plot, extent=plot_extent, origin='lower', aspect='auto', cmap=cmap_vitesse)
plt.colorbar(label='Vitesse Uy (m/s)')
plt.xlabel('Position x (m)'); plt.ylabel('Position y (m)')
plt.title('Champ de Vitesse Verticale (Uy)')
plt.show()

# Graphique 4: Pression seule
plt.figure(figsize=(10, 7))
plt.imshow(P_grille_plot, extent=plot_extent, origin='lower', aspect='auto', cmap=cmap_pression)
plt.colorbar(label='Pression (Pa)')
plt.xlabel('Position x (m)'); plt.ylabel('Position y (m)')
plt.title('Champ de Pression (P)')
plt.show()