#########################################################################################################################################################################################################################################################################################
import numpy as np
import matplotlib.pyplot as plt 
import scipy.sparse as sp 
from scipy.sparse.linalg import spsolve
import copy 

# Paramètres physiques du tube
R0 = 0.00068          # Rayon maximal (m)
alpha0_val = 0.102   # Paramètre de forme au repos
n1_const = 1.0         # Exposant branche compression
n2_const = 10.0        # Exposant branche expansion

# Calcul du rayon au repos
R_repos = R0 * np.sqrt(alpha0_val)

# Paramètres de simulation
L = 0.0047            # Longueur du canal (m)

# Pressions d'entrée et sortie
P_entree = +490.5    # Pression entrée (Pa)
P_sortie = 451.3      # Pression sortie (Pa)
P0 = 0.0               # Pression externe (Pa)

# Constantes loi comportementale tube
P1_const = 53       # Paramètre pression compression
P2_const = -4635     # Paramètre pression expansion

# Paramètres grille numérique
n_base = 100           # Points direction longitudinale
m_base = 100           # Points direction radiale
facteur_zoom_visuel = 1

n = n_base
m = int(m_base * facteur_zoom_visuel)

# Dimensions domaine calcul
l = R0 * 2*facteur_zoom_visuel             # Hauteur domaine (m)
hx = L/n               # Pas spatial x
hy = l/m               # Pas spatial y
C = 1.8e-5             # Coefficient viscosité

# Tailles systèmes
Np = n*m
Nu = (n+1)*m
Nv = n*(m+1)
N = Np + Nu + Nv

# Géométrie initiale
y_centre = l / 2.0
R = np.full(n + 1, R_repos)  # Rayon initial partout

# Fonctions d'indexation variables
def idx_P(i, j, n):
    return (j * n + i) 

def idx_U(i, j, n):
    return (j * (n + 1) + i) + Np

def idx_V(i, j, n, m):
    return (j * n + i) + Np + Nu

# Loi comportementale tube élastique
def D_tube(P, P_pl, P1, P2, n1, n2, alpha0, Dmax):
    P_trans = P - P_pl

    if P_trans <= 0:  # Compression
        terme = 1 - (P_trans / P1)
        terme = max(terme, 1e-9)
        D = Dmax * np.sqrt(alpha0 * terme ** (-n1))
    else:  # Expansion
        terme = 1 - (P_trans / P2)
        terme = max(terme, 1e-9)
        D = Dmax * np.sqrt(1 - (1 - alpha0) * terme ** (-n2))
    
    return D

# Construction matrice système
def matrixes(n, m, C, hx, hy, P_entree_BC, P_sortie_BC, R_geom, l_geom):
    Np = n * m
    Nu = (n + 1) * m
    Nv = n * (m + 1)
    N = Np + Nu + Nv
    A = sp.lil_matrix((N, N))
    B = np.zeros(N)
    y_centre_geom = l_geom / 2.0

    # Équations de continuité (pression)
    for j in range(m):
        for i in range(n):
            K = idx_P(i, j, n) 
            y_phys = (j + 0.5) * hy
            R_local = (R_geom[i] + R_geom[i+1]) / 2.0
            dist_au_centre = abs(y_phys - y_centre_geom)
            
            if dist_au_centre > R_local:  # Point dans mur
                if y_phys < y_centre_geom:  # Mur inférieur
                    A[K, K] = -1.0
                    if j+1 < m: 
                        A[K, idx_P(i, j + 1, n)] = 1.0
                else:  # Mur supérieur
                    A[K, K] = 1.0
                    if j-1 >= 0: 
                        A[K, idx_P(i, j - 1, n)] = -1.0
            elif i == 0:  # Condition entrée
                A[K, K] = 1.0
                B[K] = P_entree_BC
            elif i == n - 1:  # Condition sortie
                A[K, K] = 1.0
                B[K] = P_sortie_BC
            else:  # Point fluide intérieur
                A[K, idx_U(i + 1, j, n)] = 1.0 / hx
                A[K, idx_U(i, j, n)] = -1.0 / hx
                A[K, idx_V(i, j + 1, n, m)] = 1.0 / hy
                A[K, idx_V(i, j, n, m)] = -1.0 / hy

    # Équations quantité mouvement U
    for j in range(m):
        for i in range(n + 1):
            K = idx_U(i, j, n)
            y_phys = (j + 0.5) * hy
            R_local = R_geom[i]
            dist_au_centre = abs(y_phys - y_centre_geom)
            
            if dist_au_centre > R_local:  # Point dans mur
                A[K, K] = 1.0
            elif i == 0:  # Condition entrée
                A[K, K] = -1.0
                A[K, idx_U(i + 1, j, n)] = 1.0
            elif i == n:  # Condition sortie
                A[K, K] = 1.0
                A[K, idx_U(i - 1, j, n)] = -1.0
            else:  # Point fluide intérieur
                A[K, idx_P(i, j, n)] = -1.0 / hx
                A[K, idx_P(i - 1, j, n)] = 1.0 / hx
                A[K, K] = -2.0 * C * (1.0/hx**2 + 1.0/hy**2)
                A[K, idx_U(i + 1, j, n)] = C / hx**2
                A[K, idx_U(i - 1, j, n)] = C / hx**2
                A[K, idx_U(i, j + 1, n)] = C / hy**2
                A[K, idx_U(i, j - 1, n)] = C / hy**2

    # Équations quantité mouvement V
    for j in range(m + 1):
        for i in range(n):
            K = idx_V(i, j, n, m)
            y_phys = j * hy
            R_local = (R_geom[i] + R_geom[i+1]) / 2.0
            dist_au_centre = abs(y_phys - y_centre_geom)
            
            if j == 0 or j == m or dist_au_centre > R_local:  # Bords
                A[K, K] = 1.0
            elif i == 0:  # Condition entrée
                A[K, K] = 1.0
            elif i == n - 1:  # Condition sortie
                A[K, K] = 1.0
                A[K, idx_V(i - 1, j, n, m)] = -1.0
            else:  # Point fluide intérieur
                A[K, idx_P(i, j, n)] = -1.0 / hy
                A[K, idx_P(i, j - 1, n)] = 1.0 / hy
                A[K, K] = -2.0 * C * (1.0/hx**2 + 1.0/hy**2)
                A[K, idx_V(i + 1, j, n, m)] = C / hx**2
                A[K, idx_V(i - 1, j, n, m)] = C / hx**2
                A[K, idx_V(i, j + 1, n, m)] = C / hy**2
                A[K, idx_V(i, j - 1, n, m)] = C / hy**2
                
    return A.tocsr(), B

# Boucle adaptation géométrie
max_iterations = 1000
tolerance = 1e-5 

print("Début calcul adaptation géométrie...")
print(f"Pression entrée: {P_entree} Pa, Sortie: {P_sortie} Pa")

iteration = 0
changement = True
R_history = [R.copy()] 
convergence_data = [] 

while iteration < max_iterations and changement:
    print(f"Itération {iteration + 1}")
    
    A, B = matrixes(n, m, C, hx, hy, P_entree, P_sortie, R, l)
    
    try:
        S = spsolve(A, B)
    except Exception as e:
        print(f"Erreur résolution: {e}")
        break
    
    offset_P = 0
    offset_U = Np
    offset_V = Np + Nu
    P_vecteur = S[offset_P:offset_U]
    P_grille = P_vecteur.reshape((m, n))
    
    # Extraction pression paroi
    pression_au_mur = np.zeros(n)
    y_centre_phys = l / 2.0
    for i in range(n): 
        R_local_p = (R[i] + R[i+1]) / 2.0
        y_mur_sup_phys = y_centre_phys + R_local_p
        j_mur_idx = int((y_mur_sup_phys / hy) - 0.5 - 1e-6)
        j_mur_idx = min(max(j_mur_idx, 0), m - 1)
        pression_au_mur[i] = P_grille[j_mur_idx, i]

    R_ancien = R.copy()
    changement = False
    
    # Mise à jour géométrie
    for i in range(n + 1):
        if i == 0:
            pression_interieure = pression_au_mur[0]
        elif i == n:
            pression_interieure = pression_au_mur[n-1]
        else:
            pression_interieure = (pression_au_mur[i-1] + pression_au_mur[i]) / 2.0
        
        R_new = D_tube(pression_interieure, P0, P1_const, P2_const, 
                      n1_const, n2_const, alpha0_val, 2*R0) / 2.0
        
        # Relaxation convergence
        R_new = R_ancien[i] + (R_new - R_ancien[i]) / (-np.log10(tolerance))
        
        change = abs(R_new - R_ancien[i])
        if change > tolerance:
            changement = True

        R[i] = R_new

    R_history.append(R.copy())
    convergence_data.append({
        'iteration': iteration,
        'changement_max': change,
    })
    
    iteration += 1
    
    print(f"Rayon moyen: {np.mean(R):.6f} m")

print(f"Calcul terminé après {iteration} itérations")


























####################################################################################################################
####################################################################################################################
############################################### PARTIE PLOTAGE #####################################################
####################################################################################################################
####################################################################################################################
# Extraction solution finale
print("Extraction résultats...")

P_vecteur = S[offset_P:offset_U]
U_vecteur = S[offset_U:offset_V]
V_vecteur = S[offset_V:]

P_grille_stag = P_vecteur.reshape((m, n))
U_grille_stag = U_vecteur.reshape((m, n + 1))
V_grille_stag = V_vecteur.reshape((m + 1, n))

P_grille = P_grille_stag
Ux_grille = (U_grille_stag[:, :-1] + U_grille_stag[:, 1:]) / 2.0
Uy_grille = (V_grille_stag[:-1, :] + V_grille_stag[1:, :]) / 2.0

# Visualisation convergence
plt.figure(figsize=(12, 8))
plt.suptitle("Convergence simulation", fontsize=16)

# Évolution rayons
plt.subplot(2, 2, 1)
iterations = range(len(R_history))
for i in range(0, n+1, max(1, (n+1)//10)):
    rayons_colonne = [R_history[iter][i] for iter in iterations]
    plt.plot(iterations, rayons_colonne, label=f'Colonne {i}')
plt.xlabel('Itération')
plt.ylabel('Rayon (m)')
plt.title('Évolution rayons par colonne')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# Profil rayon final
plt.subplot(2, 2, 2)
x_positions = np.linspace(0, L, n+1)
plt.plot(x_positions * 100, R * 100, 'b-', linewidth=2, label='Rayon final')
plt.axhline(y=R0 * 100, color='r', linestyle='--', label='Rayon maximal')
plt.axhline(y=R_repos * 100, color='g', linestyle='--', label='Rayon repos')
plt.xlabel('Position x (cm)')
plt.ylabel('Rayon (cm)')
plt.title('Profil rayon final')
plt.legend()
plt.grid(True, alpha=0.3)

# Statistiques rayons
plt.subplot(2, 2, 3)
rayon_moyen = [np.mean(R_history[iter]) for iter in iterations]
rayon_min = [np.min(R_history[iter]) for iter in iterations]
rayon_max = [np.max(R_history[iter]) for iter in iterations]
plt.plot(iterations, rayon_moyen, 'b-', label='Moyen')
plt.plot(iterations, rayon_min, 'r--', label='Minimum')
plt.plot(iterations, rayon_max, 'g--', label='Maximum')
plt.xlabel('Itération')
plt.ylabel('Rayon (m)')
plt.title('Statistiques rayons')
plt.legend()
plt.grid(True, alpha=0.3)

# Convergence
plt.subplot(2, 2, 4)
changements = [data['changement_max'] for data in convergence_data if data['changement_max'] > 0]
if changements:
    plt.semilogy(range(1, len(changements)+1), changements, 'r-')
plt.xlabel('Itération')
plt.ylabel('Changement max (log)')
plt.title('Convergence')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Visualisation champs physiques
x_coords = np.linspace(hx/2.0, L - hx/2.0, n)
y_coords = np.linspace(hy/2.0, l - hy/2.0, m)
X, Y = np.meshgrid(x_coords, y_coords)

plot_extent = [0, L * 100, 0, l * 100]
pas = max(1, n // 20)

# Masque parois
R_local_p_centres = (R[:-1] + R[1:]) / 2.0 
dist_au_centre_p = abs(y_coords.reshape(m, 1) - y_centre)
masque_mur = dist_au_centre_p > R_local_p_centres.reshape(1, n)

P_grille_plot = np.where(masque_mur, np.nan, P_grille)
Ux_grille_plot = np.where(masque_mur, np.nan, Ux_grille)
Uy_grille_plot = np.where(masque_mur, np.nan, Uy_grille)

cmap_pression = copy.copy(plt.cm.viridis)
cmap_pression.set_bad('black')
cmap_vitesse = copy.copy(plt.cm.coolwarm)
cmap_vitesse.set_bad('black')

# Champ pression + vecteurs vitesse
plt.figure(figsize=(12, 7))
plt.imshow(P_grille_plot, extent=plot_extent, origin='lower', 
           aspect='auto', cmap=cmap_pression)
plt.colorbar(label='Pression (Pa)')
plt.quiver(X[::pas, ::pas] * 100, Y[::pas, ::pas] * 100, 
           Ux_grille_plot[::pas, ::pas], Uy_grille_plot[::pas, ::pas], 
           color='black', scale=np.nanmax(Ux_grille)*20)
plt.xlabel('Position x (cm)')
plt.ylabel('Position y (cm)')
plt.title('Pression et vitesse')
plt.show()

# Champ vitesse horizontale
plt.figure(figsize=(12, 7))
plt.imshow(Ux_grille_plot, extent=plot_extent, origin='lower', 
           aspect='auto', cmap=cmap_vitesse)
plt.colorbar(label='Vitesse Ux (m/s)')
plt.xlabel('Position x (cm)')
plt.ylabel('Position y (cm)')
plt.title('Vitesse horizontale')
plt.show()

# Vérification loi comportementale
eps = 1e-12

# Extraction pression paroi
P_wall = np.zeros(n)
y_centre_phys = l / 2.0
for i in range(n):
    R_local_p = (R[i] + R[i+1]) / 2.0
    y_mur_sup_phys = y_centre_phys + R_local_p
    j_mur_idx = int((y_mur_sup_phys / hy) - 0.5 - 1e-6)
    j_mur_idx = min(j_mur_idx, m - 1)
    j_mur_idx = max(j_mur_idx, 0)
    P_wall[i] = P_grille[j_mur_idx, i]

D_centers = (R[:-1] + R[1:])
Dmax = 2 * R0

x1 = y1 = np.array([])
x2 = y2 = np.array([])
fit1_ok = False
fit2_ok = False

P_wall_transmurale = P_wall - P0

# Analyse branche compression
mask1 = (P_wall_transmurale <= 0)
if np.any(mask1):
    D1 = D_centers[mask1]
    P1_wall = P_wall_transmurale[mask1]

    arg_x1 = 1.0 - (P1_wall / (P1_const + eps))
    arg_y1 = (D1 / (Dmax + eps))**2

    valid_mask = (arg_x1 > 0) & (arg_y1 > 0)
    x1 = np.log(arg_x1[valid_mask])
    y1 = np.log(arg_y1[valid_mask])

    if x1.size >= 2 and np.isfinite(x1).all() and np.isfinite(y1).all():
        a1, b1 = np.polyfit(x1, y1, 1)
        est_n1 = -a1
        est_alpha0 = np.exp(b1)
        fit1_ok = True

# Analyse branche expansion
mask2 = (P_wall_transmurale > 0)
if np.any(mask2):
    D2 = D_centers[mask2]
    P2_wall = P_wall_transmurale[mask2]

    arg_x2 = 1.0 - (P2_wall / (P2_const + eps))
    arg_y2 = 1.0 - (D2 / (Dmax + eps))**2

    valid_mask = (arg_x2 > 0) & (arg_y2 > 0)
    x2 = np.log(arg_x2[valid_mask])
    y2 = np.log(arg_y2[valid_mask])

    if x2.size >= 2 and np.isfinite(x2).all() and np.isfinite(y2).all():
        a2, b2 = np.polyfit(x2, y2, 1)
        est_n2 = -a2
        est_alpha2 = np.exp(b2)
        fit2_ok = True

# Graphiques vérification loi
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
if fit1_ok:
    plt.scatter(x1, y1, s=20, c='C0', label='Données compression')
    x1_lin = np.linspace(np.min(x1), np.max(x1), 2)
    plt.plot(x1_lin, a1*x1_lin + b1, 'r--', label=f'Régression')
    plt.text(0.05, 0.95, f'n1 estimé: {est_n1:.3f}\nalpha0 estimé: {est_alpha0:.3f}',
             transform=plt.gca().transAxes, va='top', ha='left', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.8))
else:
    plt.text(0.5, 0.5, "Pas données compression", transform=plt.gca().transAxes, ha='center')
plt.xlabel('ln(1 - Ptrans/P1)')
plt.ylabel('ln((D/Dmax)²)')
plt.title('Branche compression')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
if fit2_ok:
    plt.scatter(x2, y2, s=20, c='C1', label='Données expansion')
    x2_lin = np.linspace(np.min(x2), np.max(x2), 2)
    plt.plot(x2_lin, a2*x2_lin + b2, 'r--', label=f'Régression')
    plt.text(0.05, 0.95, f'n2 estimé: {est_n2:.3f}\n1-alpha0 estimé: {est_alpha2:.3f}',
             transform=plt.gca().transAxes, va='top', ha='left', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.8))
else:
    plt.text(0.5, 0.5, "Pas données expansion", transform=plt.gca().transAxes, ha='center')
plt.xlabel('ln(1 - Ptrans/P2)')
plt.ylabel('ln(1 - (D/Dmax)²)')
plt.title('Branche expansion')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print("Simulation terminée")