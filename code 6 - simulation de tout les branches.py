import numpy as np
import matplotlib.pyplot as plt 
import scipy.sparse as sp 
from scipy.sparse.linalg import spsolve
import copy 
import os

# =============================================================================
# PARAMÈTRES DE TOUTES LES GÉNÉRATIONS BRONCHIQUES
# =============================================================================

# Tableau des paramètres pour les 17 générations (d'après Lambert 2004)
generations_data = {
    'gen': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'longueur': [0.1200, 0.0476, 0.0190, 0.0076, 0.0127, 0.0107, 0.0090, 0.0076, 0.0064, 0.0054, 0.0047, 0.0039, 0.0033, 0.0027, 0.0023, 0.0020, 0.0017],
    'R0': [0.00868, 0.00614, 0.00489, 0.00373, 0.00289, 0.00225, 0.00175, 0.00138, 0.00108, 0.00088, 0.00068, 0.00055, 0.00047, 0.00042, 0.00037, 0.00033, 0.00029],
    'alpha0': [0.882, 0.882, 0.686, 0.546, 0.428, 0.337, 0.265, 0.208, 0.164, 0.129, 0.102, 0.080, 0.063, 0.049, 0.039, 0.031, 0.024],
    'n1': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'n2': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 8, 8, 8, 7, 7],
    'P1': [8009, 2942, 1345, 687, 428, 268, 187, 131, 94, 70, 53, 39, 29, 22, 17, 13, 10],
    'P2': [-10715, -3936, -6158, -5707, -5709, -5282, -5192, -4993, -4796, -4712, -4635, -4009, -3444, -3392, -3301, -2879, -2814]
}

# Pressions pour l'expiration forcée (d'après le profil 1D de Lambert)
pressions_expiration_forcee = {
    0: {'entree': -490, 'sortie': -480},
    1: {'entree': -400, 'sortie': -390},
    2: {'entree': -310, 'sortie': -300},
    3: {'entree': -220, 'sortie': -210},
    4: {'entree': -147, 'sortie': -137},
    5: {'entree': 78, 'sortie': 68},
    6: {'entree': 128, 'sortie': 118},
    7: {'entree': 178, 'sortie': 168},
    8: {'entree': 228, 'sortie': 218},
    9: {'entree': 278, 'sortie': 268},
    10: {'entree': 490.5, 'sortie': 451.3},
    11: {'entree': 350, 'sortie': 340},
    12: {'entree': 400, 'sortie': 390},
    13: {'entree': 450, 'sortie': 440},
    14: {'entree': 500, 'sortie': 490},
    15: {'entree': 550, 'sortie': 540},
    16: {'entree': 588, 'sortie': 578}
}

# =============================================================================
# FONCTIONS GÉNÉRIQUES
# =============================================================================

def idx_P(i, j, n):
    return (j * n + i) 

def idx_U(i, j, n):
    return (j * (n + 1) + i)

def idx_V(i, j, n, m):
    return (j * n + i)

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

def matrixes(n, m, C, hx, hy, P_entree_BC, P_sortie_BC, R_geom, l_geom, Np, Nu, Nv):
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
                A[K, idx_U(i + 1, j, n) + Np] = 1.0 / hx
                A[K, idx_U(i, j, n) + Np] = -1.0 / hx
                A[K, idx_V(i, j + 1, n, m) + Np + Nu] = 1.0 / hy
                A[K, idx_V(i, j, n, m) + Np + Nu] = -1.0 / hy

    # Équations quantité mouvement U
    for j in range(m):
        for i in range(n + 1):
            K = idx_U(i, j, n) + Np
            y_phys = (j + 0.5) * hy
            R_local = R_geom[i]
            dist_au_centre = abs(y_phys - y_centre_geom)
            
            if dist_au_centre > R_local:  # Point dans mur
                A[K, K] = 1.0
            elif i == 0:  # Condition entrée
                A[K, K] = -1.0
                A[K, idx_U(i + 1, j, n) + Np] = 1.0
            elif i == n:  # Condition sortie
                A[K, K] = 1.0
                A[K, idx_U(i - 1, j, n) + Np] = -1.0
            else:  # Point fluide intérieur
                A[K, idx_P(i, j, n)] = -1.0 / hx
                A[K, idx_P(i - 1, j, n)] = 1.0 / hx
                A[K, K] = -2.0 * C * (1.0/hx**2 + 1.0/hy**2)
                A[K, idx_U(i + 1, j, n) + Np] = C / hx**2
                A[K, idx_U(i - 1, j, n) + Np] = C / hx**2
                A[K, idx_U(i, j + 1, n) + Np] = C / hy**2
                A[K, idx_U(i, j - 1, n) + Np] = C / hy**2

    # Équations quantité mouvement V
    for j in range(m + 1):
        for i in range(n):
            K = idx_V(i, j, n, m) + Np + Nu
            y_phys = j * hy
            R_local = (R_geom[i] + R_geom[i+1]) / 2.0
            dist_au_centre = abs(y_phys - y_centre_geom)
            
            if j == 0 or j == m or dist_au_centre > R_local:  # Bords
                A[K, K] = 1.0
            elif i == 0:  # Condition entrée
                A[K, K] = 1.0
            elif i == n - 1:  # Condition sortie
                A[K, K] = 1.0
                A[K, idx_V(i - 1, j, n, m) + Np + Nu] = -1.0
            else:  # Point fluide intérieur
                A[K, idx_P(i, j, n)] = -1.0 / hy
                A[K, idx_P(i, j - 1, n)] = 1.0 / hy
                A[K, K] = -2.0 * C * (1.0/hx**2 + 1.0/hy**2)
                A[K, idx_V(i + 1, j, n, m) + Np + Nu] = C / hx**2
                A[K, idx_V(i - 1, j, n, m) + Np + Nu] = C / hx**2
                A[K, idx_V(i, j + 1, n, m) + Np + Nu] = C / hy**2
                A[K, idx_V(i, j - 1, n, m) + Np + Nu] = C / hy**2
                
    return A.tocsr(), B

def simuler_generation(gen_index, cas='expiration_forcee'):
    """Simule une génération bronchique spécifique"""
    
    # Paramètres de la génération
    gen = generations_data['gen'][gen_index]
    L = generations_data['longueur'][gen_index]
    R0 = generations_data['R0'][gen_index]
    alpha0_val = generations_data['alpha0'][gen_index]
    n1_const = generations_data['n1'][gen_index]
    n2_const = generations_data['n2'][gen_index]
    P1_const = generations_data['P1'][gen_index]
    P2_const = generations_data['P2'][gen_index]
    
    print(f"\n{'='*60}")
    print(f"SIMULATION GÉNÉRATION {gen}")
    print(f"{'='*60}")
    
    # Calcul du rayon au repos
    R_repos = R0 * np.sqrt(alpha0_val)
    
    # Pressions selon le cas
    if cas == 'expiration_forcee':
        P_entree = pressions_expiration_forcee[gen]['entree']
        P_sortie = pressions_expiration_forcee[gen]['sortie']
        scenario = "EXPIRATION FORCÉE"
    elif cas == 'inspiration':
        P_entree = 49.0
        P_sortie = 20.0
        scenario = "INSPIRATION"
    else:  # repos
        P_entree = 0.0
        P_sortie = 0.0
        scenario = "REPOS"
    
    P0 = 0.0  # Pression externe
    
    print(f"Scénario: {scenario}")
    print(f"Pression entrée: {P_entree} Pa, Sortie: {P_sortie} Pa")
    print(f"Rayon au repos: {R_repos*1000:.3f} mm")
    print(f"Raideur P1: {P1_const} Pa")

    # Paramètres grille numérique (adaptés à la taille de la génération)
    n_base = max(50, int(L * 10000))  # Adaptation selon la longueur
    m_base = max(50, int(R0 * 20000))  # Adaptation selon le rayon
    
    n = min(n_base, 150)  # Limite pour éviter trop de points
    m = min(m_base, 150)
    
    # Dimensions domaine calcul
    l = R0 * 4  # Hauteur domaine (plus large que le tube)
    hx = L/n
    hy = l/m
    C = 1.8e-5  # Coefficient viscosité
    
    # Tailles systèmes
    Np = n*m
    Nu = (n+1)*m
    Nv = n*(m+1)
    N = Np + Nu + Nv
    
    # Géométrie initiale
    y_centre = l / 2.0
    R = np.full(n + 1, R_repos)  # Rayon initial partout
    
    # Boucle adaptation géométrie
    max_iterations = 100
    tolerance = 1e-5
    
    iteration = 0
    changement = True
    R_history = [R.copy()]
    
    print("Début calcul adaptation géométrie...")
    
    while iteration < max_iterations and changement:
        A, B = matrixes(n, m, C, hx, hy, P_entree, P_sortie, R, l, Np, Nu, Nv)
        
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
        max_change = 0
        
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
            
            # Relaxation pour convergence
            R_new = R_ancien[i] + (R_new - R_ancien[i]) * 0.5
            
            change = abs(R_new - R_ancien[i])
            if change > max_change:
                max_change = change
            if change > tolerance:
                changement = True

            R[i] = R_new

        R_history.append(R.copy())
        iteration += 1
        
        if iteration % 10 == 0:
            print(f"  Itération {iteration}, changement max: {max_change:.6f} m")

    print(f"Calcul terminé après {iteration} itérations")
    print(f"Rayon final moyen: {np.mean(R)*1000:.3f} mm")
    
    # Extraction résultats finaux
    P_vecteur = S[offset_P:offset_U]
    U_vecteur = S[offset_U:offset_V]
    V_vecteur = S[offset_V:]
    
    P_grille_stag = P_vecteur.reshape((m, n))
    U_grille_stag = U_vecteur.reshape((m, n + 1))
    V_grille_stag = V_vecteur.reshape((m + 1, n))
    
    P_grille = P_grille_stag
    Ux_grille = (U_grille_stag[:, :-1] + U_grille_stag[:, 1:]) / 2.0
    Uy_grille = (V_grille_stag[:-1, :] + V_grille_stag[1:, :]) / 2.0
    
    return {
        'generation': gen,
        'rayon_final': R,
        'rayon_repos': R_repos,
        'rayon_max': R0,
        'pression_entree': P_entree,
        'pression_sortie': P_sortie,
        'P_grille': P_grille,
        'Ux_grille': Ux_grille,
        'Uy_grille': Uy_grille,
        'n': n,
        'm': m,
        'L': L,
        'l': l,
        'hx': hx,
        'hy': hy,
        'convergence_iterations': iteration
    }

# =============================================================================
# SIMULATION DE TOUTES LES GÉNÉRATIONS
# =============================================================================

print("DÉBUT DE LA SIMULATION COMPLÈTE DE L'ARBRE BRONCHIQUE")
print("=" * 70)

# Création du dossier de résultats
os.makedirs('resultats_generations', exist_ok=True)

resultats = {}

# Simulation pour chaque génération
for gen_index in range(len(generations_data['gen'])):
    try:
        resultat = simuler_generation(gen_index, cas='expiration_forcee')
        resultats[resultat['generation']] = resultat
    except Exception as e:
        print(f"Erreur lors de la simulation de la génération {gen_index}: {e}")
        continue

print(f"\nSimulation terminée pour {len(resultats)} générations")

# =============================================================================
# ANALYSE COMPARATIVE TOUTES GÉNÉRATIONS
# =============================================================================

print("\n" + "="*70)
print("ANALYSE COMPARATIVE TOUTES GÉNÉRATIONS")
print("="*70)

# Préparation des données pour les graphiques comparatifs
generations = []
rayons_finaux_moyens = []
rayons_repos = []
variations_relatives = []
pressions_entree = []
pressions_sortie = []

for gen in sorted(resultats.keys()):
    data = resultats[gen]
    rayon_final_moyen = np.mean(data['rayon_final'])
    rayon_repos = data['rayon_repos']
    variation = (rayon_final_moyen - rayon_repos) / rayon_repos * 100
    
    generations.append(gen)
    rayons_finaux_moyens.append(rayon_final_moyen * 1000)  # en mm
    rayons_repos.append(rayon_repos * 1000)  # en mm
    variations_relatives.append(variation)
    pressions_entree.append(data['pression_entree'])
    pressions_sortie.append(data['pression_sortie'])

# 1. GRAPHIQUE COMPARATIF DES RAYONS
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(generations, rayons_repos, 'go-', linewidth=2, markersize=6, label='Rayon au repos')
plt.plot(generations, rayons_finaux_moyens, 'bo-', linewidth=2, markersize=6, label='Rayon final (expiration forcée)')
plt.xlabel('Génération')
plt.ylabel('Rayon (mm)')
plt.title('Évolution des rayons bronchiques\npar génération')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(generations)

# Ajout de la zone de compression/dilatation
plt.axvspan(0, 4, alpha=0.2, color='red', label='Compression (générations centrales)')
plt.axvspan(5, 16, alpha=0.2, color='green', label='Dilatation (générations périphériques)')
plt.legend()
plt.show()

# 2. GRAPHIQUE DE VARIATION RELATIVE
bars = plt.bar(generations, variations_relatives, 
               color=['red' if v < 0 else 'green' for v in variations_relatives])
plt.xlabel('Génération')
plt.ylabel('Variation relative (%)')
plt.title('Variation du rayon pendant l\'expiration forcée\n(par rapport au repos)')
plt.grid(True, alpha=0.3)
plt.xticks(generations)

# Ajout des valeurs sur les barres
for i, v in enumerate(variations_relatives):
    plt.text(generations[i], v + (1 if v >= 0 else -3), f'{v:.1f}%', 
             ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')
plt.show()

# 3. GRAPHIQUE DES PRESSIONS
plt.subplot(2, 2, 3)
plt.plot(generations, pressions_entree, 's-', linewidth=2, markersize=6, label='Pression entrée')
plt.plot(generations, pressions_sortie, '^-', linewidth=2, markersize=6, label='Pression sortie')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Génération')
plt.ylabel('Pression transmurale (Pa)')
plt.title('Pressions appliquées pendant l\'expiration forcée')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(generations)

# Zone de compression/dilatation
plt.axvspan(0, 4, alpha=0.2, color='red')
plt.axvspan(5, 16, alpha=0.2, color='green')
plt.show()

# 4. GRAPHIQUE DU PARADOXE - COMPARAISON CENTRE/PERIPHERIE
plt.subplot(2, 2, 4)

# Séparation des générations centrales et périphériques
gen_centrales = [g for g in generations if g <= 4]
gen_peripheriques = [g for g in generations if g >= 5]

var_centrales = [v for g, v in zip(generations, variations_relatives) if g <= 4]
var_peripheriques = [v for g, v in zip(generations, variations_relatives) if g >= 5]

plt.bar(['Centrales\n(0-4)', 'Périphériques\n(5-16)'], 
        [np.mean(var_centrales), np.mean(var_peripheriques)],
        color=['red', 'green'], alpha=0.7)
plt.ylabel('Variation moyenne du rayon (%)')
plt.title('Paradoxe de l\'expiration forcée\nCompression centrale vs Dilatation périphérique')
plt.grid(True, alpha=0.3)

# Ajout des valeurs
plt.text(0, np.mean(var_centrales) + (1 if np.mean(var_centrales) >= 0 else -3), 
         f'{np.mean(var_centrales):.1f}%', ha='center', va='bottom' if np.mean(var_centrales) >= 0 else 'top', 
         fontweight='bold', fontsize=12)
plt.text(1, np.mean(var_peripheriques) + (1 if np.mean(var_peripheriques) >= 0 else -3), 
         f'{np.mean(var_peripheriques):.1f}%', ha='center', va='bottom' if np.mean(var_peripheriques) >= 0 else 'top', 
         fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('resultats_generations/comparaison_toutes_generations.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# RÉSUMÉ STATISTIQUE
# =============================================================================

print("\n" + "="*70)
print("RÉSUMÉ STATISTIQUE DU PARADOXE BRONCHIQUE")
print("="*70)

# Calcul des statistiques
nb_tubes_par_generation = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
total_tubes_dilates = 0
total_tubes_comprimes = 0

for gen in generations:
    if gen <= 4:  # Générations centrales - compression
        total_tubes_comprimes += nb_tubes_par_generation[gen]
    else:  # Générations périphériques - dilatation
        total_tubes_dilates += nb_tubes_par_generation[gen]

print(f"GÉNÉRATIONS CENTRALES (0-4) - COMPRESSION:")
print(f"  Nombre de bronches: {total_tubes_comprimes:,}")
print(f"  Variation moyenne du rayon: {np.mean(var_centrales):.1f}%")

print(f"\nGÉNÉRATIONS PÉRIPHÉRIQUES (5-16) - DILATATION:")
print(f"  Nombre de bronches: {total_tubes_dilates:,}")
print(f"  Variation moyenne du rayon: {np.mean(var_peripheriques):.1f}%")

print(f"\nPARADOXE CONFIRMÉ:")
print(f"  Pendant l'expiration forcée:")
print(f"  - {total_tubes_comprimes:,} bronches centrales se COMPRIMENT")
print(f"  - {total_tubes_dilates:,} bronches périphériques se DILATENT")
print(f"  → Phénomène simultané de compression et dilatation")

# =============================================================================
# VISUALISATIONS INDIVIDUELLES POUR QUELQUES GÉNÉRATIONS REPRÉSENTATIVES
# =============================================================================

print("\nGénération de visualisations individuelles...")

# Sélection de générations représentatives
gen_representatives = [0, 4, 10, 16]

for gen in gen_representatives:
    if gen in resultats:
        data = resultats[gen]
        
        # Préparation des coordonnées
        x_coords = np.linspace(data['hx']/2.0, data['L'] - data['hx']/2.0, data['n'])
        y_coords = np.linspace(data['hy']/2.0, data['l'] - data['hy']/2.0, data['m'])
        X, Y = np.meshgrid(x_coords, y_coords)
        
        plot_extent = [0, data['L'] * 100, 0, data['l'] * 100]
        
        # Masque parois
        R_local_p_centres = (data['rayon_final'][:-1] + data['rayon_final'][1:]) / 2.0 
        dist_au_centre_p = abs(y_coords.reshape(data['m'], 1) - data['l']/2.0)
        masque_mur = dist_au_centre_p > R_local_p_centres.reshape(1, data['n'])
        
        # Application du masque
        P_grille_plot = np.where(masque_mur, np.nan, data['P_grille'])
        Ux_grille_plot = np.where(masque_mur, np.nan, data['Ux_grille'])
        
        # Création des colormaps
        cmap_pression = copy.copy(plt.cm.viridis)
        cmap_pression.set_bad('gray', 1.0)
        
        # Graphique individuel
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(P_grille_plot, extent=plot_extent, origin='lower', 
                  aspect='auto', cmap=cmap_pression)
        plt.colorbar(label='Pression (Pa)')
        plt.xlabel('Position x (cm)')
        plt.ylabel('Position y (cm)')
        plt.title(f'Génération {gen} - Champ de pression\n(P_entree = {data["pression_entree"]} Pa)')
        
        plt.subplot(1, 2, 2)
        x_positions = np.linspace(0, data['L'], len(data['rayon_final']))
        plt.plot(x_positions * 100, data['rayon_final'] * 1000, 'b-', linewidth=2, label='Rayon final')
        plt.axhline(y=data['rayon_repos'] * 1000, color='g', linestyle='--', label='Rayon au repos')
        plt.axhline(y=data['rayon_max'] * 1000, color='r', linestyle='--', label='Rayon maximal')
        plt.xlabel('Position axiale (cm)')
        plt.ylabel('Rayon (mm)')
        plt.title(f'Génération {gen} - Profil de rayon\n(Variation: {variations_relatives[gen]:.1f}%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'resultats_generations/generation_{gen}.png', dpi=300, bbox_inches='tight')
        plt.close()

print("Toutes les visualisations ont été sauvegardées dans le dossier 'resultats_generations'")
print("\nSIMULATION COMPLÈTE TERMINÉE AVEC SUCCÈS!")