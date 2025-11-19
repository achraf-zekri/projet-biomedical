# PROJET DE SIMULATION D'ÉCOULEMENT DANS UN TUBE SOUPLE

## Description du Projet
Ce projet vise à simuler numériquement l'écoulement d'un fluide dans un tube souple. 
La simulation couple la mécanique des fluides avec la déformation de la paroi du tube, permettant 
d'étudier l'interaction fluide-structure.

## Structure des Codes
Le projet est constitué de 6 codes Python progressifs :

### 1. Code 1 - Dérivée seconde de f = C
- Résolution d'une EDP simple 1D : f''(x) = C
- Introduction aux méthodes aux différences finies
- Base mathématique pour les solveurs plus complexes

### 2. Code 2 - Laplacien de U = C  
- Extension 2D avec résolution de ΔU = C
- Mise en place des conditions aux limites
- Préparation pour la résolution d'équations de Navier-Stokes

### 3. Code 3 - Simulation d'un tube rigide
- Première simulation complète d'écoulement
- Résolution des équations de Stokes
- Géométrie rectangulaire fixe avec conditions de non-glissement

### 4. Code 4 - Géométrie variable
- **Version cylindrique** : Géométrie tubulaire convergente
- **Version aléatoire** : Géométrie complexe non-uniforme
- Adaptation des conditions aux limites à des formes variables
- Masquage des régions solides pour la visualisation

### 5. Code 5 - Cas souple (PRINCIPAL)
- Simulation couplée fluide-structure
- Loi comportementale élastique du tube avec deux branches :
  - Compression (P < P0)
  - Expansion (P > P0)
- Algorithme itératif de convergence géométrique
- Validation numérique de la loi de comportement

### 6. Code 6 - Simulation de toutes les branches
- Simulation complète de l'arbre bronchique (17 générations)
- Analyse comparative des générations
- Visualisation du paradoxe de l'expiration forcée

## Caractéristiques Techniques
- **Méthode** : Différences Finies 
- **Maillage** : Grille cartésienne avec maillage décalé (staggered)
- **Solveur** : Systèmes linéaires creux (scipy.sparse)
- **Visualisation** : Champs de pression, vitesse et déformation
- **Validation** : Vérification de la loi comportementale par régression

## Paramètres Physiques Modifiables
- Rayon initial et paramètres élastiques (R0, alpha0, P1_const, P2_const)
- Pressions d'entrée/sortie (P_entree, P_sortie) 
- Propriétés du fluide (viscosité C)
- Paramètres de convergence (tolerance, max_iterations)

## Utilisation
1. Exécuter le code 5 pour la simulation complète tube souple
2. Exécuter le code 6 pour la simulation de l'arbre bronchique complet
3. Les codes 1-4 sont des étapes préparatoires et démonstratives
4. Modifier les paramètres dans les sections dédiées en tête de fichier
5. Les résultats sont sauvegardés sous forme de graphiques

## Résultats Obtenus
- Champs de pression et vitesse dans le tube déformé
- Évolution de la géométrie pendant la convergence
- Profil final du rayon le long du tube
- Validation de la loi comportementale élastique
- Statistiques de convergence numérique
- Analyse du paradoxe bronchique (compression centrale et dilatation périphérique)

## Dépendances
- Python 3.x
- NumPy, Matplotlib, SciPy
- Bibliothèques scientifiques standard

## Notes
- Le code 5 implémente la méthode de relaxation pour assurer la convergence
- La visualisation inclut des masques pour distinguer fluide/paroi
- Des vérifications numériques validant le modèle physique sont incluses
- Adaptable à différentes lois comportementales et géométries

## Auteurs
- AIT ANIBA Mohamed
- ZEKRI Achraf  
- ENNAJI Soufiane
- FEDDA Mohammed
- ELHANAFI Hassan
- LEMKHANTAR Ilyass
- EL HOUSNI Khalid
- MOUNJI Houssam

## Date
11/11/2025