


from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

FICHIER_DONNEES = Path(__file__).with_name("annexe_TP3_donnees.json")

# ---------------------------------------------------------------------------
# 0. Constantes — ne pas modifier
# ---------------------------------------------------------------------------
SEED_POSE = 3031
LARGEUR_PX, HAUTEUR_PX = 640, 480
BRUIT_POSE_PX = 0.5
COTE_BASE = 100.0
SCENARIO_AMBIGU = (25.0, 1000.0, 60.0)       # angle (°), distance (mm), côté (mm)


def verifie(condition: bool, message: str) -> None:
    """Affiche ✓ ou lève une erreur explicite (utilisé par les étapes)."""
    if not condition:
        raise AssertionError(message)
    print(f"  \u2713 {message}")


def verifie_presque(valeur, attendu, tolerance: float, message: str) -> None:
    """Contrôle numérique avec tolérance et message lisible."""
    if not np.allclose(np.asarray(valeur, float), np.asarray(attendu, float),
                       atol=tolerance):
        raise AssertionError(
            f"{message} — obtenu {np.round(np.asarray(valeur), 5).tolist()}, "
            f"attendu {np.round(np.asarray(attendu), 5).tolist()} (tolérance {tolerance:g})")
    print(f"  \u2713 {message}")


# ---------------------------------------------------------------------------
# 1. Outils fournis — géométrie, projection, optimisation, Monte-Carlo
# ---------------------------------------------------------------------------
def antisym(v):
    """Matrice [v]x telle que [v]x w = v ^ w."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rodrigues(rvec):
    """Matrice de rotation d'un vecteur rotation (formule de Rodrigues)."""
    rvec = np.asarray(rvec, dtype=float)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    Kx = antisym(rvec / theta)
    return np.eye(3) + np.sin(theta) * Kx + (1.0 - np.cos(theta)) * (Kx @ Kx)


def rvec_depuis_R(R):
    """Vecteur rotation d'une matrice de rotation (inverse de rodrigues)."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(c))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-10:
        return 0.5 * v
    if np.pi - theta < 1e-6:                  # rotation d'angle pi
        M = (R + np.eye(3)) / 2.0
        axe = np.sqrt(np.maximum(np.diag(M), 0.0))
        k = int(np.argmax(axe))
        axe[(k + 1) % 3] = np.copysign(axe[(k + 1) % 3], M[k, (k + 1) % 3])
        axe[(k + 2) % 3] = np.copysign(axe[(k + 2) % 3], M[k, (k + 2) % 3])
        return theta * axe / np.linalg.norm(axe)
    return theta / (2.0 * np.sin(theta)) * v


def angle_rotation_deg(Ra, Rb):
    """Angle (en degrés) de la rotation relative Ra^T Rb."""
    c = np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def projeter(K, R, t, X):
    """Projette des points 3D (N,3) : renvoie les pixels (N,2) et les profondeurs."""
    Xc = (R @ np.asarray(X, float).T).T + t
    uvw = (K @ Xc.T).T
    return uvw[:, :2] / uvw[:, 2:3], Xc[:, 2]


def hartley(points):
    """Normalisation de Hartley : centre à l'origine, distance moyenne sqrt(2)."""
    points = np.asarray(points, float)[:, :2]
    c = points.mean(axis=0)
    d = np.mean(np.linalg.norm(points - c, axis=1))
    s = np.sqrt(2.0) / d
    T = np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])
    h = np.column_stack([points, np.ones(len(points))])
    return (T @ h.T).T[:, :2], T


def normaliser_F(F):
    """Fixe l'échelle (norme de Frobenius 1) et le signe (plus grand terme > 0)."""
    F = np.asarray(F, float) / np.linalg.norm(F)
    k = np.unravel_index(np.argmax(np.abs(F)), F.shape)
    return -F if F[k] < 0 else F


def homogene(x):
    x = np.asarray(x, float)
    return np.column_stack([x, np.ones(len(x))])


def noyau_droit(M):
    """Dernier vecteur singulier droit de M (vecteur du noyau approché)."""
    return np.linalg.svd(M)[2][-1]


def epipole(F):
    """Épipole de l'image 1 (noyau de F), en pixels ; None s'il est à l'infini."""
    e = noyau_droit(F)
    return e[:2] / e[2] if abs(e[2]) > 1e-9 * np.linalg.norm(e) else None


def epipole_2(F):
    """Épipole de l'image 2 (noyau de F^T), en pixels."""
    return epipole(F.T)


def ligne_epipolaire(F, x1):
    """Ligne (a, b, c) de l'image 2 associée au pixel x1 de l'image 1."""
    return F @ np.array([x1[0], x1[1], 1.0])


def residus_pose(p, K, X, uv):
    """Résidus de reprojection (2N) pour p = (rvec, t)."""
    R = rodrigues(p[:3])
    pred, _ = projeter(K, R, p[3:], X)
    return (pred - uv).ravel()


def erreur_reprojection(K, R, t, X, uv):
    """Erreur RMS de reprojection (px) : sqrt(moyenne des ||u_i - u_i^||^2)."""
    pred, _ = projeter(K, R, t, X)
    return float(np.sqrt(np.mean(np.sum((pred - uv) ** 2, axis=1))))


def raffiner_pose(R0, t0, K, X, uv, iterations=60):
    """Levenberg–Marquardt sur (rvec, t) : minimise l'erreur de reprojection.

    Jacobienne par différences finies centrées ; amortissement adaptatif.
    Renvoie (R, t, nombre d'itérations).
    """
    p = np.concatenate([rvec_depuis_R(R0), np.asarray(t0, float)])
    lam = 1e-3
    r = residus_pose(p, K, X, uv)
    cout = float(r @ r)
    for it in range(iterations):
        J = np.empty((len(r), 6))
        for k in range(6):
            h = 1e-6 * (1.0 + abs(p[k]))
            dp = np.zeros(6)
            dp[k] = h
            J[:, k] = (residus_pose(p + dp, K, X, uv)
                       - residus_pose(p - dp, K, X, uv)) / (2.0 * h)
        A, g = J.T @ J, J.T @ r
        succes = False
        for _ in range(12):
            try:
                pas = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            cand = p + pas
            rc = residus_pose(cand, K, X, uv)
            cc = float(rc @ rc)
            if np.isfinite(cc) and cc < cout:
                p, r, cout, lam = cand, rc, cc, max(lam / 5.0, 1e-12)
                succes = True
                break
            lam *= 5.0
        if not succes or np.linalg.norm(pas) < 1e-9 * (1.0 + np.linalg.norm(p)):
            break
    return rodrigues(p[:3]), p[3:].copy(), it + 1


def homographie_dlt(modele_xy, uv):
    """Homographie plan -> image par DLT normalisée (fournie ; >= 4 points)."""
    n1, T1 = hartley(modele_xy)
    n2, T2 = hartley(uv)
    lignes = []
    for (x, y), (u, v) in zip(n1, n2):
        lignes.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        lignes.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    H = noyau_droit(np.array(lignes)).reshape(3, 3)
    H = np.linalg.inv(T2) @ H @ T1
    return H / H[2, 2]


def marqueur_modele(cote):
    """Quatre coins du marqueur (plan Z = 0), ordre : haut-gauche, haut-droit,
    bas-droit, bas-gauche ; y vers le haut."""
    a = cote / 2.0
    return np.array([[-a, a, 0.0], [a, a, 0.0], [a, -a, 0.0], [-a, -a, 0.0]])


def cube_modele(cote):
    """Huit sommets du cube posé sur le marqueur (base Z=0, sommet Z=+côté)."""
    base = marqueur_modele(cote)
    return np.vstack([base, base + np.array([0.0, 0.0, cote])])


ARETES_CUBE = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
               (0, 4), (1, 5), (2, 6), (3, 7)]


def pose_vraie(angle_deg, distance):
    """Pose du marqueur : face à la caméra (R0 = diag(1,-1,-1)), inclinée de
    `angle_deg` autour de l'axe y de la caméra, avec un roulis de 20° et un
    décalage latéral ; t = distance * (0,03 ; -0,02 ; 1)."""
    R0 = np.diag([1.0, -1.0, -1.0])
    R = (rodrigues([0, 0, np.radians(20.0)])
         @ rodrigues([0, np.radians(angle_deg), 0]) @ R0)
    t = distance * np.array([0.03, -0.02, 1.0])
    return R, t


def angle_de_vue_deg(R, t):
    """Angle entre la normale du plan et la ligne de visée (0° = de face)."""
    n = R[:, 2]
    v = -t / np.linalg.norm(t)          # du marqueur vers la caméra
    return float(np.degrees(np.arccos(np.clip(abs(n @ v), 0.0, 1.0))))


def _normaliser_3d(X):
    """Normalisation de Hartley des points 3D (centre, distance moyenne sqrt(3)).

    Fournie pour le DLT-PnP : renvoie (points normalisés (N,3), T 4x4)."""
    X = np.asarray(X, float)
    c = X.mean(axis=0)
    d = np.mean(np.linalg.norm(X - c, axis=1))
    s = np.sqrt(3.0) / d
    T = np.eye(4)
    T[:3, :3] *= s
    T[:3, 3] = -s * c
    return (X - c) * s, T


def pose_marqueur(K, coins_uv, cote, raffiner=True):
    """Pose du marqueur à partir de ses quatre coins : H, puis LM facultatif."""
    modele = marqueur_modele(cote)
    H = homographie_dlt(modele[:, :2], coins_uv)
    R, t = pose_depuis_homographie(K, H)
    if raffiner:
        R, t, _ = raffiner_pose(R, t, K, modele, coins_uv)
    return R, t


def poses_alternatives(K, coins_uv, cote, R_init, t_init, graine=7, essais=40):
    """Minima locaux de l'erreur de reprojection d'un plan (ambiguïté du plan).

    Départs multiples (rotations aléatoires, t de la solution initiale).
    Renvoie une liste triée [(rms, R, t), ...] de minima distincts (> 3°).
    """
    modele = marqueur_modele(cote)
    rng = np.random.default_rng(graine)
    trouves = [(erreur_reprojection(K, R_init, t_init, modele, coins_uv),
                R_init, t_init)]
    for _ in range(essais):
        axe = rng.standard_normal(3)
        axe *= rng.uniform(0.2, 2.2) / np.linalg.norm(axe)
        R0 = rodrigues(axe) @ R_init
        R, t, _ = raffiner_pose(R0, t_init, K, modele, coins_uv)
        _, prof = projeter(K, R, t, modele)
        if np.any(prof <= 0):
            continue
        rms = erreur_reprojection(K, R, t, modele, coins_uv)
        if rms < 5.0 and all(angle_rotation_deg(R, Rk) > 3.0 for _, Rk, _ in trouves):
            trouves.append((rms, R, t))
    trouves.sort(key=lambda e: e[0])
    return trouves


# ---------------------------------------------------------------------------
# 2. Fonctions À COMPLÉTER
# ---------------------------------------------------------------------------
def imposer_rang2(F):
    """Plus proche matrice de rang 2 de F (au sens de Frobenius).

    Indice : SVD  F = U diag(s) V^T, puis annuler la plus petite valeur
    singulière s[2] et recomposer.
    """
    U, s, Vt = np.linalg.svd(np.asarray(F, dtype=float), full_matrices=True)
    s2 = np.diag(s)
    s2[-1, -1] = 0.0
    return U @ s2 @ Vt


def huit_points(x1, x2, normaliser=True, rang2=True):
    """Matrice fondamentale par l'algorithme des huit points (>= 8 couples).

    x1, x2 : pixels (N, 2) dans les images 1 et 2 ; contrainte x2^T F x1 = 0.
    Étapes (diapositives « huit points », « Hartley », « rang 2 ») :
      1. si `normaliser` : n1, T1 = hartley(x1) ; n2, T2 = hartley(x2)
         (sinon points bruts et T = I) ;
      2. une ligne de A par couple (u1, v1) <-> (u2, v2), dans l'ordre de F
         lue ligne par ligne :
             [u2 u1, u2 v1, u2, v2 u1, v2 v1, v2, u1, v1, 1]
      3. F = noyau_droit(A) rangé en 3x3 ;
      4. si `rang2` : F = imposer_rang2(F)   (AVANT la dénormalisation) ;
      5. dénormalisation F = T2^T F T1, puis normaliser_F(F) (fournie).
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if normaliser:
        n1, T1 = hartley(x1)
        n2, T2 = hartley(x2)
    else:
        n1, T1 = x1, np.eye(3)
        n2, T2 = x2, np.eye(3)

    A = np.empty((len(n1), 9), dtype=float)
    for i, ((u1, v1), (u2, v2)) in enumerate(zip(n1, n2)):
        A[i] = [u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1.0]

    F = noyau_droit(A).reshape(3, 3)
    if rang2:
        F = imposer_rang2(F)

    F = T2.T @ F @ T1
    return normaliser_F(F)


def distance_point_droite(F, x1, x2):
    """Distance (px) de chaque x2 à la ligne épipolaire l = F x1.

    x1, x2 : tableaux (N, 2) en pixels. Pour une ligne (a, b, c) et un point
    (u, v) : |a u + b v + c| / sqrt(a² + b²). Renvoie un tableau de N valeurs.
    Indice : `homogene(x1)` donne les coordonnées homogènes (N, 3).
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x1h = homogene(x1)
    lignes = (np.asarray(F, dtype=float) @ x1h.T).T
    num = np.sum(lignes * np.column_stack([x2[:, 0], x2[:, 1], np.ones(len(x2))]), axis=1)
    den = np.hypot(lignes[:, 0], lignes[:, 1])
    return np.abs(num) / np.where(den > 0, den, 1.0)


def dlt_pnp(X, uv, K):
    """Pose (R, t) par DLT à partir de n >= 6 points 3D généraux, K connue.

    X : (N, 3) points du modèle ; uv : (N, 2) pixels. Étapes :
      1. Xn, Tx = _normaliser_3d(X) ; un, Tu = hartley(uv) ;
      2. deux lignes de A (12 colonnes) par point, avec P_i = (x, y, z, 1) :
             [-P_i, 0, u P_i]   et   [0, -P_i, v P_i]
      3. P = noyau_droit(A) rangée en 3x4, puis dénormalisée
         P = Tu^-1 P Tx ;
      4. M = K^-1 P ; si det(M[:, :3]) < 0 : M = -M ;
      5. R = U V^T avec U, s, V^T = svd(M[:, :3]) (rotation la plus proche) ;
         t = M[:, 3] / moyenne(s)   (l'échelle est la moyenne des valeurs
         singulières).
    """
    X = np.asarray(X, dtype=float)
    uv = np.asarray(uv, dtype=float)
    Xn, Tx = _normaliser_3d(X)
    un, Tu = hartley(uv)
    A = np.empty((2 * len(Xn), 12), dtype=float)
    for i, P in enumerate(np.hstack([Xn, np.ones((len(Xn), 1))])):
        row_u = np.hstack([-P, np.zeros(4), un[i, 0] * P])
        row_v = np.hstack([np.zeros(4), -P, un[i, 1] * P])
        A[2 * i] = row_u
        A[2 * i + 1] = row_v

    P = noyau_droit(A).reshape(3, 4)
    P = np.linalg.inv(Tu) @ P @ Tx

    M = np.linalg.inv(K) @ P
    if np.linalg.det(M[:, :3]) < 0:
        M = -M

    U, s, Vt = np.linalg.svd(M[:, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    t = M[:, 3] / np.mean(s)
    return R, t


def pose_depuis_homographie(K, H):
    """Pose (R, t) d'un plan Z = 0 à partir de H = K [r1 r2 t] (à un facteur près).

    Étapes (diapositive « pose depuis l'homographie ») :
      1. M = K^-1 H ; facteur lam = 2 / (||M[:,0]|| + ||M[:,1]||) ;
         si lam * M[2, 2] < 0 : lam = -lam (le plan est devant la caméra) ;
      2. r1 = lam M[:,0] ; r2 = lam M[:,1] ; t = lam M[:,2] ;
      3. r3 = r1 x r2 (np.cross) ;
      4. ré-orthogonalisation : U, _, Vt = svd([r1 r2 r3]) ; R = U Vt.
    """
    M = np.linalg.inv(K) @ H
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    if lam * M[2, 2] < 0:
        lam = -lam

    r1 = lam * M[:, 0]
    r2 = lam * M[:, 1]
    t = lam * M[:, 2]
    r3 = np.cross(r1, r2)

    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R, t


# ---------------------------------------------------------------------------
# 3. Étapes de validation
# ---------------------------------------------------------------------------
def charger():
    with open(FICHIER_DONNEES, encoding="utf-8") as f:
        return json.load(f)


def etape_1_donnees(etat):
    d = charger()
    etat["d"] = d
    s, m, ref = d["stereo"], d["marqueur"], d["reference"]
    etat["K1"], etat["K2"] = np.array(s["K1"]), np.array(s["K2"])
    etat["x1"], etat["x2"] = np.array(s["x1_px"]), np.array(s["x2_px"])
    etat["idx8"] = np.array(s["indices_8_points"])
    etat["K"] = np.array(m["K"])
    etat["coins"] = np.array(m["coins_px"])
    etat["cube_obs"] = np.array(m["sommets_cube_px"])
    etat["ref"] = ref
    verifie(etat["x1"].shape == (40, 2), "40 couples de points stéréo chargés")
    verifie(etat["cube_obs"].shape == (8, 2), "4 coins et 8 sommets de cube chargés")
    print(f"    indices des 8 points d'estimation : {etat['idx8'].tolist()}")


def etape_2_rang2(etat):
    rng = np.random.default_rng(1)
    M = rng.standard_normal((3, 3))
    F2 = imposer_rang2(M)
    s = np.linalg.svd(M, compute_uv=False)
    verifie(abs(np.linalg.det(F2)) < 1e-12, "la matrice obtenue est de rang 2 (det = 0)")
    verifie_presque(np.linalg.norm(M - F2), s[2], 1e-9,
                    "distance de Frobenius à M = plus petite valeur singulière")


def etape_3_huit_points(etat):
    ref = etat["ref"]["stereo"]
    i, x1, x2 = etat["idx8"], etat["x1"], etat["x2"]
    F = huit_points(x1[i], x2[i])
    verifie(abs(np.linalg.det(F)) < 1e-12, "F est de rang 2 (det F = 0)")
    verifie_presque(F, ref["F_8pts"], 1e-6, "F (8 points, Hartley, rang 2) conforme")
    etat["F"] = F
    print(f"    F[2,2] = {F[2, 2]:.6f} ; F[2,0] = {F[2, 0]:.6f} ; F[2,1] = {F[2, 1]:.6f}")
    Fb = huit_points(x1[i], x2[i], normaliser=False, rang2=False)
    verifie_presque(Fb, ref["F_8pts"], 1.0, "la version brute s'exécute (comparer ses résidus à l'étape 4)")
    etat["F_brut"] = Fb
    E = etat["K2"].T @ F @ etat["K1"]
    sv = np.linalg.svd(E, compute_uv=False)
    print(f"    E = K'^T F K : valeurs singulières {np.round(sv, 3).tolist()} "
          f"(idéal : deux égales, une nulle)")


def etape_4_distance(etat):
    ref = etat["ref"]["stereo"]
    i = etat["idx8"]
    x1, x2 = etat["x1"], etat["x2"]
    reste = np.setdiff1d(np.arange(len(x1)), i)
    d = distance_point_droite(etat["F"], x1[reste], x2[reste])
    rms = float(np.sqrt(np.mean(d ** 2)))
    print(f"    RMS point–droite sur les 32 autres points : {rms:.3f} px (max {d.max():.2f} px)")
    verifie_presque(rms, ref["RMS_8pts_32_px"], 1e-3, "RMS point–droite (F à 8 points, Hartley)")
    db = distance_point_droite(etat["F_brut"], x1[reste], x2[reste])
    print(f"    sans normalisation ni rang 2 : RMS = {np.sqrt(np.mean(db ** 2)):.3f} px")
    Fv = np.array(ref["F_vraie"])
    dv = distance_point_droite(Fv, x1, x2)
    verifie_presque(np.sqrt(np.mean(dv ** 2)), ref["RMS_F_vraie_px"], 1e-3,
                    "RMS avec la F vraie = plancher dû au bruit")
    F40 = huit_points(x1, x2)
    d40 = distance_point_droite(F40, x1, x2)
    print(f"    avec les 40 points : RMS = {np.sqrt(np.mean(d40 ** 2)):.3f} px")
    verifie(np.sqrt(np.mean(d40 ** 2)) < np.sqrt(np.mean(d ** 2)),
            "40 points valent mieux que 8 (RMS plus faible)")


def etape_5_pnp(etat):
    ref = etat["ref"]["pose"]
    K, cube_obs = etat["K"], etat["cube_obs"]
    cube = cube_modele(COTE_BASE)
    R, t = dlt_pnp(cube, cube_obs, K)
    verifie(abs(np.linalg.det(R) - 1.0) < 1e-9 and np.allclose(R.T @ R, np.eye(3), atol=1e-9),
            "R est une rotation (RᵀR = I, det R = +1)")
    verifie_presque(R, ref["R_PnP"], 1e-5, "R du DLT-PnP conforme")
    verifie_presque(t, ref["t_PnP_mm"], 1e-3, "t du DLT-PnP conforme (mm)")
    Rv, tv = np.array(ref["R_vraie"]), np.array(ref["t_vrai_mm"])
    print(f"    erreur de rotation = {angle_rotation_deg(R, Rv):.3f}° ; "
          f"||Δt|| = {np.linalg.norm(t - tv):.2f} mm ; "
          f"RMS = {erreur_reprojection(K, R, t, cube, cube_obs):.3f} px")
    uv0, _ = projeter(K, Rv, tv, cube)
    R0, t0 = dlt_pnp(cube, uv0, K)
    verifie(angle_rotation_deg(R0, Rv) < 1e-6 and np.linalg.norm(t0 - tv) < 1e-6,
            "sans bruit, la pose vraie est retrouvée exactement")
    etat["pnp"] = (R, t)


def etape_6_homographie(etat):
    ref = etat["ref"]["pose"]
    K, coins = etat["K"], etat["coins"]
    modele = marqueur_modele(COTE_BASE)
    H = homographie_dlt(modele[:, :2], coins)
    verifie_presque(H, ref["H"], 1e-4, "homographie H fournie conforme")
    R, t = pose_depuis_homographie(K, H)
    verifie(abs(np.linalg.det(R) - 1.0) < 1e-9 and np.allclose(R.T @ R, np.eye(3), atol=1e-9),
            "R est une rotation après ré-orthogonalisation")
    verifie_presque(R, ref["R_homographie"], 1e-5, "R (homographie) conforme")
    verifie_presque(t, ref["t_homographie_mm"], 1e-3, "t (homographie) conforme (mm)")
    rms = erreur_reprojection(K, R, t, modele, coins)
    print(f"    RMS de reprojection (4 coins) : {rms:.3f} px")
    verifie(t[2] > 0, "le plan est devant la caméra (tz > 0)")
    etat["hom"] = (R, t)


def etape_7_raffinement(etat):
    ref = etat["ref"]["pose"]
    K, coins = etat["K"], etat["coins"]
    R, t = etat["hom"]
    modele = marqueur_modele(COTE_BASE)
    Rl, tl, nit = raffiner_pose(R, t, K, modele, coins)
    rms = erreur_reprojection(K, Rl, tl, modele, coins)
    Rv, tv = np.array(ref["R_vraie"]), np.array(ref["t_vrai_mm"])
    print(f"    LM ({nit} itérations) : RMS {erreur_reprojection(K, R, t, modele, coins):.3f} -> {rms:.3f} px ;"
          f" erreur de rotation {angle_rotation_deg(R, Rv):.2f}° -> {angle_rotation_deg(Rl, Rv):.2f}°")
    verifie_presque(rms, ref["rms_homographie_LM_px"], 1e-3, "RMS après raffinement conforme")
    verifie(rms < erreur_reprojection(K, R, t, modele, coins), "le raffinement diminue l'erreur de reprojection")
    # cube projeté
    cube_est, _ = projeter(K, Rl, tl, cube_modele(COTE_BASE))
    cube_vrai, _ = projeter(K, Rv, tv, cube_modele(COTE_BASE))
    ecart = np.linalg.norm(cube_est - cube_vrai, axis=1)
    print(f"    cube projeté : écart moyen au cube vrai = {ecart.mean():.2f} px (max {ecart.max():.2f} px)")
    etat["pose_lm"] = (Rl, tl)
    # ambiguïté du plan
    a_ang, a_dist, a_cote = SCENARIO_AMBIGU
    Ra, ta = pose_vraie(a_ang, a_dist)
    modele_a = marqueur_modele(a_cote)
    uv_a, _ = projeter(K, Ra, ta, modele_a)
    uv_a = uv_a + np.random.default_rng(SEED_POSE + 1).normal(0, BRUIT_POSE_PX, (4, 2))
    R1, t1 = pose_marqueur(K, uv_a, a_cote)
    mins = poses_alternatives(K, uv_a, a_cote, R1, t1)
    print(f"    ambiguïté du plan (marqueur de {a_cote:.0f} mm à {a_dist:.0f} mm) : "
          f"{len(mins)} minima de l'erreur de reprojection")
    for k, (r, Rk, _) in enumerate(mins[:2]):
        print(f"      minimum {k + 1} : RMS = {r:.3f} px, écart à la vraie rotation = "
              f"{angle_rotation_deg(Rk, Ra):.1f}°")
    verifie(len(mins) >= 2, "deux poses plausibles : ambiguïté du plan")
    verifie_presque([mins[0][0], mins[1][0]], [ref["ambiguite"][0]["rms_px"], ref["ambiguite"][1]["rms_px"]],
                    1e-3, "RMS des deux minima conformes")


ETAPES = (
    ("Étape 1 — charger les données (fourni)", etape_1_donnees),
    ("Étape 2 — rang 2 par SVD (atelier 1)", etape_2_rang2),
    ("Étape 3 — huit points avec Hartley (atelier 1)", etape_3_huit_points),
    ("Étape 4 — distance point–droite (atelier 1)", etape_4_distance),
    ("Étape 5 — DLT-PnP sur le cube (atelier 2)", etape_5_pnp),
    ("Étape 6 — pose depuis l'homographie (atelier 2)", etape_6_homographie),
    ("Étape 7 — raffinement LM, cube, ambiguïté (atelier 2, fourni)", etape_7_raffinement),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="TP3 - geometrie epipolaire et pose")
    parser.add_argument("--sortie", type=Path, default=None,
                        help="écrit une figure de contrôle (cube projeté)")
    args = parser.parse_args()

    restantes = 0
    etat: dict = {}
    for titre, fonction in ETAPES:
        print(f"\n=== {titre} ===")
        try:
            fonction(etat)
        except NotImplementedError as e:
            restantes += 1
            print(f"  \u2717 fonction à écrire : {e}")
        except AssertionError as e:
            restantes += 1
            print(f"  \u2717 contrôle échoué : {e}")
        except (KeyError, TypeError) as e:
            restantes += 1
            print(f"  \u2717 étape bloquée (une étape précédente n'est pas validée) : {type(e).__name__} {e}")
        except Exception as e:  # erreur de programmation dans une fonction à compléter
            restantes += 1
            print(f"  \u2717 erreur ({type(e).__name__}) : {e}")

    print("\n" + "=" * 66)
    if restantes == 0:
        print("Toutes les étapes sont validées : TP3 (ateliers 1 et 2) terminé.")
    else:
        print(f"{restantes} étape(s) restante(s) : complétez les fonctions "
              f"« À COMPLÉTER » puis relancez.")
    print("=" * 66)

    if args.sortie is not None and restantes == 0:
        ecrire_figure(args.sortie, etat)


def ecrire_figure(chemin, etat) -> None:
    """Figure de contrôle : cube projeté avec la pose estimée (optionnelle)."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib absent : figure non écrite.")
        return
    K = etat["K"]
    R, t = etat["pose_lm"]
    cube = cube_modele(COTE_BASE)
    pts, _ = projeter(K, R, t, cube)
    fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=200)
    ax.add_patch(plt.Rectangle((0, 0), LARGEUR_PX, HAUTEUR_PX, fc="#F7F8FA", ec="#58595B", lw=0.8))
    for i, j in ARETES_CUBE:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color="#1F4E79", lw=1.4)
    ax.plot(etat["coins"][:, 0], etat["coins"][:, 1], "o", color="#B3261E", ms=4,
            label="coins observés")
    ax.set_xlim(0, LARGEUR_PX)
    ax.set_ylim(HAUTEUR_PX, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.set_title("Cube projeté avec la pose estimée", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(chemin)
    print(f"\nFigure écrite : {chemin}")


if __name__ == "__main__":
    main()