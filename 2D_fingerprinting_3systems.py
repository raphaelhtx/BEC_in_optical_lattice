# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 08:54:18 2026

@author: Raphael
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: 2D_fingerprinting_3systems.py
# Author: Raphaël Heurtaux
# Description: Algorithme GRAPE appliqué à 3 systèmes BEC pour implémenter la méthode de fingerprinting dans un réseau bidimensionnel. Code optimisé pour GPU (nécessite CUDA)


import numpy as np
import cupy as cp
from scipy.optimize import minimize
from time import perf_counter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],'size':20})
rc('text', usetex=True)

import os
nom_fichier = os.path.splitext(os.path.basename(__file__))[0]
#----------------------------------------------------------------------------------
#
# Système
#
#----------------------------------------------------------------------------------


class BEC_2D:
    """
    Définit un système de BEC pour un réseau optique 2D.
    """
    def __init__(self, mmax, nmax, q, s, lamb, theta):
        self.mmax = mmax
        self.nmax = nmax
        self.q = q
        self.s = float(s)
        self.lamb = float(lamb)
        self.theta = theta
        N = 2 * nmax + 1
        self.dim = (2 * mmax + 1) * N

        # m et n vectorisés
        m_np = np.arange(-mmax, mmax + 1)
        n_np = np.arange(-nmax, nmax + 1)
        mg, ng = np.meshgrid(m_np, n_np, indexing='ij')
        self._m  = mg.ravel()
        self._n  = ng.ravel()
        self._k  = (self._m + mmax) * N + (self._n + nmax)
        self._m_gpu = cp.asarray(self._m, dtype=cp.float64)
        self._n_gpu = cp.asarray(self._n, dtype=cp.float64)

        Hl = self._build_Hl()  # (6, dim, dim)
        self.D, self.P = cp.linalg.eigh(Hl) #(6, dim) et (6, dim, dim)
        self.Pt = self.P.conj().transpose(0, 2, 1).copy()

        self._H0_cache = {}

    def get_k(self, m, n):
        """
        Passage d'un état (m,n) à l'indice k correspondant.
        """
        return int((m + self.mmax) * (2 * self.nmax + 1) + (n + self.nmax))
    

    def _build_Hl(self):
        """
        Passage d'un état (m,n) à l'indice k correspondant.
        """
        dim = self.dim
        N  = 2 * self.nmax + 1
        val = np.float64(-0.25 * self.s)
        m, n, k = self._m, self._n, self._k

        def _mat(mask, offset):
            H = np.zeros((dim, dim), dtype=np.complex128)
            H[k[mask], k[mask] + offset] = val
            return H

        H12 = _mat(n < self.nmax,    +1)
        H21 = _mat(n > -self.nmax,   -1)
        H31 = _mat(m < self.mmax,    +N)
        H13 = _mat(m > -self.mmax,   -N)
        H32 = _mat((m < self.mmax) & (n < self.nmax),    N + 1)
        H23 = _mat((m > -self.mmax) & (n > -self.nmax), -N - 1)

        Hl_np = np.stack([
            H12 + H21,
            1j * (H12 - H21),
            H23 + H32,
            1j * (H23 - H32),
            H31 + H13,
            1j * (H31 - H13),
        ]).astype(np.complex128)

        return cp.asarray(Hl_np)

    def get_H0(self, t):
        """
        Passage d'un état (m,n) à l'indice k correspondant.
        """
        if t not in self._H0_cache:
            lambx = np.float64(self.lamb * np.cos(self.theta))
            lamby = np.float64(self.lamb * np.sin(self.theta))
            m_moins_n = self._m_gpu - self._n_gpu   # m−n vectorisé
            m_plus_n = self._m_gpu + self._n_gpu
            tf = np.float64(t)
            H0 = (np.float64(1.5)             * m_moins_n - lambx * tf) ** 2 + \
                 (np.float64(np.sqrt(3.) / 2) * m_plus_n - lamby * tf) ** 2
            self._H0_cache[t] = H0  
        return self._H0_cache[t]
    

    def step_fwd(self, psi, t, dt, u12, u23, u31):
        """
        Passage d'un état (m,n) à l'indice k correspondant.
        """
        # U0
        psi = psi * cp.exp(np.complex128(-1j * dt) * self.get_H0(t))

        # U12, U21, U23, U32, U31, U13 à la suite
        ctrls = (np.cos(u12), np.sin(u12),
                 np.cos(u23), np.sin(u23),
                 np.cos(u31), np.sin(u31))
        c_list, ph_list = [], []
        for i, ctrl in enumerate(ctrls):
            ph = cp.exp(np.complex128(-1j * dt * ctrl) * self.D[i])
            c  = self.Pt[i] @ psi
            psi = self.P[i] @ (ph * c)
            c_list.append(c)
            ph_list.append(ph)

        return psi, c_list, ph_list


    def step_adj(self, d, t, dt, u12, u23, u31):
        """
        Propagation arrière en utilisant le propagateur approximé U = U0@U12@U21@U23@U32@U31@U13.
        On stocke et renvoie tout dans le cache du Propagateur car on fait la même propagation avant pour le calcul du gradient PMP.
        """
        # Ordre inversé car on applique U'.d = U
        ctrls_rev = (np.sin(u31), np.cos(u31),
                     np.sin(u23), np.cos(u23),
                     np.sin(u12), np.cos(u12))
        idxs_rev  = (5, 4, 3, 2, 1, 0)

        cw_list = []
        for ctrl, i in zip(ctrls_rev, idxs_rev):
            ph_dag = cp.exp(np.complex128(1j * dt * ctrl) * self.D[i])
            cw = self.Pt[i] @ d
            d  = self.P[i] @ (ph_dag * cw)
            cw_list.append(cw)

        # U0†
        d = d * cp.exp(np.complex128(1j * dt) * self.get_H0(t))
        return d, cw_list


    def grad_step(self, c_list, ph_list, cw_list, dt, u12, u23, u31):
        """
        Calcul vectoriel du gradient au pas de temps correspondant.
        """
        D = self.D
        sin12, cos12 = np.sin(u12), np.cos(u12)
        sin23, cos23 = np.sin(u23), np.cos(u23)
        sin31, cos31 = np.sin(u31), np.cos(u31)

        def inner(cw, i, dfact):
            return cp.dot(cw.conj(),
                          np.complex128(dfact) * D[i] * ph_list[i] * c_list[i])

        g0 = inner(cw_list[5], 0,  1j * sin12 * dt)
        g1 = inner(cw_list[4], 1, -1j * cos12 * dt)
        g2 = inner(cw_list[3], 2,  1j * sin23 * dt)
        g3 = inner(cw_list[2], 3, -1j * cos23 * dt)
        g4 = inner(cw_list[1], 4,  1j * sin31 * dt)
        g5 = inner(cw_list[0], 5, -1j * cos31 * dt)

        # Un seul transfert (couteux en temps) du GPU vers le CPU
        gs = cp.real(cp.stack([g0, g1, g2, g3, g4, g5])).get()
        return (2.0 * (gs[0] + gs[1]),
                2.0 * (gs[2] + gs[3]),
                2.0 * (gs[4] + gs[5]))


#----------------------------------------------------------------------------------
#
# Dynamique
#
#----------------------------------------------------------------------------------


class Propagation_2D:
    """
    Pour un système donné, calcule sa propagation temporelle,
    et calcule le gradient de l'Hamiltonien de Pontryagin.
    """

    def __init__(self, time_grid):
        self.t  = time_grid
        self.dt = float(time_grid[1] - time_grid[0])
        self.Nt = len(time_grid)

    def forward(self, sys, psi0, u):
        """
        Calcule le l'ensemble des fonctions d'onde à chaque pas de temps.
        On propage, par pas de temps, en utilisant la popagateur exact.
        """
        Nt, dt = self.Nt, self.dt
        states = cp.zeros((sys.dim, Nt), dtype=cp.complex128)
        states[:, 0] = psi0
        all_c, all_ph = [], []
        psi = psi0.copy()

        for n in range(Nt - 1):
            u12 = float(u[n])
            u23 = float(u[n + Nt])
            u31 = float(u[n + 2*Nt])
            psi, c, ph = sys.step_fwd(psi, self.t[n], dt, u12, u23, u31)
            states[:, n + 1] = psi
            all_c.append(c)
            all_ph.append(ph)

        return states, all_c, all_ph    

    def compute_gradient(self, sys, states, psi_tgt, u, all_c, all_ph):
        """
        Calcul du gradient d'un système donné.
        Pour chaque pas temps, on calcule l'adjoint et la contribution au gradient.
        L'ensemble des valeurs prises par la fonction d'onde pour chaque pas de temps
        est entièrement stocké dans states au préalable, car nécessaire pour définir l'ajoint.
        Complexité en O(Nt x dim²)  (comme forward).
        """
        Nt, dt = self.Nt, self.dt
        dF = np.zeros(3 * Nt)
        d = (-cp.vdot(psi_tgt, states[:, -1]) * psi_tgt).astype(cp.complex128)

        for n in range(Nt - 2, -1, -1):
            u12 = float(u[n])
            u23 = float(u[n + Nt])
            u31 = float(u[n + 2*Nt])

            # Propagation de l'adjoint
            d, cw = sys.step_adj(d, self.t[n], dt, u12, u23, u31)

            # Contributions au gradient (O(dim))
            dF[n], dF[n + Nt], dF[n + 2*Nt] = sys.grad_step(
                all_c[n], all_ph[n], cw, dt, u12, u23, u31)

        # Dernières valeurs
        dF[Nt - 1] = dF[2*Nt - 1] = dF[-1] = 0.0
        return dF


#----------------------------------------------------------------------------------
#
# Coûts et gradients pour l'optimisation
#
#----------------------------------------------------------------------------------

class OptimalControlProblem:
    """
    Implémente l'algorithme GRAPE.
    Accepte plusieurs systèmes : le coût/gradient total est la moyenne des coûts/gradients de chaque système.
    A l'inverse, les 3 contrôles sont stockées à la suite : u = [u12, u23, u31].
    """

    def __init__(self, time_grid: np.ndarray):
        self.prop = Propagation_2D(time_grid)
        self.systems: list = []
        self.iter_count = 0
        # Cache du dernier forward
        self._cached_u_cpu = None
        self._cached_u_gpu = None
        self._cached_fwd = None

    def add_system(self, sys, psi0_gpu, psi_tgt_gpu):
        self.systems.append((sys, psi0_gpu, psi_tgt_gpu))

    # Cache 
    def check_cache(self, u_cpu):
        """Lance le forward seulement si u a changé depuis le dernier appel"""
        if (self._cached_u_cpu is None or
                not np.array_equal(u_cpu, self._cached_u_cpu)):
            u_gpu = cp.asarray(u_cpu)
            fwd = []
            for sys, psi0, _ in self.systems:
                states, ac, aph = self.prop.forward(sys, psi0, u_gpu)
                fwd.append((states, ac, aph))
            self._cached_u_cpu     = u_cpu.copy()
            self._cached_u_gpu = u_gpu
            self._cached_fwd   = fwd

    def total_cost(self, u_cpu):
        self.check_cache(u_cpu)
        cost = 0.0
        for i, (_, _, psit) in enumerate(self.systems):
            psi_f = self._cached_fwd[i][0][:, -1]
            cost += 1.0 - float(abs(cp.vdot(psit, psi_f)) ** 2)
        return cost / len(self.systems)

    def total_gradient(self, u_cpu):
        self.check_cache(u_cpu)
        u_gpu = self._cached_u_gpu
        Nt    = self.prop.Nt
        grad  = np.zeros(3 * Nt)
        for i, (sys, _, psit) in enumerate(self.systems):
            states, ac, aph = self._cached_fwd[i]
            g = self.prop.compute_gradient(sys, states, psit, u_gpu, ac, aph)
            grad += g
        self._cached_u_cpu = None
        return grad / len(self.systems)

    def callback(self, xk_cpu):
        J = self.total_cost(xk_cpu)
        print(f"  Iter {self.iter_count:4d} | fidelité = {1 - J:.10f}")
        self.iter_count += 1

        if 1-J>0.9999:
            raise StopIteration




#----------------------------------------------------------------------------------
#
# Constantes et temps
#
#----------------------------------------------------------------------------------

start = perf_counter()

# Constantes
mrb   = 86.909180527*1.66054e-27   # Masse d'un atome de Rb-87
wl    = 1064e-9                    # Longueur d'onde du laser
d     = wl/2.0                     # Période spatiale
hbar  = 1.0545718e-34              # Constante de Planck réduite
kL   = (2.0*np.pi)/d              # Vecteur d'onde du réseau
EL   = (hbar*kL)**2/(2.0*mrb)    # Énergie du réseau
nu_L  = EL/(2.0*np.pi*hbar)       # Fréquence du réseau

# Grille temporelle
Nt    = 400
thold = 117   # µs
tf    = 3 * thold * 1e-6 * EL / hbar
t_cpu = np.linspace(0, tf, Nt)
t = t_cpu
time_us = (t * hbar / (3*EL)) * 1.0e6
time_s = (t * hbar / (3*EL))

#----------------------------------------------------------------------------------
#
# Paramètres du réseau
#
#----------------------------------------------------------------------------------
mmax, nmax = 10, 10
q = 0.
s = 15. / 3
d = (2 * mmax + 1) * (2 * nmax + 1)

print(f"Initialisation des systèmes BEC (dim={d})")
t0 = perf_counter()
delta = 1e-3
theta0 = np.pi/3
theta_min = np.pi/2 - theta0
theta_max = np.pi/2 + theta0
sys1 = BEC_2D(mmax, nmax, q, s, lamb=0, theta=0)
sys2 = BEC_2D(mmax, nmax, q, s, lamb=delta, theta=theta_min)
sys3 = BEC_2D(mmax, nmax, q, s, lamb=delta, theta=theta_max)
print(f"  Hamiltoniens diagonalisés en {perf_counter()-t0:.2f} s")



# États gpu
psi0_gpu  = cp.zeros(d, dtype=cp.complex128); psi0_gpu[sys1.get_k(0, 0)] = 1

psit1_gpu = cp.zeros(d, dtype=cp.complex128); psit1_gpu[sys1.get_k( 0, 0)] = 1
psit2_gpu = cp.zeros(d, dtype=cp.complex128); psit2_gpu[sys2.get_k( 1, 0)] = 1
psit3_gpu = cp.zeros(d, dtype=cp.complex128); psit3_gpu[sys3.get_k( 0, 1)] = 1


# Problème de contrôle optimal
problem = OptimalControlProblem(t_cpu)
problem.add_system(sys1, psi0_gpu, psit1_gpu)
problem.add_system(sys2, psi0_gpu, psit2_gpu)
problem.add_system(sys3, psi0_gpu, psit3_gpu)

#----------------------------------------------------------------------------------
#
# Contrôle initial
#
#----------------------------------------------------------------------------------
# Choix 1 : somme de cosinus : contrôle initial régulier.
# u0_cpu = np.zeros(3*Nt)
# for k in range(200):
#     a_k = np.random.uniform(-1, 1)
#     f_k = np.random.uniform(0, 1)
#     phi_k = np.random.uniform(0, 2*np.pi)

#     u0_cpu[:Nt] += a_k * np.cos(2*np.pi*f_k*t + phi_k)
#     u0_cpu[Nt:2*Nt] += a_k * np.cos(2*np.pi*f_k*t + phi_k)
#     u0_cpu[2*Nt:3*Nt] += a_k * np.cos(2*np.pi*f_k*t + phi_k)


# Choix 2 : u_0 complètement aléatoire. Fréquences jusqu'au MHz. Donne de meilleurs résultats.
np.random.seed(0)
u0_cpu = 2*np.pi * np.random.randn(3*Nt)

u0_cpu = u0_cpu % 2*np.pi - np.pi
u0_12 = u0_cpu[:Nt]
u0_23 = u0_cpu[Nt:2*Nt]
u0_31 = u0_cpu[2*Nt:3*Nt]

#----------------------------------------------------------------------------------
#
# Optimisation L-BFGS-B
#
#----------------------------------------------------------------------------------
new_optimisation = False
path = f"./data/optimal_controls/{nom_fichier}_{thold}us_s_{s}.npz" # A ADAPTER SELON LE DOSSIER DE TRAVAIL
if new_optimisation:
    # Optimisation L-BFGS-B
    print("\nDébut de l'optimisation L-BFGS-B...")
    sol = minimize(
        problem.total_cost,
        u0_cpu,
        jac=problem.total_gradient,
        method='L-BFGS-B',
        callback=problem.callback,
        options={'maxiter': 10000},
    )

    print(f"\nStatut : {sol.message}")

    uopt_cpu = sol.x
    uopt = uopt_cpu

    # Sauvegarde
    np.savez(path, uopt=uopt_cpu)

else:
    # Récupération de la dernière sauvegarde
    results = np.load(path)
    uopt_cpu = results["uopt"]
    uopt = uopt_cpu
    uopt_gpu = cp.asarray(uopt)


#----------------------------------------------------------------------------------
#
# Résultats
#
#----------------------------------------------------------------------------------
prop = Propagation_2D(t_cpu)
uopt_gpu = cp.asarray(uopt_cpu)

Psi1_gpu, _, _ = prop.forward(sys1, psi0_gpu, uopt_gpu)
Psi2_gpu, _, _ = prop.forward(sys2, psi0_gpu, uopt_gpu)
Psi3_gpu, _, _ = prop.forward(sys3, psi0_gpu, uopt_gpu)

pop1 = float(abs(cp.vdot(psit1_gpu, Psi1_gpu[:, -1])) ** 2)
pop2 = float(abs(cp.vdot(psit2_gpu, Psi2_gpu[:, -1])) ** 2)
pop3 = float(abs(cp.vdot(psit3_gpu, Psi3_gpu[:, -1])) ** 2)

print(f"Population sys1 dans |1,0⟩ : {pop1:.8f}")
print(f"Population sys2 dans |0,1⟩ : {pop2:.8f}")
print(f"Population sys3 dans |-1,0⟩ : {pop3:.8f}")
print(f"Fidélité moyenne              : {(pop1+pop2+pop3)/3:.8f}")
print(f"\nTemps total écoulé : {perf_counter()-start:.2f} s")

# Visualisation des contrôles optimaux
Psi1 = Psi1_gpu.get()
Psi2 = Psi2_gpu.get()
Psi3 = Psi3_gpu.get()


u12 = uopt_cpu[:Nt] % 2*np.pi -np.pi
u23 = uopt_cpu[Nt:2*Nt] % 2*np.pi -np.pi
u31 = uopt_cpu[2*Nt:3*Nt] % 2*np.pi -np.pi

m_vals = np.arange(-mmax, mmax + 1)
n_vals = np.arange(-nmax, nmax + 1)













#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Plots
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------
#
# Calcul des résultats des simulations une et une seule fois
#
#----------------------------------------------------------------------------------
new_plots = False
N_theta = 50
N_lambda = 40
lambda_tab = np.linspace(0, delta, N_lambda)
# Choix :
path = f"./data/plots/{nom_fichier}_{thold}us_s_{s}_0_2pi.npz"
theta_tab = np.linspace(0, 2*np.pi, N_theta) # pour tout theta
# path = f"./data/plots/{nom_fichier}_{thold}us_s_{s}_pi6_5pi6.npz"
# theta_tab = np.linspace(theta_min, theta_max, N_theta) # pour theta dans l'ouverture angulaire

N_lambda_geo = 7
N_theta_geo = 11
lambda_tab_geo = np.linspace(0, delta, 7)
theta_tab_geo = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 5*np.pi/6, np.pi, -5*np.pi/6, -3*np.pi/4, -np.pi/2, -np.pi/6])

Theta, Lambda = np.meshgrid(theta_tab, lambda_tab)
if new_plots:

    P1_theta = np.zeros((N_lambda, N_theta_geo))
    P2_theta = np.zeros((N_lambda, N_theta_geo))
    P3_theta = np.zeros((N_lambda, N_theta_geo))

    for i, lamb in enumerate(lambda_tab):
        for j, theta in enumerate(theta_tab_geo):

            sysl = BEC_2D(mmax, nmax, q, s, lamb=lamb, theta=theta)
            Psil_gpu, _, _ = prop.forward(sysl, psi0_gpu, uopt_gpu)

            popl1 = abs(cp.vdot(psit1_gpu, Psil_gpu[:,-1]))**2
            popl2 = abs(cp.vdot(psit2_gpu, Psil_gpu[:,-1]))**2
            popl3 = abs(cp.vdot(psit3_gpu, Psil_gpu[:,-1]))**2

            P1_theta[i,j] = popl1
            P2_theta[i,j] = popl2
            P3_theta[i,j] = popl3

    P1_lambda = np.zeros((N_lambda_geo, N_theta))
    P2_lambda = np.zeros((N_lambda_geo, N_theta))
    P3_lambda = np.zeros((N_lambda_geo, N_theta))

    for i, lamb in enumerate(lambda_tab_geo):
        for j, theta in enumerate(theta_tab):

            sysl = BEC_2D(mmax, nmax, q, s, lamb=lamb, theta=theta)
            Psil_gpu, _, _ = prop.forward(sysl, psi0_gpu, uopt_gpu)

            popl1 = abs(cp.vdot(psit1_gpu, Psil_gpu[:,-1]))**2
            popl2 = abs(cp.vdot(psit2_gpu, Psil_gpu[:,-1]))**2
            popl3 = abs(cp.vdot(psit3_gpu, Psil_gpu[:,-1]))**2

            P1_lambda[i,j] = popl1
            P2_lambda[i,j] = popl2
            P3_lambda[i,j] = popl3
    

    P1 = np.zeros((N_lambda, N_theta))
    P2 = np.zeros((N_lambda, N_theta))
    P3 = np.zeros((N_lambda, N_theta))

    for i, lamb in enumerate(lambda_tab):
        print(f"lambda = {lamb*1000:.3f}*10^(-3)")
        for j, theta in enumerate(theta_tab):

            sysl = BEC_2D(mmax, nmax, q, s, lamb=lamb, theta=theta)
            Psil_gpu, _, _ = prop.forward(sysl, psi0_gpu, uopt_gpu)

            popl1 = abs(cp.vdot(psit1_gpu, Psil_gpu[:,-1]))**2
            popl2 = abs(cp.vdot(psit2_gpu, Psil_gpu[:,-1]))**2
            popl3 = abs(cp.vdot(psit3_gpu, Psil_gpu[:,-1]))**2

            P1[i,j] = popl1
            P2[i,j] = popl2
            P3[i,j] = popl3

    np.savez(path, Theta=Theta, Lambda=Lambda, P1=P1, P2=P2, P3=P3, P1_theta=P1_theta, P2_theta=P2_theta, P3_theta=P3_theta, P1_lambda=P1_lambda, P2_lambda=P2_lambda, P3_lambda=P3_lambda)

else:
    results = np.load(path)
    P1 = results["P1"]
    P2 = results["P2"]
    P3 = results["P3"]

    P1_theta = results["P1_theta"]
    P2_theta = results["P2_theta"]
    P3_theta = results["P3_theta"]

    P1_lambda = results["P1_lambda"]
    P2_lambda = results["P2_lambda"]
    P3_lambda = results["P3_lambda"]








#----------------------------------------------------------------------------------
#
# Populations finales en fonction de theta, à lambda fixé
#
#----------------------------------------------------------------------------------

p1_tab = np.zeros(N_theta)
p2_tab = np.zeros(N_theta)
p3_tab = np.zeros(N_theta)
p123_tab = np.zeros(N_theta)
x1_tab = np.zeros(N_theta)
x2_tab = np.zeros(N_theta)
x3_tab = np.zeros(N_theta)
x123_tab = np.zeros(N_theta)

for i, lamb in enumerate(lambda_tab_geo):
    for j, theta in enumerate(theta_tab):

        alpha = np.pi/2 * lamb/delta * np.cos(theta) / np.sin(theta0)
        beta = np.pi/4 *(1 - lamb/delta * np.cos(theta) / np.sin(theta0))

        x1 = np.cos(alpha)
        x2 = np.sin(alpha) * np.cos(beta)
        x3 = np.sin(alpha) * np.sin(beta)
        
        popl1 = P1_lambda[i,j]
        popl2 = P2_lambda[i,j]
        popl3 = P3_lambda[i,j]

        p1_tab[j] = popl1
        p2_tab[j] = popl2
        p3_tab[j] = popl3
        p123_tab[j] = popl1 + popl2 + popl3

        x1_tab[j] = x1*x1
        x2_tab[j] = x2*x2
        x3_tab[j] = x3*x3
    
        x123_tab[j] = x1*x1 + x2*x2 + x3*x3
        

    plt.figure(figsize=(8, 3))
    plt.plot(theta_tab, p1_tab, marker='o', linestyle='-', color='C1')
    plt.plot(theta_tab, p2_tab, marker='o', linestyle='-', color='C2')
    plt.plot(theta_tab, p3_tab, marker='o', linestyle='-', color='C3')
    plt.plot(theta_tab, p123_tab, marker='o', linestyle='-', color='C5')


    plt.plot(theta_tab, x1_tab, marker='o', linestyle='-.', color='C1')
    plt.plot(theta_tab, x2_tab, marker='o', linestyle='-.', color='C2')
    plt.plot(theta_tab, x3_tab, marker='o', linestyle=':', color='C3')
    plt.plot(theta_tab, x123_tab, marker='o', linestyle=':', color='C5')

    plt.xlabel(r'$\theta$', fontsize=14)
    plt.ylabel(r'Populations', fontsize=14)
    plt.title(fr"$\lambda=${lamb}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()



#----------------------------------------------------------------------------------
#
# Populations finales en fonction de lambda, à theta fixé
#
#----------------------------------------------------------------------------------

p1_tab = np.zeros(N_lambda)
p2_tab = np.zeros(N_lambda)
p3_tab = np.zeros(N_lambda)
p123_tab = np.zeros(N_lambda)
x1_tab = np.zeros(N_lambda)
x2_tab = np.zeros(N_lambda)
x3_tab = np.zeros(N_lambda)
x123_tab = np.zeros(N_theta)

for j, theta in enumerate(theta_tab_geo):
    for i, lamb in enumerate(lambda_tab):

        alpha = np.pi/2 * lamb/delta * np.cos(theta) / np.sin(theta0)
        beta = np.pi/4 *(1 - lamb/delta * np.cos(theta) / np.sin(theta0))

        x1 = np.cos(alpha)
        x2 = np.sin(alpha) * np.cos(beta)
        x3 = np.sin(alpha) * np.sin(beta)

        popl1 = P1_theta[i,j]
        popl2 = P2_theta[i,j]
        popl3 = P3_theta[i,j]

        p1_tab[i] = popl1
        p2_tab[i] = popl2
        p3_tab[i] = popl3
        p123_tab[i] = popl1 + popl2 + popl3

        x1_tab[i] = x1*x1
        x2_tab[i] = x2*x2
        x3_tab[i] = x3*x3
        

    plt.figure(figsize=(8, 3))
    plt.plot(lambda_tab, p1_tab, marker='o', linestyle='-', color='C1')
    plt.plot(lambda_tab, p2_tab, marker='o', linestyle='-', color='C2')
    plt.plot(lambda_tab, p3_tab, marker='o', linestyle='-', color='C3')
    plt.plot(lambda_tab, p123_tab, marker='o', linestyle='-', color='C5')

    plt.plot(lambda_tab, x1_tab, marker='o', linestyle=':', color='C1')
    plt.plot(lambda_tab, x2_tab, marker='o', linestyle=':', color='C2')
    plt.plot(lambda_tab, x3_tab, marker='o', linestyle=':', color='C3')

    plt.xlabel(r'$\lambda$', fontsize=14)
    plt.ylabel(r'Populations', fontsize=14)
    plt.title(fr"$\theta=${theta}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()




#----------------------------------------------------------------------------------
#
# Contour plot : Populations
#
#----------------------------------------------------------------------------------
X = Lambda * np.cos(Theta)
Y = Lambda * np.sin(Theta)

plt.figure(figsize=(5,4))
plt.contourf(X, Y, P1, levels=100, cmap="magma")
plt.colorbar(label=r"$p_1$")
plt.xlabel(r"X")
plt.ylabel(r"Y")
plt.title(r"Population simulée $p_1$")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
plt.contourf(Theta, Lambda, P1, levels=100, cmap="magma")
plt.colorbar(label=r"$p_1$")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\theta$")
plt.title(r"Population simulée $p_1$")
plt.tight_layout()
plt.show()



plt.figure(figsize=(5,4))
plt.contourf(X, Y, P2, levels=100, cmap="magma")
plt.colorbar(label=r"$p_2$")
plt.xlabel(r"X")
plt.ylabel(r"Y")
plt.title(r"Population simulée $p_2$")
plt.tight_layout()
plt.show()



plt.figure(figsize=(5,4))
plt.contourf(X, Y, P3, levels=100, cmap="magma")
plt.colorbar(label=r"$p_3$")
plt.xlabel(r"X")
plt.ylabel(r"Y")
plt.title(r"Population simulée $p_3$")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,4))
plt.contourf(X, Y, P1+P2+P3, levels=100, cmap="magma")
plt.colorbar(label=r"$p_1^2+p_2^2+p_3^2$")
plt.xlabel(r"X")
plt.ylabel(r"Y")
plt.title(r"$p_1^2+p_2^2+p_3^2$")
plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------
#
# Contour plot : Erreurs
#
#----------------------------------------------------------------------------------

X1 = np.zeros((N_lambda, N_theta))
X2 = np.zeros((N_lambda, N_theta))
X3 = np.zeros((N_lambda, N_theta))

for i, lamb in enumerate(lambda_tab):
    for j, theta in enumerate(theta_tab):

        alpha = np.pi/2 * lamb/delta * np.cos(theta) / np.sin(theta0)
        beta = np.pi/4 *(1 - lamb/delta * np.cos(theta) / np.sin(theta0))

        x1 = np.cos(alpha)
        x2 = np.sin(alpha) * np.cos(beta)
        x3 = np.sin(alpha) * np.sin(beta)

        X1[i,j] = x1*x1
        X2[i,j] = x2*x2
        X3[i,j] = x3*x3

# 1
error_1 = np.abs(P1-X1)
plt.figure(figsize=(6,4))
plt.contourf(Theta, Lambda, error_1, 
             levels=100, cmap="magma")
plt.colorbar()
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\lambda$")
plt.title("Erreur pop 1 : modèle vs simulation")
plt.tight_layout()
plt.show()

# 2
error_2 = np.abs(P2-X2)
plt.figure(figsize=(6,4))
plt.contourf(Theta, Lambda, error_2, 
             levels=100, cmap="magma")
plt.colorbar()
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\lambda$")
plt.title("Erreur pop 2 : modèle vs simulation")
plt.tight_layout()
plt.show()


# 3
error_3 = np.abs(P3-X3)
plt.figure(figsize=(6,4))
plt.contourf(Theta, Lambda, error_3, 
             levels=100, cmap="magma")
plt.colorbar()
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\lambda$")
plt.title("Erreur pop 3 : modèle vs simulation")
plt.tight_layout()
plt.show()









#----------------------------------------------------------------------------------
#
# 3 contrôles
#
#----------------------------------------------------------------------------------


fig, ax = plt.subplots(figsize=(10,6))

ax.plot(time_us, u0_12, label=r'$\varphi_{0,1,2}$', color='C3', linewidth=1, linestyle='-.')
ax.plot(time_us, u0_23, label=r'$\varphi_{0,2,3}$', color='C4', linewidth=1, linestyle='-.')
ax.plot(time_us, u0_31, label=r'$\varphi_{0,3,1}$', color='C5', linewidth=1, linestyle='-.')

ax.plot(time_us, u12, label=r'$\varphi_{1,2}$', color='C0', linewidth=1.5, linestyle='-')
ax.plot(time_us, u23, label=r'$\varphi_{2,3}$', color='C1', linewidth=1.5, linestyle='-')
ax.plot(time_us, u31, label=r'$\varphi_{3,1}$', color='C2', linewidth=1.5, linestyle='-')

ax.axhline(0, linestyle='--', color='grey', alpha=0.5)

plt.axvline(0, color="black", linewidth=0.5)

ax.set_xlim(0, thold)
ax.set_xlabel(r'$t \ (\mu s)$', fontsize=20)
ax.set_ylabel(r'Phase $\varphi_{i,j}(t)$', fontsize=20)
ax.legend(fontsize=14)
ax.grid(alpha=0.3)

plt.show()

#----------------------------------------------------------------------------------
#
# Population à l'instant final 2D
#
#----------------------------------------------------------------------------------

m_vals = np.arange(-mmax, mmax + 1)
n_vals = np.arange(-nmax, nmax + 1)

M, N = np.meshgrid(m_vals, n_vals)

pop_matrix0 = np.zeros(M.shape)

for i, n in enumerate(n_vals):
    for j, m in enumerate(m_vals):
        k = sys2.get_k(m, n)
        pop_matrix0[i, j] = np.abs(Psi1[k, -1])**2 #Pop1

fig, ax = plt.subplots(figsize=(8, 7))

sizes = pop_matrix0.flatten() * 3000
col = pop_matrix0.flatten()

sc = ax.scatter(
    M.flatten(), N.flatten(),
    s=sizes,
    c=col,
    cmap='viridis',
    edgecolors='black'
)

for i, n in enumerate(n_vals):
    for j, m in enumerate(m_vals):
        if pop_matrix0[i, j] > 0.01:
            ax.text(
                m, n,
                f"{pop_matrix0[i, j]:.2f}",
                ha='center', va='center',
                fontsize=15, color='black', 
                fontweight='bold',
            )

ax.set_xticks(m_vals)
ax.set_yticks(n_vals)
ax.grid(True, linestyle='--', alpha=0.3)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(r'$|c_{m,n}|^2$', fontsize=20)

ax.set_xlabel(r'$m$', fontsize=30)
ax.set_ylabel(r'$n$', fontsize=30)

plt.tight_layout()
plt.show()


#Pop3

pop_matrix1 = np.zeros(M.shape)

for i, n in enumerate(n_vals):
    for j, m in enumerate(m_vals):
        k = sys2.get_k(m, n)
        pop_matrix1[i, j] = np.abs(Psi3[k, -1])**2
        
        
fig, ax = plt.subplots(figsize=(8, 7))

sizes = pop_matrix1.flatten() * 3000
col = pop_matrix1.flatten()

sc = ax.scatter(
    M.flatten(), N.flatten(),
    s=sizes,
    c=col,
    cmap='viridis',
    edgecolors='black'
)

for i, n in enumerate(n_vals):
    for j, m in enumerate(m_vals):
        if pop_matrix1[i, j] > 0.01:
            ax.text(
                m, n,
                f"{pop_matrix1[i, j]:.2f}",
                ha='center', va='center',
                fontsize=15, color='black', 
                fontweight='bold',
            )

ax.set_xticks(m_vals)
ax.set_yticks(n_vals)
ax.grid(True, linestyle='--', alpha=0.3)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(r'$|c_{m,n}|^2$', fontsize=20)

ax.set_xlabel(r'$m$', fontsize=30)
ax.set_ylabel(r'$n$', fontsize=30)

plt.tight_layout()
plt.show()


#----------------------------------------------------------------------------------
#
# Fidélité en fonction du temps
#
#----------------------------------------------------------------------------------

# Calcul des fidélités
# On projette l'état à chaque instant t sur l'état cible final
fidelity_sys1 = np.array([abs(cp.vdot(psit1_gpu, Psi1_gpu[:, i]).get())**2 for i in range(Nt)])
fidelity_sys2 = np.array([abs(cp.vdot(psit2_gpu, Psi2_gpu[:, i]).get())**2 for i in range(Nt)])
fidelity_sys3 = np.array([abs(cp.vdot(psit3_gpu, Psi3_gpu[:, i]).get())**2 for i in range(Nt)])

plt.figure(figsize=(8, 5))
plt.plot(time_us, fidelity_sys1, label="Fidélité sys1 (vers |0,0⟩)")
plt.plot(time_us, fidelity_sys2, label="Fidélité sys2 (vers |0,1⟩)")
plt.plot(time_us, fidelity_sys3, linestyle='-.', label="Fidélité sys3 (vers |1,0⟩)")

plt.xlabel(r"$t \ (\mu s)$", fontsize=16)
plt.ylabel("Fidélité instantanée", fontsize=16)
plt.title("Évolution de la fidélité vers les cibles")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


#----------------------------------------------------------------------------------
#
# FFT
#
#----------------------------------------------------------------------------------

# u12
U_fft = np.fft.fft(u0_12)
freq = np.fft.fftfreq(len(u12), d=(time_s[1]-time_s[0]))

plt.figure(figsize=(8, 5))
mask = freq > 0
plt.plot(freq[mask], np.abs(U_fft[mask]))
plt.xlabel("Fréquence (Hz)")
plt.ylabel(r"$|U_{12}(f)|$")
plt.grid(alpha=0.3)
plt.show()


# u23
U_fft = np.fft.fft(u0_23)
freq = np.fft.fftfreq(len(u23), d=(time_s[1]-time_s[0]))

plt.figure(figsize=(8, 5))
mask = freq > 0
plt.plot(freq[mask], np.abs(U_fft[mask]))
plt.xlabel("Fréquence (Hz)")
plt.ylabel(r"$|U_{23}(f)|$")
plt.grid(alpha=0.3)
plt.show()

# u31
U_fft = np.fft.fft(u0_31)
freq = np.fft.fftfreq(len(u31), d=(time_s[1]-time_s[0]))

plt.figure(figsize=(8, 5))
mask = freq > 0
plt.plot(freq[mask], np.abs(U_fft[mask]))
plt.xlabel("Fréquence (Hz)")
plt.ylabel(r"$|U_{31}(f)|$")
plt.grid(alpha=0.3)
plt.show()

#----------------------------------------------------------------------------------
#
# Population en m au cours du temps, du système 1
#
#----------------------------------------------------------------------------------

pop_m_time = np.zeros((len(m_vals), Nt))

for i, t_idx in enumerate(range(Nt)):
    for j, m in enumerate(m_vals):
        temp_pop = 0
        for n in n_vals:
            idx = sys1.get_k(m, n)
            temp_pop += np.abs(Psi1[idx, t_idx])**2
        pop_m_time[j, i] = temp_pop

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(
    pop_m_time, 
    aspect='auto', 
    origin='lower',
    extent=[time_us[0], time_us[-1], m_vals[0], m_vals[-1]],
    cmap='magma'
)

ax.set_xlabel(r'$t \ (\mu s)$', fontsize=18)
ax.set_ylabel(r'Impulsion $m$', fontsize=18)
ax.set_title(r'Évolution de la population projetée sur $m$ (sys1)', fontsize=16)

cbar = plt.colorbar(im)
cbar.set_label(r'$\sum_n |c_{m,n}|^2$', fontsize=14)

plt.tight_layout()
plt.show()


#----------------------------------------------------------------------------------
#
# Population en n au cours du temps, du système 1
#
#----------------------------------------------------------------------------------

pop_n_time = np.zeros((len(n_vals), Nt))

for i, t_idx in enumerate(range(Nt)):
    for j, n in enumerate(n_vals):
        temp_pop = 0
        for m in m_vals:
            idx = sys1.get_k(m, n)
            temp_pop += np.abs(Psi1[idx, t_idx])**2
        pop_n_time[j, i] = temp_pop

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(
    pop_n_time, 
    aspect='auto', 
    origin='lower',
    extent=[time_us[0], time_us[-1], n_vals[0], n_vals[-1]],
    cmap='magma'
)

ax.set_xlabel(r'$t \ (\mu s)$', fontsize=18)
ax.set_ylabel(r'Impulsion $n$', fontsize=18)
ax.set_title(r'Évolution de la population projetée sur $n$ (sys1)', fontsize=16)

cbar = plt.colorbar(im)
cbar.set_label(r'$\sum_n |c_{m,n}|^2$', fontsize=14)

plt.tight_layout()
plt.show()


#----------------------------------------------------------------------------------
#
# Animation
#
#----------------------------------------------------------------------------------

# Animation du système 1

fig, ax = plt.subplots(figsize=(8, 7))

pop0 = np.zeros((len(n_vals), len(m_vals)))

im = ax.imshow(
    pop0,
    origin='lower',
    extent=[m_vals[0]-0.5, m_vals[-1]+0.5,
            n_vals[0]-0.5, n_vals[-1]+0.5],
    cmap='magma',
    vmin=0,
    vmax=0.5
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$|c_{m,n}(t)|^2$')

ax.set_xlabel(r'$m$')
ax.set_ylabel(r'$n$')

def update(frame):
    pop_t = np.zeros_like(pop0)
    for i, n in enumerate(n_vals):
        for j, m in enumerate(m_vals):
            k = sysl.get_k(m, n)
            pop_t[i, j] = cp.abs(Psi1[k, frame])**2

    im.set_data(pop_t)
    ax.set_title(f"Évolution à t = {time_us[frame]:.2f} µs")
    return [im]

total_frames = Psi1.shape[1]

# frames_to_plot = total_frames 
frames_to_plot = list(range(0, total_frames, 2))
if frames_to_plot[-1] != total_frames - 1:
    frames_to_plot.append(total_frames - 1)

ani = animation.FuncAnimation(
    fig, update,
    frames=frames_to_plot, 
    interval=50,
    blit=False
)

html = ani.to_jshtml()
path = f"./data/animation/{nom_fichier}_{thold}_1.html"
with open(path, "w") as f:
    f.write(html)


# Animation du système 3

fig, ax = plt.subplots(figsize=(8, 7))

pop0 = np.zeros((len(n_vals), len(m_vals)))

im = ax.imshow(
    pop0,
    origin='lower',
    extent=[m_vals[0]-0.5, m_vals[-1]+0.5,
            n_vals[0]-0.5, n_vals[-1]+0.5],
    cmap='magma',
    vmin=0,
    vmax=0.5
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$|c_{m,n}(t)|^2$')

ax.set_xlabel(r'$m$')
ax.set_ylabel(r'$n$')

def update(frame):
    pop_t = np.zeros_like(pop0)
    for i, n in enumerate(n_vals):
        for j, m in enumerate(m_vals):
            k = sysl.get_k(m, n)
            pop_t[i, j] = cp.abs(Psi3[k, frame])**2

    im.set_data(pop_t)
    ax.set_title(f"Évolution à t = {time_us[frame]:.2f} µs")
    return [im]

total_frames = Psi3.shape[1]

# frames_to_plot = total_frames 
frames_to_plot = list(range(0, total_frames, 2))
if frames_to_plot[-1] != total_frames - 1:
    frames_to_plot.append(total_frames - 1)

ani = animation.FuncAnimation(
    fig, update,
    frames=frames_to_plot, 
    interval=50,
    blit=False
)

html = ani.to_jshtml()
path = f"./data/animation/{nom_fichier}_{thold}_3.html"
with open(path, "w") as f:
    f.write(html)
