#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: 1D_fingerpriting.py
# Author: Raphaël Heurtaux
# Description: Fingerprinting de condensats de Bose-Einstein dans un réseau à une dimension, selon l'algorithme GRAPE.

#----------------------------------------------------------------------------------
#
# Modules
#
#----------------------------------------------------------------------------------

import numpy as np
from scipy.optimize import minimize
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
import time
import qutip as qt

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],'size':20})
rc('text', usetex=True)

#----------------------------------------------------------------------------------
#
# Système
#
#----------------------------------------------------------------------------------

class BEC():
    """
    Définit un système de BEC pour fingerprinting
    """
    def __init__(self, dimension, q, s, lamb):
        self.dim = dimension
        self.q = q
        self.s = s
        self.lamb = lamb # Le paramètre à identifier
        self.nmax = (dimension - 1) // 2
        self.H1, self.H2 = self.get_H1_H2()
        
    def get_H1_H2(self):
        """
        Définition des matrices hamiltoniennes de couplage.
        Fonction appelée une unique fois lors de l'instanciation de l'objet BEC.
        """
        H1 = np.zeros((self.dim, self.dim), dtype=complex)
        H2 = np.zeros((self.dim, self.dim), dtype=complex)
        for k in range(-self.nmax, self.nmax):
            j = k + self.nmax #Indices des matrices >=0
            H1[j, j+1] = -0.25 * self.s
            H1[j+1, j] = -0.25 * self.s
            H2[j, j+1] = 1j * 0.25 * self.s
            H2[j+1, j] = -1j * 0.25 * self.s
        return H1, H2

    def get_hamiltonian(self, t, u_t):
        """
        Définition de l'hamiltonien total H au temps t.
        La construction de H0 se fait ici car il dépend de t.
        """
        H0 = np.zeros((self.dim, self.dim), dtype=complex)
        for k in range(-self.nmax, self.nmax+1):
            j = k + self.nmax
            H0[j,j] = (self.q + k -self.lamb*t)**2
        return H0 + np.cos(u_t) * self.H1 + np.sin(u_t) * self.H2

    def get_dH_du(self, t, u_t):
        """
        Dérivée de l'hamiltonien total par rapport au contrôle, à l'instant t.
        """
        return -np.sin(u_t) * self.H1 + np.cos(u_t) * self.H2
    
    def get_dH_dlambda(self, t):
        """
        Calcule dH/d(lambda) pour le QFI.
        """
        dH_dlamb = np.zeros((self.dim, self.dim), dtype=complex)
        for k in range(-self.nmax, self.nmax+1):
            j = k + self.nmax
            dH_dlamb[j,j] = -2.0 * t * (self.q + k - self.lamb * t)
        return dH_dlamb

#----------------------------------------------------------------------------------
#
# Dynamique
#
#----------------------------------------------------------------------------------

class Propagation:
    """
    Pour un système donné, calcule sa propagation temporelle et calcule le gradient de Hp.
    """
    def __init__(self, time_grid):
        self.t = time_grid
        self.dt = time_grid[1] - time_grid[0]
        self.Nt = len(time_grid)

    def forward(self, system, psi0, u_control):
        """
        Calcule le l'ensemble des fonctions d'onde à chaque pas de temps.
        On propage, par pas de temps, en utilisant la popagateur exact.
        """
        states = np.zeros((system.dim, self.Nt), dtype=complex)
        states[:, 0] = psi0
        current_psi = psi0.copy()
        
        for i in range(self.Nt - 1):
            t_curr = self.t[i]
            ut = u_control[i]
            H = system.get_hamiltonian(t_curr, ut)
            U = linalg.expm(-1j * H * self.dt)
            current_psi = U @ current_psi
            states[:, i+1] = current_psi
        return states

    def get_gradient_pmp(self, system, states, psi_target, u_control):
        """
        Renvoie le gradient selon la condition de maximisation du PMP.
        """
        dim = system.dim
        D = np.zeros((dim, self.Nt), dtype=complex)
        dF = np.zeros(self.Nt)
        D[:, -1] = - np.vdot(psi_target, states[:, -1]) * psi_target #khi0=1
        
        for n in range(self.Nt - 1, 0, -1):
            t_curr = self.t[n-1]
            ut = u_control[n-1]
            H = system.get_hamiltonian(t_curr, ut)
            U = linalg.expm(-1j * H * self.dt)
            D[:, n-1] = U.conj().T @ D[:, n]
            
        for n in range(self.Nt):
            t_curr = self.t[n]
            ut = u_control[n]
            dH_du = system.get_dH_du(t_curr, ut)
            dF[n] = (D[:, n].conj().T @ dH_du @ states[:, n]).imag
        return dF

class PropagationEtendue(Propagation):
    """
    Une fois l'optimisation terminée, on propage le grand vecteur Psi=(psi, dpsi).
    Permet (uniquement) de calculer la QFI/CFI.
    """
    def forward_extended(self, system, psi0, u_control):
        # L'état est maintenant de taille 2*Nk : [psi, dpsi/dlambda]
        dim = system.dim
        extended_dim = 2 * dim
        
        # Initialisation : dpsi/dlambda(0) = 0
        current_extended_psi = np.zeros(extended_dim, dtype=complex)
        current_extended_psi[:dim] = psi0 
        
        # Historique (on sépare psi et dpsi)
        psi_hist = np.zeros((dim, self.Nt), dtype=complex)
        dpsi_hist = np.zeros((dim, self.Nt), dtype=complex)
        
        psi_hist[:, 0] = psi0
        
        for i in range(self.Nt - 1):
            t_curr = self.t[i]
            ut = u_control[i]
            
            # Récupération des matrices
            H = system.get_hamiltonian(t_curr, ut)
            dH_dlambda = system.get_dH_dlambda(t_curr)
            
            # Construction de la matrice étendue 2Nx2N (Eq 2.63)
            # [ H       0 ]
            # [ dH/dl   H ]
            H_tilde = np.zeros((extended_dim, extended_dim), dtype=complex)
            H_tilde[:dim, :dim] = H
            H_tilde[dim:, dim:] = H
            H_tilde[dim:, :dim] = dH_dlambda
            
            # Propagation
            U_tilde = linalg.expm(-1j * H_tilde * self.dt)
            current_extended_psi = U_tilde @ current_extended_psi
            
            psi_hist[:, i+1] = current_extended_psi[:dim]
            dpsi_hist[:, i+1] = current_extended_psi[dim:]
            
        return psi_hist, dpsi_hist

#----------------------------------------------------------------------------------
#
# Coûts et gradients pour l'optimisation
#
#----------------------------------------------------------------------------------

class OptimalControlProblem:
    """
    Gère l'algorithme GRAPE.
    Accepte plusieurs systèmes : le coût/gradient total est la moyenne des coûts/gradients de chaque système.
    """
    def __init__(self, time_grid):
        self.propagateur = Propagation(time_grid)
        self.systems = [] 
        self.iter_count = 0
        
    def add_system(self, system, psi0, psi_target):
        self.systems.append((system, psi0, psi_target))
        
    def total_cost(self, u_flat):
        cout_total = 0
        for sys, psi0, psit in self.systems:
            states = self.propagateur.forward(sys, psi0, u_flat)
            cout_total += (1.0 - abs(np.vdot(psit, states[:, -1]))**2)
        return cout_total / len(self.systems)

    def total_gradient(self, u_flat):
        grad_total = np.zeros_like(u_flat)
        for sys, psi0, psit in self.systems:
            states = self.propagateur.forward(sys, psi0, u_flat)
            grad_i = self.propagateur.get_gradient_pmp(sys, states, psit, u_flat)
            grad_total += grad_i
        return grad_total / len(self.systems)

    def callback(self, xk):
        J = self.total_cost(xk)
        print(f'Iteration: {self.iter_count}, F={1-J:.15f}')
        self.iter_count += 1

#----------------------------------------------------------------------------------
#
# Fonctions QFI / CFI
#
#----------------------------------------------------------------------------------

def QFI(psi, dpsi):
    """Calcule QFI."""
    return  (np.vdot(dpsi, dpsi).real - abs(np.vdot(psi, dpsi))**2) * 4.0

def CFI(psi, dpsi, dim):
    """Calcule CFI par rapport au PVM {|k><k|}."""
    CFI = 0
    for k in range(dim):
        alpha_k = np.zeros(dim)
        alpha_k[k] = 1
        denominator = abs(np.vdot(alpha_k, psi))**2
        if denominator != 0:
            CFI += (np.vdot(dpsi, alpha_k) * np.vdot(alpha_k, psi) + np.vdot(psi, alpha_k) * np.vdot(alpha_k, dpsi))**2 / denominator            
    return CFI.real


#----------------------------------------------------------------------------------
#
# Plot
#
#----------------------------------------------------------------------------------

def plot_population_map(t, C, nmax, E_L=None, h_bar=None, cmap="viridis", show_cbar=True):
    """
    Map de la population de chaque état (tronqués) à chaque pas de temps.
    Réécriture du code de Charles.
    """
    
    population = np.abs(C)**2
    momentum_indices = np.arange(-nmax, nmax + 1)

    if E_L is not None and h_bar is not None:
        t_plot = (t * h_bar * 1.0e6) / E_L
        x_label = r"$t$ ($\mu s$)"
    else:
        t_plot = t
        x_label = "$t$ (a.u.)"

    X, Y = np.meshgrid(t_plot, momentum_indices)

    golden = (1 + 5**0.5) / 2
    fig, ax = plt.subplots(layout="constrained", figsize=(6 * golden, 6))
    
    CS = ax.pcolormesh(
        X, 
        Y, 
        population, 
        cmap=cmap, 
        shading='nearest',
        vmin=0, 
        vmax=1 
    )
    
    ax.set_title("Population du BEC au cours du temps")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"État d'impulsion $k$")
    
    ax.set_yticks(momentum_indices)
    ax.set_yticks(momentum_indices[:-1] + 0.5, minor=True)
    ax.grid(True, which="minor", axis="y", linestyle="-", linewidth=0.5, color="white", alpha=0.5)
    
    ax.set_ylim(momentum_indices[0] - 0.5, momentum_indices[-1] + 0.5)
    ax.set_xlim(t_plot[0], t_plot[-1])

    if show_cbar:
        cbar = fig.colorbar(CS, ax=ax)
        cbar.ax.set_ylabel(r"$|C_n(t)|^2$")

    return fig, ax

#----------------------------------------------------------------------------------
#
# Paramètres
#
#----------------------------------------------------------------------------------
start_time = time.time()

# Constantes
mrb   = 86.909180527*1.66054e-27   # Masse d'un atome de Rb-87
wl    = 1064e-9                    # Longueur d'onde du laser
d     = wl/2.0                     # Période spatiale
hbar  = 1.0545718e-34              # Constante de Planck réduite
k_L   = (2.0*np.pi)/d              # Vecteur d'onde du réseau
E_L   = (hbar*k_L)**2/(2.0*mrb)    # Énergie du réseau
nu_L  = E_L/(2.0*np.pi*hbar)       # Fréquence du réseau

# Temps
Nt      = 1000
thold   = 800
tf      = (thold*1.0e-6*E_L)/hbar
t1       = np.linspace(0,tf,Nt)     # Choix un : avant de connaitre t*
tstar = 27.976                      # 550us
t2      = np.linspace(0,tstar,Nt)   # Choix deux : après mesure de t*
t=t2
t_us = (t*hbar*1.0e6)/E_L

# Données système
nmax  = 8
Nk    = 2*nmax+1
q     = 0
s     = 5.
lamb0 = 0.0
lamb1 = 8e-4

# États
psi0       = np.zeros(Nk, dtype=complex)
psi0[nmax] = 1

psit0 = np.zeros(Nk, dtype=complex)
psit0[nmax-1] = 1                   # Cible pour lambda 1
sys0 = BEC(Nk, q, s, lamb=lamb0)

psit1 = np.zeros(Nk, dtype=complex)
psit1[nmax+1] = 1                   # Cible pour lambda 2
sys1 = BEC(Nk, q, s, lamb=lamb1)


#----------------------------------------------------------------------------------
#
# Résolution du problème de contrôle optimal
#
#----------------------------------------------------------------------------------

problem = OptimalControlProblem(t)
problem.add_system(sys0, psi0, psit0)
problem.add_system(sys1, psi0, psit1)

# Guess initial
#u0 = np.cos(2*t)
u0 = np.zeros(Nt)

# Optimisation
sol = minimize(problem.total_cost, u0, jac=problem.total_gradient, 
                method='L-BFGS-B', callback=problem.callback, options={'maxiter': 150})
    
uopt = sol.x

#----------------------------------------------------------------------------------
#
# QFI / CFI et population
#
#----------------------------------------------------------------------------------

propagateurEtendu = PropagationEtendue(t)
psif0, dpsif0 = propagateurEtendu.forward_extended(sys0, psi0, uopt)
psif1, dpsif1 = propagateurEtendu.forward_extended(sys1, psi0, uopt)

# Population
popf0 = abs(np.vdot(psit0, psif0[:,-1]))**2
popf1 = abs(np.vdot(psit1, psif1[:,-1]))**2
print(f"Population du système 1 dans l'état -1 : {popf0}")
print(f"Population du système 2 dans l'état 1 : {popf1}")

QFI0_tab = np.zeros(Nt)
CFI0_tab = np.zeros(Nt)

for n in range(Nt):
    QFI0_tab[n] = QFI(psif0[:, n], dpsif0[:, n])
    CFI0_tab[n] = CFI(psif0[:, n], dpsif0[:, n], Nk)

end_time = time.time()
print(f"Temps écoulé : {end_time - start_time:.2f} secondes")



#----------------------------------------------------------------------------------
#
# Plots
#
#----------------------------------------------------------------------------------


# Contrôle optimal
fig = plt.figure(figsize=(10.4,7.49))
gs = gridspec.GridSpec(1, 1)

ax0 = plt.subplot(gs[0,0])
line0, = ax0.plot(t, u0/np.pi, color='C1')
line0, = ax0.plot(t, uopt/np.pi, color='C2')
ax0.set_xlim(0,t[-1])
ax0.axhline(0, linestyle='--', color='black', linewidth=0.5)
ax0.set_ylabel(r'$\varphi(t)/\pi$') 
ax0.set_xlabel(r'$t$')
plt.subplots_adjust(hspace=.0, wspace=0.3)
plt.show()


# QFI/CFI pour le temps optimal
fig = plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])

ax1 = plt.subplot(gs[0, 1])
ax1.plot(t_us, QFI0_tab, color='C0', linewidth=3, label='QFI')
ax1.plot(t_us, CFI0_tab, color='C1', linewidth=3, label='CFI', linestyle='--')
ax1.set_xlim(0, t_us[-1])
ax1.set_xlabel(r'$t$ ($\mu s$)')
ax1.set_ylabel(r'Information de Fisher')
ax1.set_title(r'Information de Fisher après optimisation à $t^*$')
ax1.legend()
ax1.grid(True)

plt.tight_layout()
plt.show()


# Population dans l'espace des impulsions au cours du temps
plot_population_map(t, psif0, nmax, E_L, hbar, cmap="viridis", show_cbar=True)
plt.show()
plot_population_map(t, psif1, nmax, E_L, hbar, cmap="viridis", show_cbar=True)
plt.show()



# Géodésique : population dans les états |-1> et |1> en fonction de lambda
N = 20
lambda_tab = np.linspace(0., 0.0008, N)
propagateur = Propagation(t)
p0_tab = np.zeros(N)
p1_tab = np.zeros(N)
for k in range(N):
    l = lambda_tab[k]
    sysl = BEC(Nk, q, s, l)
    psil = propagateur.forward(sysl, psi0, uopt)
    popl0 = abs(np.vdot(psit0, psil[:,-1]))**2
    popl1 = abs(np.vdot(psit1, psil[:,-1]))**2
    p0_tab[k] = popl0
    p1_tab[k] = popl1
    
    
plt.figure(figsize=(8, 3))
plt.plot(lambda_tab, p0_tab, marker='o', linestyle='-', color='blue', label='-1')
plt.plot(lambda_tab, p1_tab, marker='o', linestyle='-', color='orange', label='1')
plt.xlabel(r'$\lambda$', fontsize=14)
plt.ylabel(r'$p\pm1$', fontsize=14)
plt.title(r"Population des états $|-\!1\rangle$ et $|1\rangle$  à $t^*$ selon la force $\lambda$") #Géodésique pour tf=t*
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()



# Sphère de Bloch
b = qt.Bloch()

# Les deux états finaux psi1 et psi2
psif0 = psif0[:,-1]
psif1 = psif1[:,-1]

# Orthogonalisation Gram Schmidt de psi1 et psi2
e0 = psif0 / np.linalg.norm(psif0)
v = psif1 - np.vdot(e0, psif1) * e0
e1 = v / np.linalg.norm(v)

# Réécriture des matrices de Pauli dans cette base du sous-espace {psi1, psi2}
sx = np.outer(e0, e1.conj()) + np.outer(e1, e0.conj())
sy = -1j*np.outer(e0, e1.conj()) + 1j*np.outer(e1, e0.conj())
sz = np.outer(e0, e0.conj()) - np.outer(e1, e1.conj())

# Calculs des (x, y, z) le long de la trajectoire pour lambda variant de lambda1 à lambda2
points = np.zeros((3, N))
for k in range(N):
    l = lambda_tab[k]
    sysl = BEC(Nk, q, s, l)
    psil = propagateur.forward(sysl, psi0, uopt)
    psilf = psil[:,-1]
    psilf /= np.linalg.norm(psilf)
    x = np.real(np.vdot(psilf, sx @ psilf))
    y = np.real(np.vdot(psilf, sy @ psilf))
    z = np.real(np.vdot(psilf, sz @ psilf))
    points[0, k] = x
    points[1, k] = y
    points[2, k] = z
    
b.add_points(points, 'l') #cf doc : 'm' for multicolored, 'l' for points connected with a line.
b.show()

