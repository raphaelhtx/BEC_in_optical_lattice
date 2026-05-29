#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: 1D_maxQFI.py
# Author: Raphaël Heurtaux
# Description: Sensing method aiming at finding the value of a parameter by maximizing the QFI in a BEC system, in a 1D optical lattice, using GRAPE algorithm. 

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

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],'size':20})
rc('text', usetex=True)

#----------------------------------------------------------------------------------
#
# Système
#
#----------------------------------------------------------------------------------

class BEC():
    """
    Définit un système de BEC pour fingerprinting ou maximisation de la QFI
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
    Par rapport au fingerprinting, le calcul de la propagation se fait sur le système étendu (psi, dpsi).
    """
    def __init__(self, time_grid):
        self.t = time_grid
        self.dt = time_grid[1] - time_grid[0]
        self.Nt = len(time_grid)

    def forward_extended(self, system, psi0, u_control):
        # L'état est maintenant de taille 2*Nk : [psi, dpsi/dlambda]
        dim = system.dim
        extended_dim = 2 * dim
        
        # Initialisation : dpsi/dlambda(0) = 0
        current_extended_psi = np.zeros(extended_dim, dtype=complex)
        current_extended_psi[:dim] = psi0 
        
        # psi and dpsi à chaque pas de temps
        Psi = np.zeros((dim, self.Nt), dtype=complex)
        dPsi = np.zeros((dim, self.Nt), dtype=complex)
        
        Psi[:, 0] = psi0
        
        for n in range(self.Nt - 1):
            t_n = self.t[n]
            u_n = u_control[n]
            
            # Récupération des matrices
            H = system.get_hamiltonian(t_n, u_n)
            dH_dlambda = system.get_dH_dlambda(t_n)
            
            # Construction de la matrice étendue 2Nx2N
            # [ H       0 ]
            # [ dH/dl   H ]
            H_tilde = np.zeros((extended_dim, extended_dim), dtype=complex)
            H_tilde[:dim, :dim] = H
            H_tilde[dim:, dim:] = H
            H_tilde[dim:, :dim] = dH_dlambda
            
            # Propagation
            U_tilde = linalg.expm(-1j * H_tilde * self.dt)
            current_extended_psi = U_tilde @ current_extended_psi
            
            Psi[:, n+1] = current_extended_psi[:dim]
            dPsi[:, n+1] = current_extended_psi[dim:]
            
        return Psi, dPsi
    
    def get_gradient_pmp(self, system, psi, dpsi, u_control):
        dim = system.dim
        extended_dim = 2 * dim
        khi_ext = np.zeros((extended_dim, self.Nt), dtype=complex)
        dF = np.zeros(self.Nt)
        khi_ext[:dim, -1] = - np.vdot(dpsi[:,-1], psi[:, -1]) * dpsi[:,-1] #* 4.0
        khi_ext[dim:, -1] = (dpsi[:,-1] - np.vdot(psi[:,-1], dpsi[:,-1]) * psi[:,-1]) #* 4.0
        
        # Backward Propagation   
        for n in range(self.Nt - 2, -1, -1): #n de Nt - 2 à 0
            tn = self.t[n]
            un = u_control[n]
            
            # Récupération des matrices
            H = system.get_hamiltonian(tn, un)
            dH_dlambda = system.get_dH_dlambda(tn)
            
            # Construction de la matrice étendue de backward propagation
            # [ H       0 ]
            # [ dH/dl   H ]
            H_tilde = np.zeros((extended_dim, extended_dim), dtype=complex)
            H_tilde[:dim, :dim] = H
            H_tilde[dim:, dim:] = H
            H_tilde[:dim, dim:] = dH_dlambda
            
            # Backward propagation
            U_tilde = linalg.expm(+1j * H_tilde * self.dt)
            khi_ext[:, n] = U_tilde @ khi_ext[:, n+1]
                       
        khi = khi_ext[:dim, :]
        dkhi = khi_ext[dim:, :]
        
        for n in range(self.Nt):
            t_n = self.t[n]
            u_n = u_control[n]
            dH_du = system.get_dH_du(t_n, u_n)
            grad1 = -np.imag(np.vdot(khi[:, n], dH_du @ psi[:, n]))
            grad2 = -np.imag(np.vdot(dkhi[:, n], dH_du @ dpsi[:, n]))
            dF[n] = grad1 + grad2
        return dF         

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
        
    def add_system(self, system, psi0):
        self.systems.append((system, psi0))
        
    def total_cost(self, u):
        cout_total = 0
        for sys, psi0 in self.systems:
            psi, dpsi = self.propagateur.forward_extended(sys, psi0, u)
            cout_total += QFI(psi[:,-1], dpsi[:,-1]) 
        return -cout_total / (len(self.systems)* 4.0)

    def total_gradient(self, u):
        grad_total = np.zeros_like(u)
        for sys, psi0 in self.systems:
            psi, dpsi = self.propagateur.forward_extended(sys, psi0, u)
            grad_i = self.propagateur.get_gradient_pmp(sys, psi, dpsi, u)
            grad_total += grad_i
        return grad_total / len(self.systems)
    
    def callback(self, xk):
        QFI = -self.total_cost(xk)
        print(f'Iteration: {self.iter_count}, F={QFI/1e6 :.6f}e6')
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
Nt      = 500
thold   = 500
tf      = (thold*1.0e-6*E_L)/hbar #40 temps normalisé
t       = np.linspace(0,tf,Nt)
t_us = (t*hbar*1.0e6)/E_L

# Données système
nmax  = 16
Nk    = 2*nmax+1
q     = 0.
s     = 5.
lamb  = 7e-4

# État initial. Pas d'état cible avec cette méthode.
psi0       = np.zeros(Nk, dtype=complex)
psi0[nmax] = 1
sys = BEC(Nk, q, s, lamb)



#----------------------------------------------------------------------------------
#
# Résolution du problème de contrôle optimal
#
#----------------------------------------------------------------------------------


problem = OptimalControlProblem(t)
problem.add_system(sys, psi0)

# Guess initial
u0 = np.zeros(Nt)

# Optimisation
sol = minimize(problem.total_cost, u0, jac=problem.total_gradient, 
                method='L-BFGS-B', callback=problem.callback, options={'maxiter': 100})

uopt = sol.x

#----------------------------------------------------------------------------------
#
# QFI / CFI et population
#
#----------------------------------------------------------------------------------

propagateur = Propagation(t)
psif, dpsif = propagateur.forward_extended(sys, psi0, uopt)

QFI_tab = np.zeros(Nt)
CFI_tab = np.zeros(Nt)

for n in range(Nt):
    QFI_tab[n] = QFI(psif[:, n], dpsif[:, n])
    CFI_tab[n] = CFI(psif[:, n], dpsif[:, n], Nk)

print(f"QFI($t_f$) = {QFI_tab[-1]/1e6 :.3f}e6")
end_time = time.time()
print(f"Temps écoulé : {end_time - start_time:.2f} secondes")


#----------------------------------------------------------------------------------
#
# Plots
#
#----------------------------------------------------------------------------------

# Control
fig1 = plt.figure(figsize=(10.4,7.49))
gs = gridspec.GridSpec(1, 1)

t_us = (t*hbar*1.0e6)/E_L
ax0 = plt.subplot(gs[0,0])

ax0.plot(t, uopt/np.pi, color='C1', linewidth=3, label='$u_{opt}$')
ax0.plot(t, u0/np.pi, color='C0', linewidth=3, label=r'$u_0$', linestyle='--')

ax0.set_xlim(0,t[-1])

ax0.axhline(0, linestyle='--', color='black', linewidth=0.5)
ax0.set_ylabel(r'$\varphi(t)/\pi$') 
ax0.set_xlabel(r'$t$')
ax0.legend()
ax0.grid(True)

plt.subplots_adjust(hspace=.0, wspace=0.3)
plt.show()


# QFI/CFI pour le temps optimal
fig2 = plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])

ax1 = plt.subplot(gs[0, 1])
ax1.plot(t_us, QFI_tab, color='C0', linewidth=3, label='QFI')
ax1.plot(t_us, CFI_tab, color='C1', linewidth=3, label='CFI', linestyle='--')
ax1.set_xlim(0, t_us[-1])
ax1.set_xlabel(r'$t$ ($\mu s$)')
ax1.set_ylabel(r'Information de Fisher')
#ax1.set_title(r'Information de Fisher après optimisation à $t^*$')
ax1.legend()
ax1.grid(True)

plt.tight_layout()
plt.show()


# Population au cours du temps dans l'espace des impulsions
plot_population_map(t, psif, nmax, E_L, hbar, cmap="viridis", show_cbar=True)