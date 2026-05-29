# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 08:54:18 2026

@author: Raphael
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: bec_2D_state_to_state.py
# Author: Raphaël Heurtaux
# Description: 2D GRAPE for state-to-state transfer. Sparse CPU version.

#----------------------------------------------------------------------------------
#
# Modules
#
#----------------------------------------------------------------------------------

import numpy as np
from scipy.optimize import minimize
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.cm as cm
import matplotlib.colors as colors
from time import perf_counter

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],'size':20})
rc('text', usetex=True)

#----------------------------------------------------------------------------------
#
# Système
#
#----------------------------------------------------------------------------------

 
class BEC_2D():
    """
    Définit un système de BEC pour un réseau optique 2D.
    """
    def __init__(self, mmax, nmax, q, s):
        self.q = q
        self.s = s
        self.mmax = mmax
        self.nmax = nmax
        self.dim = (2*self.mmax + 1) * (2*self.nmax + 1)
        self.H12, self.H21, self.H23, self.H32, self.H31, self.H13 = self.get_H_l()
     
    def get_k(self, m, n):
        """
        Passage d'un état (m,n) à l'indice k correspondant.
        """
        return (m + self.mmax) * (2*self.nmax+1) + n + self.nmax
        
    def get_H_l(self):
        """
        Calcule les matrices hamiltoniennes une et une seule fois lors de l'instanciation de l'objet BEC_2D.
        """
        H12 = np.zeros((self.dim, self.dim), dtype=complex)
        H21 = np.zeros((self.dim, self.dim), dtype=complex)
        H23 = np.zeros((self.dim, self.dim), dtype=complex)
        H32 = np.zeros((self.dim, self.dim), dtype=complex)
        H31 = np.zeros((self.dim, self.dim), dtype=complex)
        H13 = np.zeros((self.dim, self.dim), dtype=complex)
        for m in range(-self.mmax, self.mmax + 1):
            for n in range(-self.nmax, self.nmax + 1):
                k = self.get_k(m, n)
                if n != self.nmax:
                    H12[k, self.get_k(m, n+1)] = -0.25 * self.s
                if m != self.mmax:
                    H31[k, self.get_k(m+1, n)] = -0.25 * self.s
                if n != -self.nmax:
                    H21[k, self.get_k(m, n-1)] = -0.25 * self.s
                if m != -self.mmax:
                    H13[k, self.get_k(m-1, n)] = -0.25 * self.s    
                if m != self.mmax and n != self.nmax:
                    H32[k, self.get_k(m+1, n+1)] = -0.25 * self.s
                if m != -self.mmax and n != -self.nmax:
                    H23[k, self.get_k(m-1, n-1)] = -0.25 * self.s
        return H12, H21, H23, H32, H31, H13

    def get_hamiltonian(self, t, u12, u23, u31):
        """
        Définition de l'hamiltonien total H au temps t.
        La construction de H0 se fait ici car il dépend de t.
        """
        H0 = np.zeros((self.dim, self.dim), dtype=complex)
        for m in range(-self.mmax, self.mmax + 1):
            for n in range(-self.nmax, self.nmax + 1):
                k = self.get_k(m, n)
                H0[k, k] = m**2 + n**2 - m*n
        return H0 + np.exp(1j*u12)*self.H12 + np.exp(-1j*u12)*self.H21 + np.exp(1j*u23)*self.H23 \
            + np.exp(-1j*u23)*self.H32 + np.exp(1j*u31)*self.H31 + np.exp(-1j*u31)*self.H13

    def get_dH_du12(self, t, u_t):
        """
        Dérivée de l'hamiltonien total par rapport au premier contrôle, à l'instant t.
        Permet de calculer le gradient pour la méthode du PMP.
        """
        return  1j * (np.exp(1j*u_t) * self.H12 - np.exp(-1j*u_t) * self.H21)
    
    def get_dH_du23(self, t, u_t):
        return 1j * (np.exp(1j*u_t) * self.H23 - np.exp(-1j*u_t) * self.H32)
    
    def get_dH_du31(self, t, u_t):
        return 1j * (np.exp(1j*u_t) * self.H31 - np.exp(-1j*u_t) * self.H13)   
        

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
        
        for n in range(self.Nt - 1):
            t_n = self.t[n]
            u12_n = u_control[n]
            u23_n = u_control[n + self.Nt]
            u31_n = u_control[n + 2*self.Nt]
            H = system.get_hamiltonian(t_n, u12_n, u23_n, u31_n)
            U = linalg.expm(-1j * H * self.dt)
            current_psi = U @ current_psi #Utiliser plutôt scipy.sparse.linalg.expm_multiply
            states[:, n+1] = current_psi
        return states

    def get_gradient_pmp(self, system, Psi, psi_target, u_control):
        dim = system.dim
        D = np.zeros((dim, self.Nt), dtype=complex)
        dF = np.zeros(3 * self.Nt)
        D[:, -1] = - np.vdot(psi_target, Psi[:, -1]) * psi_target #khi0=1
        
        for n in range(self.Nt - 2, -1, -1):
            t_n = self.t[n]
            u12_n = u_control[n]
            u23_n = u_control[n + self.Nt]
            u31_n = u_control[n + 2*self.Nt]
            H = system.get_hamiltonian(t_n, u12_n, u23_n, u31_n)
            U = linalg.expm(-1j * H * self.dt)
            D[:, n] = U.conj().T @ D[:, n+1]
            
        for n in range(self.Nt):
            t_n = self.t[n]
            u12_n = u_control[n]
            u23_n = u_control[n + self.Nt]
            u31_n = u_control[n + 2*self.Nt]
            
            dH_du12 = system.get_dH_du12(t_n, u12_n)
            dF[n] = (D[:, n].conj().T @ dH_du12 @ Psi[:, n]).imag

            dH_du23 = system.get_dH_du23(t_n, u23_n)
            dF[n + self.Nt] = (D[:, n].conj().T @ dH_du23 @ Psi[:, n]).imag

            dH_du31 = system.get_dH_du31(t_n, u31_n)
            dF[n + 2*self.Nt] = (D[:, n].conj().T @ dH_du31 @ Psi[:, n]).imag            
            
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
    def __init__(self, time_grid):
        self.evolver = Propagation_2D(time_grid)
        self.systems = [] 
        self.iter_count = 0
        
    def add_system(self, system, psi0, psi_target):
        self.systems.append((system, psi0, psi_target))
        
    def total_cost(self, u):
        cout_total = 0
        for sys, psi0, psit in self.systems:
            states = self.evolver.forward(sys, psi0, u)
            cout_total += (1.0 - abs(np.vdot(psit, states[:, -1]))**2)
        return cout_total / len(self.systems)

    def total_gradient(self, u):
        Nt = u.shape[0] // 3
        grad_total = np.zeros(3 * Nt)
        for sys, psi0, psit in self.systems:
            states = self.evolver.forward(sys, psi0, u)
            grad_i = self.evolver.get_gradient_pmp(sys, states, psit, u)
            grad_total += grad_i
        return grad_total / len(self.systems)

    def callback(self, xk):
        J = self.total_cost(xk)
        print(f'Iteration: {self.iter_count}, F={1-J:.15f}')
        self.iter_count += 1

#----------------------------------------------------------------------------------
#
# Paramètres
#
#----------------------------------------------------------------------------------

start_time = perf_counter()

# Constantes
mrb   = 86.909180527*1.66054e-27   # Masse d'un atome de Rb-87
wl    = 1064e-9                    # Longueur d'onde du laser
d     = wl/2.0                     # Période spatiale
hbar  = 1.0545718e-34              # Constante de Planck réduite
k_L   = (2.0*np.pi)/d              # Vecteur d'onde du réseau
E_L   = (hbar*k_L)**2/(2.0*mrb)    # Énergie du réseau
nu_L  = E_L/(2.0*np.pi*hbar)       # Fréquence du réseau

# Temps
Nt      = 400
thold   = 250
tf      = (thold*1.0e-6*E_L)/hbar
t       = np.linspace(0,tf,Nt) 

# Données système
mmax  = 4
nmax  = 4
q     = 0
s     = 5.
d = (2*mmax + 1) * (2*nmax + 1)

# Système
sys = BEC_2D(mmax, nmax, q, s)

#État initial et état cible
psi0       = np.zeros(d, dtype=complex)
k = sys.get_k(0,0)
psi0[k] = 1

psit = np.zeros(d, dtype=complex)
psit[sys.get_k(3,2)] = 1 /np.sqrt(3.)
psit[sys.get_k(-3,-1)] = 1/np.sqrt(3.)
psit[sys.get_k(-3,-3)] = 1/np.sqrt(3.)

#----------------------------------------------------------------------------------
#
# Résolution du problème de contrôle optimal
#
#----------------------------------------------------------------------------------
problem = OptimalControlProblem(t)
problem.add_system(sys, psi0, psit)

# Guess initial
u0 = np.pi* np.ones(3*Nt)

# Optimisation
sol = minimize(problem.total_cost, u0, jac=problem.total_gradient, 
                method='L-BFGS-B', callback=problem.callback, options={'maxiter': 150})
print(sol.status)
print(sol.message)


uopt = sol.x
u12 = uopt[:Nt]
u23 = uopt[Nt:2*Nt]
u31 = uopt[2*Nt:]

#----------------------------------------------------------------------------------
#
# Results
#
#----------------------------------------------------------------------------------

propagateur = Propagation_2D(t)
Psi = propagateur.forward(sys, psi0, uopt)

# Population
popf = abs(np.vdot(psit, Psi[:,-1]))**2
print(f"Population du système dans l'état final : {popf}")

end_time = perf_counter()
elapsed_time = end_time - start_time
print(f"Temps écoulé : {elapsed_time:.4f} secondes")













#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Plots
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------




#----------------------------------------------------------------------------------
#
# Plot 1 : 3 contrôles sur une seule courbe
#
#----------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10,6))

time_us = (t * hbar / E_L) * 1.0e6

ax.plot(time_us, u12, label=r'$\varphi_{1,2}$', color='C0', linewidth=1.5, linestyle=':')
ax.plot(time_us, u23, label=r'$\varphi_{2,3}$', color='C1', linewidth=1.5, linestyle='-')
ax.plot(time_us, u31, label=r'$\varphi_{3,1}$', color='C2', linewidth=1.5, linestyle=':')

ax.axhline(0, linestyle='--', color='grey', alpha=0.5)

plt.axvline(0, color="black", linewidth=0.5)

ax.set_xlim(0, thold)
ax.set_xlabel(r'$t \ (\mu s)$', fontsize=20)
#ax.set_ylim(-1.0, 1.0)
ax.set_ylabel(r'Phase $\varphi_{i,j}(t)$', fontsize=20)
ax.legend(fontsize=14)
ax.grid(alpha=0.3)

plt.show()

#----------------------------------------------------------------------------------
#
# Plot 2 : 2D
#
#----------------------------------------------------------------------------------

m_vals = np.arange(-mmax, mmax + 1)
n_vals = np.arange(-nmax, nmax + 1)

M, N = np.meshgrid(m_vals, n_vals)

pop_matrix = np.zeros(M.shape)

for i, n in enumerate(n_vals):
    for j, m in enumerate(m_vals):
        k = sys.get_k(m, n)
        pop_matrix[i, j] = np.abs(Psi[k, -1])**2

fig, ax = plt.subplots(figsize=(8, 7))

sizes = pop_matrix.flatten() * 3000
col = pop_matrix.flatten()

sc = ax.scatter(
    M.flatten(), N.flatten(),
    s=sizes,
    c=col,
    cmap='viridis',
    edgecolors='black'
)

for i, n in enumerate(n_vals):
    for j, m in enumerate(m_vals):
        if pop_matrix[i, j] > 0.01:
            ax.text(
                m, n,
                f"{pop_matrix[i, j]:.2f}",
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
#ax.set_title("Population dans chaque état d'impulsion", fontsize=20)

plt.tight_layout()
plt.show()


#----------------------------------------------------------------------------------
#
# Animation
#
#----------------------------------------------------------------------------------


import matplotlib.animation as animation


fig, ax = plt.subplots(figsize=(8, 7))

pop0 = np.zeros((len(n_vals), len(m_vals)))

im = ax.imshow(
    pop0,
    origin='lower',
    extent=[m_vals[0]-0.5, m_vals[-1]+0.5,
            n_vals[0]-0.5, n_vals[-1]+0.5],
    cmap='viridis',
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
            k = sys.get_k(m, n)
            pop_t[i, j] = np.abs(Psi[k, frame])**2

    im.set_data(pop_t)
    ax.set_title(f"Évolution à t = {time_us[frame]:.2f} µs")
    return [im]

ani = animation.FuncAnimation(
    fig, update,
    frames=range(0, Psi.shape[1], 5),
    interval=50,
    blit=False
)

html = ani.to_jshtml()
with open("./data/animation/2D_state_to_state.html", "w") as f:
    f.write(html)

