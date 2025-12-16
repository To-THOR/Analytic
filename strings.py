# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#%%

# Données issues de Woodhouse (2012)

string_chosen   = "E2" 

string_freqs    = {"E4"  :329.6,
                   "B3" : 246.9,
                   "G3" : 196.0,
                   "D3" : 146.8,
                   "A2" : 110.0,
                   "E2" : 82.4}
 
string_mu       = {"E4" : 0.38e-3,
                   "B3" : 0.52e-3,
                   "G3" : 0.90e-3,
                   "D3" : 1.95e-3,
                   "A2" : 3.61e-3,
                   "E2" : 6.24e-3}

string_etaF      = {"E4" : 40e-5,
                   "B3" : 40e-5,
                   "G3" : 14e-5,
                   "D3" : 5e-5,
                   "A2" : 7e-5,
                   "E2" : 2e-5}

string_etaB     = {"E4" : 2.4e-2,
                   "B3" : 2e-2,
                   "G3" : 2e-2,
                   "D3" : 2e-2,
                   "A2" : 2.5e-2,
                   "E2" : 2.0e-2} 

string_etaA     = {"E4" : 1.5,
                   "B3" : 1.2,
                   "G3" : 1.7,
                   "D3" : 1.2,
                   "A2" : 0.9,
                   "E2" : 1.2}

string_EI       = {"E4" : 130e-6,
                   "B3" : 160e-6,
                   "G3" : 310e-6,
                   "D3" : 51e-6,
                   "A2" : 40e-6,
                   "E2" : 57e-6}

string_T        = {"E4" : 70.3,
                   "B3" : 53.4,
                   "G3" : 58.3,
                   "D3" : 71.2,
                   "A2" : 73.9,
                   "E2" : 71.6} 

L               = 0.65
D               = 0.001
mu              = string_mu[string_chosen]
T               = string_T[string_chosen]
fric            = 1e-5 # Coeff K for the friction force F = -K . v
etaF            = string_etaF[string_chosen]
etaB            = string_etaB[string_chosen]
etaA            = string_etaA[string_chosen]
EI              = string_EI[string_chosen]

params_names = np.array(("L",
                         "T",
                         "D",
                         "mu",
                         "fric"))

params = np.array((L,
                   T,
                   D,
                   mu,
                   fric))

Delta_y = 20e-3
Ny      = int(np.ceil(L / Delta_y)) 

f_max       = 20000
f_res       = 1e-8
w_res       = 2 * np.pi * f_res

x   = np.array(0)
y   = np.linspace(0, L, Ny)
z   = np.array(0) 

Delta_y = y[1] - y[0]

X,Y,Z = np.meshgrid(x,y,z)

x = X.flatten()
y = Y.flatten()
z = Z.flatten()

Nx, Ny, Nz = np.unique(x).size, np.unique(y).size, np.unique(z).size

Np          = Nx*Ny*Nz

#%% Flexion en z

kappa_max   = np.sqrt(mu/T) * 2 * np.pi * f_max 
n_max       = int(np.ceil(L/np.pi*kappa_max - 1/2))
n           = np.arange(n_max)
kappan_flz  = np.pi/L * (1/2 + n)

wn_flz      = kappan_flz * np.sqrt(T / mu)   * (1 + EI/(2*T) * kappan_flz**2)

Nm_flz      = wn_flz.size     

phinx_flz   = np.zeros((Nm_flz, Np))
phiny_flz   = np.zeros((Nm_flz, Np))
phinz_flz   = np.sin(kappan_flz[:,np.newaxis] * y[np.newaxis])

phin_flz    = np.sqrt(phinx_flz**2 + phiny_flz**2 + phinz_flz**2)

etan_flz    = (T * (etaF + etaA / wn_flz) + EI * etaB * kappan_flz**2) / \
    (T + EI * kappan_flz**2) 

m_flz       = mu * L / 2 * np.ones(n.size)
k_flz       = T * kappan_flz**2 * L / 2 
k_flz       = k_flz * (1 + 1j * etan_flz) 
c_flz       = np.imag(k_flz) / np.real(wn_flz)
k_flz       = np.real(k_flz)

Fx_fact_flz = np.zeros((Nm_flz, Np))
Fy_fact_flz = np.zeros((Nm_flz, Np))
Fz_fact_flz = phinz_flz

# Normalization

#fact        = 1 / np.abs(phin_flz).max(axis=1)
fact        = 1 / np.sqrt(m_flz)
fact        = fact[:,np.newaxis] 

phinx_flz   = fact * phinx_flz
phiny_flz   = fact * phiny_flz
phinz_flz   = fact * phinz_flz
Fx_fact_flz = fact * Fx_fact_flz
Fy_fact_flz = fact * Fy_fact_flz
Fz_fact_flz = fact * Fz_fact_flz

fact        = fact.flatten()

m_flz       = fact**2 * m_flz
k_flz       = fact**2 * k_flz
c_flz       = fact**2 * c_flz

#%% Flexion en x

kappa_max   = np.sqrt(mu/T) * 2 * np.pi * f_max 
n_max       = int(np.ceil(L/np.pi*kappa_max - 1/2))
n           = np.arange(n_max)
kappan_flx  = np.pi/L * (1/2 + n)

wn_flx      = kappan_flx * np.sqrt(T / mu)   * (1 + EI/(2*T) * kappan_flx**2)

Nm_flx      = wn_flx.size     

phinx_flx   = np.sin(kappan_flx[:,np.newaxis] * y[np.newaxis])
phiny_flx   = np.zeros((Nm_flx, Np))
phinz_flx   = np.zeros((Nm_flx, Np))

phin_flx    = np.sqrt(phinx_flx**2 + phiny_flx**2 + phinz_flx**2)

etan_flx    = (T * (etaF + etaA / wn_flx) + EI * etaB * kappan_flx**2) / \
    (T + EI * kappan_flx**2) 

m_flx       = mu * L / 2 * np.ones(n.size)
k_flx       = T * kappan_flx**2 * L / 2
k_flx       = k_flx * (1 + 1j * etan_flx) 
c_flx       = np.imag(k_flx) / np.real(wn_flx)
k_flx       = np.real(k_flx)

Fx_fact_flx = phinx_flx
Fy_fact_flx = np.zeros((Nm_flx, Np))
Fz_fact_flx = np.zeros((Nm_flx, Np))

# Normalization

# fact        = 1 / np.abs(phin_flx).max(axis=1)
fact        = 1 / np.sqrt(m_flx)
fact        = fact[:,np.newaxis]

phinx_flx   = fact * phinx_flx
phiny_flx   = fact * phiny_flx
phinz_flx   = fact * phinz_flx
Fx_fact_flx = fact * Fx_fact_flx
Fy_fact_flx = fact * Fy_fact_flx
Fz_fact_flx = fact * Fz_fact_flx

fact        = fact.flatten()

m_flx       = fact**2 * m_flx
k_flx       = fact**2 * k_flx
c_flx       = fact**2 * c_flx

#%% Corps rigide

# Rotation en x

# phinx_rx    = np.zeros((1,Np))
# phiny_rx    = np.zeros((1,Np))
# phinz_rx    = y[np.newaxis]
# phin_rx     = np.sqrt(phinx_rx**2 + phiny_rx**2 + phinz_rx**2)
# wn_rx       = np.array([0])
# m_rx        = np.array([mu * L**3 / 3])
# k_rx        = np.array([0])
# c_rx        = np.array([fric * D * L**3 * np.pi / 6])
# Fx_fact_rx  = np.zeros((1,Np))
# Fy_fact_rx  = np.zeros((1,Np))
# Fz_fact_rx  = phinz_rx

# Normalization

# fact        = 1 / np.abs(phin_rx).max(axis=1)[:,np.newaxis]

# phinx_rx    = fact * phinx_rx
# phiny_rx    = fact * phiny_rx
# phinz_rx    = fact * phinz_rx

# fact        = fact.flatten()

# m_rx        = fact * m_rx
# c_rx        = fact * c_rx

# Rotation en z

# phinx_rz    = -y[np.newaxis]
# phiny_rz    = np.zeros((1,Np))
# phinz_rz    = np.zeros((1,Np))
# phin_rz     = np.sqrt(phinx_rz**2 + phiny_rz**2 + phinz_rz**2)
# wn_rz       = np.array([0])
# m_rz        = np.array([mu * L**3 / 3])
# k_rz        = np.array([0])
# c_rz        = np.array([fric * D * L**3 * np.pi / 6])
# Fx_fact_rz  = phinx_rz
# Fy_fact_rz  = np.zeros((1,Np))
# Fz_fact_rz  = np.zeros((1,Np))

# Normalization

# fact        = 1 / np.abs(phin_rz).max(axis=1)[:,np.newaxis]

# phinx_rz    = fact * phinx_rz
# phiny_rz    = fact * phiny_rz
# phinz_rz    = fact * phinz_rz

# fact        = fact.flatten()

# m_rz        = fact * m_rz
# c_rz        = fact * c_rz

#%% Base modale unifiée

wn = np.concatenate((#wn_rx,
                     #wn_rz,
                     wn_flx,
                     wn_flz))

mn = np.concatenate((#m_rx,
                     #m_rz,
                     m_flx,
                     m_flz))

cn = np.concatenate((#c_rx,
                     #c_rz,
                     c_flx,
                     c_flz)) 

kn = np.concatenate((#k_rx,
                     #k_rz,
                     k_flx,
                     k_flz))

phinx = np.concatenate((#phinx_rx,
                        #phinx_rz,
                        phinx_flx,
                        phinx_flz))

phiny = np.concatenate((#phiny_rx,
                        #phiny_rz,
                        phiny_flx,
                        phiny_flz))

phinz = np.concatenate((#phinz_rx,
                        #phinz_rz,
                        phinz_flx,
                        phinz_flz))

Fx_fact = np.concatenate((#Fx_fact_rx,
                          #Fx_fact_rz,
                          Fx_fact_flx,
                          Fx_fact_flz))

Fy_fact = np.concatenate((#Fy_fact_rx,
                          #Fy_fact_rz,
                          Fy_fact_flx,
                          Fy_fact_flz))

Fz_fact = np.concatenate((#Fz_fact_rx,
                          #Fz_fact_rz,
                          Fz_fact_flx,
                          Fz_fact_flz))

idx_sort = wn.argsort()

wn          = wn[idx_sort]
mn          = mn[idx_sort] 
kn          = kn[idx_sort]
cn          = cn[idx_sort]
phinx       = phinx[idx_sort]
phiny       = phiny[idx_sort]
phinz       = phinz[idx_sort]
Fx_fact     = Fx_fact[idx_sort]
Fy_fact     = Fy_fact[idx_sort]
Fz_fact     = Fz_fact[idx_sort]

phin    = np.sqrt(phinx**2 + phiny**2 + phinz**2) 

fn      =  wn / (2 * np.pi) 

#%% Plot box

mode    = -1

phin_plot   = phin[mode]
x_plot      = x
y_plot      = y

plt.figure()
plt.scatter(x_plot,y_plot,c=phin_plot)
plt.axis('equal')
plt.title("f = "+ str(np.round(fn[mode].real,1)) + " Hz")
plt.colorbar()

# %% Save modal basis

name = "String_modal_basis_Dy_"+ string_chosen + f"_{(np.round(Delta_y*100,3)):.3f}" +"_cm_T_"+\
        f"{(np.round(T,0)):.0f}"+"_N_mu_"+f"{(np.round(mu*1e6,0)):.0f}"+"_mg.m-1"

np.savez("Data/"+name, x=x, y=y, z=z, wn=wn, mn=mn, kn=kn, cn=cn, phinx=phinx, 
         phiny=phiny, phinz=phinz, Fx_fact=Fx_fact, Fy_fact=Fy_fact, 
         Fz_fact=Fz_fact, params_names=params_names, params=params)

print("Saved as " + name)