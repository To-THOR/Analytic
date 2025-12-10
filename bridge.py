# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as  mv

# Unités SI

L               = 0.205
w               = 0.035
h               = 0.010
rho             = 750
beta            = 1/16 * (16 / 3 - 3.36 * h / w * (1 - 1/12 * h**4 / w**4))
K               = beta * h**3 * w
eta             = 0.01 
E               = 13.3e9 * (1 + 1j * eta)
G               = 0.93e9 * (1 + 1j * eta)
Iz              = w * h**3 / 12
Iy              = h * w**3 / 12
J               = (w * h**3 + h * w**3) / 12
fric            = 1e-5 # Coeff K for the friction force F = -K . v

params_names = np.array(("L",
                         "w",
                         "h",
                         "rho",
                         "K",
                         "eta",
                         "E",
                         "G",
                         "fric"))

params = np.array((L,
                   w,
                   h,
                   rho,
                   K,
                   eta,
                   E,
                   G,
                   fric))

Delta_x = 40e-3
Delta_y = 20e-3

f_max       = 5000
f_res       = 1e-8
w_res       = 2 * np.pi * f_res
kappa_max   = 2 * np.pi / Delta_x

Nx = int(np.ceil(L / Delta_x))
Ny = int(np.ceil(w / Delta_y))

x   = np.linspace(0, L, Nx)
y   = np.linspace(-w/2, w/2, Ny)
z   = np.array([-h/2]) 

Delta_x = x[1] - x[0]
Delta_y = y[1] - y[0]

X,Y,Z = np.meshgrid(x,y,z)

x = X.flatten()
y = Y.flatten()
z = Z.flatten()

Nx, Ny, Nz = np.unique(x).size, np.unique(y).size, np.unique(z).size

Nx_other = 20
Ny_other = 5
Nz_other = 3

x_other   = np.linspace(0, L, Nx_other)
y_other   = np.linspace(-w/2, w/2, Ny_other)
z_other   = np.linspace(-h*0.49, h/2, Nz_other) 

X,Y,Z = np.meshgrid(x_other,y_other,z_other)

x_other = X.flatten()
y_other = Y.flatten()
z_other = Z.flatten()

x = np.concatenate((x,x_other))
y = np.concatenate((y,y_other))
z = np.concatenate((z,z_other))

Np          = Nx*Ny*Nz + Nx_other*Ny_other*Nz_other 

#%% Flexion en z

# kappa_max   = (rho * h * w / (E * I))**(1/4) * np.sqrt(w_max)

kappanL         = np.load("beam_kappaL.npy")
kappan          = kappanL / L
kappan          = kappan[kappan < kappa_max]

if rho != 0:
    wn_flz          = np.sqrt(E * Iz / (rho * h * w)) * kappan**2 
else:
    wn_flz          = np.zeros(kappan.shape)
fn_flz              = wn_flz / (2 * np.pi)

N_flz = wn_flz.size

phinx_flz   = np.zeros((N_flz,Np))
phiny_flz   = np.zeros((N_flz,Np))
phinz_flz   = np.zeros((N_flz,Np))

kap = kappan[0]
alphan          = (np.sin(kap * L) - np.sinh(kap * L)) / \
                    (np.cosh(kap * L) - np.cos(kap * L))
phinx_flz[0]    = - z * kap * \
    ( np.cosh(kap * x) + np.cos(kap * x) + \
                     alphan * ( np.sinh(kap * x) - np.sin(kap * x) ))
phinz_flz[0]    = np.sinh(kap * x) + np.sin(kap * x) + \
    alphan * (np.cosh(kap * x) + np.cos(kap * x))

for i,kap in enumerate(kappan[1:],1):
    phinx_flz[i]   = - z * kap * (np.exp(-kap*x) + \
                                  np.exp(kap*(x-L)) * np.sin(kap * L) +\
                                  np.sin(kap * x) + np.cos(kap * x))
    phinz_flz[i]   = -np.exp(-kap*x) + np.exp(kap*(x-L)) * np.sin(kap*L) + \
                        np.sin(kap*x) - np.cos(kap*x)

phin_flz    = np.sqrt(phinx_flz**2 + phiny_flz**2 + phinz_flz**2)


Int_flz     = np.zeros(N_flz)

file        = np.load("Integral_beam_mode_0.npz")
Int_flz[0]   = 1/kappan[0] * file["I_phi_phi"] 
Int_flz[1:]  = 1/(2 * kappan[1:]) * (np.sin(kappan[1:]*L)**2 - \
                                    2 * np.sin(2*kappan[1:]*L) + \
                                    2 * kappan[1:] * L - 1)

m_flz                       = rho * w * h * Int_flz
k_flz                       = E * Iz * kappan**4 * Int_flz
c_flz                       = np.zeros(N_flz) 
c_flz[wn_flz!=0]            = np.imag(k_flz[wn_flz!=0]) / np.real(wn_flz[wn_flz!=0])
k_flz                       = np.real(k_flz)
Fx_fact_flz                 = np.zeros((N_flz,Np))
Fy_fact_flz                 = np.zeros((N_flz,Np))
Fz_fact_flz                 = np.zeros((N_flz,Np)) 
Fz_fact_flz[:,z==z.max()]   = phinz_flz[:,z==z.max()]
Fz_fact_flz[:,z==z.min()]   = phinz_flz[:,z==z.min()]

# Normalization

fact        = 1 / np.abs(phin_flz).max(axis=1)[:,np.newaxis]

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

#%% Flexion en y

# kappa_max   = (rho * h * w / (E * I))**(1/4) * np.sqrt(w_max)

kappanL         = np.load("beam_kappaL.npy")
kappan          = kappanL / L
kappan          = kappan[kappan < kappa_max]

if rho != 0:
    wn_fly          = np.sqrt(E * Iy / (rho * h * w)) * kappan**2   
else : 
    wn_fly          = np.zeros(kappan.shape)
fn_fly          = wn_fly / (2 * np.pi)

N_fly   = wn_fly.size

phinx_fly   = np.zeros((N_fly,Np))
phiny_fly   = np.zeros((N_fly,Np))
phinz_fly   = np.zeros((N_fly,Np))

kap = kappan[0]
alphan          = (np.sin(kap * L) - np.sinh(kap * L)) / \
                    (np.cosh(kap * L) - np.cos(kap * L))
phinx_fly[0]    = - y * kap * \
    ( np.cosh(kap * x) + np.cos(kap * x) + \
                     alphan * ( np.sinh(kap * x) - np.sin(kap * x) ))
phiny_fly[0]    = np.sinh(kap * x) + np.sin(kap * x) + \
    alphan * (np.cosh(kap * x) + np.cos(kap * x))

for i,kap in enumerate(kappan[1:],1):
    phinx_fly[i]   = - y * kap * (np.exp(-kap*x) + \
                                  np.exp(kap*(x-L)) * np.sin(kap * L) +\
                                  np.sin(kap * x) + np.cos(kap * x))
    phiny_fly[i]   = -np.exp(-kap*x) + np.exp(kap*(x-L)) * np.sin(kap*L) + \
                        np.sin(kap*x) - np.cos(kap*x)

phin_fly = np.sqrt(phinx_fly**2 + phiny_fly**2 + phinz_fly**2)


Int_fly      = np.zeros(N_fly)

file        = np.load("Integral_beam_mode_0.npz")
Int_fly[0]   = 1/kappan[0] * file["I_phi_phi"] 
Int_fly[1:]  = 1/(2 * kappan[1:]) * (np.sin(kappan[1:]*L)**2 - \
                                    2 * np.sin(2*kappan[1:]*L) + \
                                    2 * kappan[1:] * L - 1)

m_fly                       = rho * w * h * Int_fly
k_fly                       = E * Iy * kappan**4 * Int_fly
c_fly                       = np.zeros(N_fly) 
c_fly[wn_fly!=0]            = np.imag(k_fly[wn_fly!=0]) / np.real(wn_fly[wn_fly!=0])
k_fly                       = np.real(k_fly)
Fx_fact_fly                 = np.zeros((N_fly,Np))
Fy_fact_fly                 = np.zeros((N_fly,Np))
Fz_fact_fly                 = np.zeros((N_fly,Np)) 
Fz_fact_fly[:,z==z.max()]   = phinz_fly[:,z==z.max()]
Fz_fact_fly[:,z==z.min()]   = phinz_fly[:,z==z.min()]

# Normalization

fact        = 1 / np.abs(phin_fly).max(axis=1)[:,np.newaxis]

phinx_fly   = fact * phinx_fly
phiny_fly   = fact * phiny_fly
phinz_fly   = fact * phinz_fly
Fx_fact_fly = fact * Fx_fact_fly
Fy_fact_fly = fact * Fy_fact_fly
Fz_fact_fly = fact * Fz_fact_fly

fact        = fact.flatten()

m_fly       = fact**2 * m_fly
k_fly       = fact**2 * k_fly
c_fly       = fact**2 * c_fly

#%% Torsion

# n_max = np.real(np.sqrt(rho * J/ (G * K))) * L / np.pi * w_max
n_max   = int(np.ceil(kappa_max  * L / np.pi))  
n       = np.arange(1, n_max)
N_to    = int(n[-1])

if rho!=0:
    wn_to   = np.sqrt(G * K / (rho * J)) * n * np.pi / L 
else: 
    wn_to = np.zeros(n.shape)

phinx_to = np.zeros((N_to, Np))
phiny_to = np.zeros((N_to, Np))
phinz_to = np.zeros((N_to, Np))

for i,nn in enumerate(n):
    thetan  = np.cos(nn * np.pi / L * x)
    phiny_to[i] = -z * thetan
    phinz_to[i] = y * thetan

phin_to = np.sqrt(phinx_to**2 + phiny_to**2 + phinz_to**2)

m_to                        = np.ones(N_to) * rho * J * L / 2
k_to                        = G * K * (n * np.pi)**2 / (2 * L)
c_to                        = np.zeros(N_to) 
c_to[wn_to!=0]              = np.imag(k_to[wn_to!=0]) / np.real(wn_to[wn_to!=0])
k_to                        = np.real(k_to)
Fx_fact_to                  = np.zeros((N_to,Np))
Fy_fact_to                  = np.zeros((N_to,Np))
Fz_fact_to                  = np.zeros((N_to,Np))
Fz_fact_to[:,z==z.max()]    = phinz_to[:,z==z.max()]
Fz_fact_to[:,z==z.min()]    = phinz_to[:,z==z.min()]

# Normalization

fact        = 1 / np.abs(phin_to).max(axis=1)[:,np.newaxis]

phinx_to    = fact * phinx_to
phiny_to    = fact * phiny_to
phinz_to    = fact * phinz_to
Fx_fact_to  = fact * Fx_fact_to
Fy_fact_to  = fact * Fy_fact_to
Fz_fact_to  = fact * Fz_fact_to

fact        = fact.flatten()

m_to        = fact**2 * m_to
k_to        = fact**2 * k_to
c_to        = fact**2 * c_to

#%% Traction / Compression en x

n_max   = int(np.ceil(kappa_max * L / np.pi))
n       = np.arange(1, n_max)
N_tr    = n.size 

if rho != 0:
    wn_tr   = np.sqrt(E / rho) * n * np.pi / L
else:
    wn_tr   = np.zeros(n.shape)

phinx_tr = np.zeros((N_tr, Np))
phiny_tr = np.zeros((N_tr, Np))
phinz_tr = np.zeros((N_tr, Np))

for i,nn in enumerate(n):
    phinx_tr[i] = np.cos(nn*np.pi/L*x)

phin_tr = np.sqrt(phinx_tr**2 + phiny_tr**2 + phinz_tr**2)

m_tr            = rho*w*h*L/2*np.ones(N_tr)
k_tr            = E * w * h * (n * np.pi)**2 / (2 * L)
c_tr            = np.zeros(N_tr)
c_tr[wn_tr!=0]  = np.imag(k_tr[wn_tr!=0]) / np.real(wn_tr[wn_tr!=0])
k_tr            = np.real(k_tr)
Fx_fact_tr      = np.zeros((N_tr,Np))
Fy_fact_tr      = np.zeros((N_tr,Np))
Fz_fact_tr      = np.zeros((N_tr,Np)) 

# Normalization

fact        = 1 / np.abs(phin_tr).max(axis=1)[:,np.newaxis]

phinx_tr    = fact * phinx_tr
phiny_tr    = fact * phiny_tr
phinz_tr    = fact * phinz_tr
Fx_fact_tr  = fact * Fx_fact_tr
Fy_fact_tr  = fact * Fy_fact_tr
Fz_fact_tr  = fact * Fz_fact_tr

fact        = fact.flatten()

m_tr        = fact**2 * m_tr
k_tr        = fact**2 * k_tr
c_tr        = fact**2 * c_tr

#%% Corps rigide

# Translation en x

phinx_tx    = np.ones((1,Np))
phiny_tx    = np.zeros((1,Np))
phinz_tx    = np.zeros((1,Np))
wn_tx       = np.array([0])
m_tx        = np.array([w * h * L * rho])
k_tx        = np.array([0])
c_tx        = np.array([fric*h*w])
Fx_fact_tx                  = np.zeros((1,Np))
Fx_fact_tx[:,z==z.max()]    = phinx_tx[:,z==z.max()] 
Fx_fact_tx[:,z==z.min()]    = phinx_tx[:,z==z.min()]
Fy_fact_tx                  = np.zeros((1,Np))
Fy_fact_tx[:,z==z.max()]    = phiny_tx[:,z==z.max()] 
Fy_fact_tx[:,z==z.min()]    = phiny_tx[:,z==z.min()]
Fz_fact_tx                  = np.zeros((1,Np))
Fz_fact_tx[:,z==z.max()]    = phinz_tx[:,z==z.max()] 
Fz_fact_tx[:,z==z.min()]    = phinz_tx[:,z==z.min()]

# Translation en y

phinx_ty    = np.zeros((1,Np))
phiny_ty    = np.ones((1,Np))
phinz_ty    = np.zeros((1,Np))
wn_ty       = np.array([0])
m_ty        = np.array([w * h * L * rho])
k_ty        = np.array([0])
c_ty        = np.array([fric*L*h])
Fx_fact_ty                  = np.zeros((1,Np))
Fx_fact_ty[:,z==z.max()]    = phinx_ty[:,z==z.max()] 
Fx_fact_ty[:,z==z.min()]    = phinx_ty[:,z==z.min()]
Fy_fact_ty                  = np.zeros((1,Np))
Fy_fact_ty[:,z==z.max()]    = phiny_ty[:,z==z.max()] 
Fy_fact_ty[:,z==z.min()]    = phiny_ty[:,z==z.min()]
Fz_fact_ty                  = np.zeros((1,Np))
Fz_fact_ty[:,z==z.max()]    = phinz_ty[:,z==z.max()] 
Fz_fact_ty[:,z==z.min()]    = phinz_ty[:,z==z.min()]

# Translation en z

phinx_tz    = np.zeros((1,Np))
phiny_tz    = np.zeros((1,Np))
phinz_tz    = np.ones((1,Np))
wn_tz       = np.array([0])
m_tz        = np.array([w * h * L * rho])
k_tz        = np.array([0])
c_tz        = np.array([fric*L*w])
Fx_fact_tz                  = np.zeros((1,Np))
Fx_fact_tz[:,z==z.max()]    = phinx_tz[:,z==z.max()] 
Fx_fact_tz[:,z==z.min()]    = phinx_tz[:,z==z.min()]
Fy_fact_tz                  = np.zeros((1,Np))
Fy_fact_tz[:,z==z.max()]    = phiny_tz[:,z==z.max()] 
Fy_fact_tz[:,z==z.min()]    = phiny_tz[:,z==z.min()]
Fz_fact_tz                  = np.zeros((1,Np))
Fz_fact_tz[:,z==z.max()]    = phinz_tz[:,z==z.max()] 
Fz_fact_tz[:,z==z.min()]    = phinz_tz[:,z==z.min()]

# Rotation en x

phinx_rx    = np.zeros((1,Np))
phiny_rx    = -z[np.newaxis]
phinz_rx    = y[np.newaxis]
phin_rx     = np.sqrt(phinx_rx**2 + phiny_rx**2 + phinz_rx**2)
wn_rx       = np.array([0])
m_rx        = np.array([rho * (w * h * L) / 12 * (w**2 + h**2)])
k_rx        = np.array([0])
c_rx        = np.array([fric * L * w / 12 * (3 * h**2 + w**2) ])
Fx_fact_rx                  = np.zeros((1,Np))
Fx_fact_rx[:,z==z.max()]    = phinx_rx[:,z==z.max()] 
Fx_fact_rx[:,z==z.min()]    = phinx_rx[:,z==z.min()]
Fy_fact_rx                  = np.zeros((1,Np))
Fy_fact_rx[:,z==z.max()]    = phiny_rx[:,z==z.max()] 
Fy_fact_rx[:,z==z.min()]    = phiny_rx[:,z==z.min()]
Fz_fact_rx                  = np.zeros((1,Np))
Fz_fact_rx[:,z==z.max()]    = phinz_rx[:,z==z.max()] 
Fz_fact_rx[:,z==z.min()]    = phinz_rx[:,z==z.min()]

# Normalization

fact        = 1 / np.abs(phin_rx).max(axis=1)[:,np.newaxis]

phinx_rx    = fact * phinx_rx
phiny_rx    = fact * phiny_rx
phinz_rx    = fact * phinz_rx

fact        = fact.flatten()

m_rx        = fact * m_rx
c_rx        = fact * c_rx

# Rotation en y

phinx_ry    = z[np.newaxis]
phiny_ry    = np.zeros((1,Np))
phinz_ry    = -x[np.newaxis]
phin_ry     = np.sqrt(phinx_ry**2 + phiny_ry**2 + phinz_ry**2)
wn_ry       = np.array([0])
m_ry        = np.array([rho * (h * w * L)/12 * (4 * L**2 + h**2)])
k_ry        = np.array([0])
c_ry        = np.array([fric * w * L / 12 * (4 * L**2 + 3 * h**2)])
Fx_fact_ry                  = np.zeros((1,Np))
Fx_fact_ry[:,z==z.max()]    = phinx_ry[:,z==z.max()] 
Fx_fact_ry[:,z==z.min()]    = phinx_ry[:,z==z.min()]
Fy_fact_ry                  = np.zeros((1,Np))
Fy_fact_ry[:,z==z.max()]    = phiny_ry[:,z==z.max()] 
Fy_fact_ry[:,z==z.min()]    = phiny_ry[:,z==z.min()]
Fz_fact_ry                  = np.zeros((1,Np))
Fz_fact_ry[:,z==z.max()]    = phinz_ry[:,z==z.max()] 
Fz_fact_ry[:,z==z.min()]    = phinz_ry[:,z==z.min()]

# Normalization

fact        = 1 / np.abs(phin_ry).max(axis=1)[:,np.newaxis]

phinx_ry    = fact * phinx_ry
phiny_ry    = fact * phiny_ry
phinz_ry    = fact * phinz_ry

fact        = fact.flatten()

m_ry        = fact * m_ry
c_ry        = fact * c_ry

# Rotation en z

phinx_rz    = -y[np.newaxis]
phiny_rz    = x[np.newaxis]
phinz_rz    = np.zeros((1,Np))
phin_rz     = np.sqrt(phinx_rz**2 + phiny_rz**2 + phinz_rz**2)
wn_rz       = np.array([0])
m_rz        = np.array([rho * (h * w * L)/12 * (4 * L**2 + w**2)])
k_rz        = np.array([0])
c_rz        = np.array([fric * L * h / 12 * (4 * L**2 + 3 * w**2)])
Fx_fact_rz                  = np.zeros((1,Np))
Fx_fact_rz[:,z==z.max()]    = phinx_rz[:,z==z.max()] 
Fx_fact_rz[:,z==z.min()]    = phinx_rz[:,z==z.min()]
Fy_fact_rz                  = np.zeros((1,Np))
Fy_fact_rz[:,z==z.max()]    = phiny_rz[:,z==z.max()] 
Fy_fact_rz[:,z==z.min()]    = phiny_rz[:,z==z.min()]
Fz_fact_rz                  = np.zeros((1,Np))
Fz_fact_rz[:,z==z.max()]    = phinz_rz[:,z==z.max()] 
Fz_fact_rz[:,z==z.min()]    = phinz_rz[:,z==z.min()]

# Normalization

fact        = 1 / np.abs(phin_rz).max(axis=1)[:,np.newaxis]

phinx_rz    = fact * phinx_rz
phiny_rz    = fact * phiny_rz
phinz_rz    = fact * phinz_rz

fact        = fact.flatten()

m_rz        = fact * m_rz
c_rz        = fact * c_rz

#%% Base modale unifiée

wn = np.concatenate((wn_tx,
                     wn_ty,
                     wn_tz,
                     wn_rx,
                     wn_ry,
                     wn_rz,
                     wn_to,
                     wn_flz,
                     wn_fly,
                     wn_tr))

mn = np.concatenate((m_tx,
                     m_ty,
                     m_tz,
                     m_rx,
                     m_ry,
                     m_rz,
                     m_to,
                     m_flz,
                     m_fly,
                     m_tr))

cn = np.concatenate((c_tx,
                     c_ty,
                     c_tz,
                     c_rx,
                     c_ry,
                     c_rz,
                     c_to,
                     c_flz,
                     c_fly,
                     c_tr)) 

kn = np.concatenate((k_tx,
                     k_ty,
                     k_tz,
                     k_rx,
                     k_ry,
                     k_rz,
                     k_to,
                     k_flz,
                     k_fly,
                     k_tr))

phinx = np.concatenate((phinx_tx,
                        phinx_ty,
                        phinx_tz,
                        phinx_rx,
                        phinx_ry,
                        phinx_rz,
                        phinx_to,
                        phinx_flz,
                        phinx_fly,
                        phinx_tr))

phiny = np.concatenate((phiny_tx,
                        phiny_ty,
                        phiny_tz,
                        phiny_rx,
                        phiny_ry,
                        phiny_rz,
                        phiny_to,
                        phiny_flz,
                        phiny_fly,
                        phiny_tr))

phinz = np.concatenate((phinz_tx,
                        phinz_ty,
                        phinz_tz,
                        phinz_rx,
                        phinz_ry,
                        phinz_rz,
                        phinz_to,
                        phinz_flz,
                        phinz_fly,
                        phinz_tr))

Fx_fact = np.concatenate((Fx_fact_tx,
                          Fx_fact_ty,
                          Fx_fact_tz,
                          Fx_fact_rx,
                          Fx_fact_ry,
                          Fx_fact_rz,
                          Fx_fact_to,
                          Fx_fact_flz,
                          Fx_fact_fly,
                          Fx_fact_tr))

Fy_fact = np.concatenate((Fy_fact_tx,
                          Fy_fact_ty,
                          Fy_fact_tz,
                          Fy_fact_rx,
                          Fy_fact_ry,
                          Fy_fact_rz,
                          Fy_fact_to,
                          Fy_fact_flz,
                          Fy_fact_fly,
                          Fy_fact_tr))

Fz_fact = np.concatenate((Fz_fact_tx,
                          Fz_fact_ty,
                          Fz_fact_tz,
                          Fz_fact_rx,
                          Fz_fact_ry,
                          Fz_fact_rz,
                          Fz_fact_to,
                          Fz_fact_flz,
                          Fy_fact_fly,
                          Fz_fact_tr))

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
z_val   = z.min() 
idx_z   = z == z_val

phin_plot   = phin[mode,idx_z]
x_plot      = x[idx_z]
y_plot      = y[idx_z]
print('Plane at z/h = ' + str(z_val/h))

plt.figure()
plt.scatter(x_plot,y_plot,c=phin_plot)
plt.axis('equal')
plt.title("f = "+ str(np.round(fn[mode].real,1)) + " Hz")
plt.colorbar()

# %% Save modal basis

name = "Bridge_modal_basis_Dx_"+ f"{(np.round(Delta_x*100,3)):.3f}" +"_Dy_"+\
        f"{(np.round(Delta_y*100,3)):.3f}"+"_cm"

np.savez(name, x=x, y=y, z=z, wn=wn, mn=mn, kn=kn, cn=cn, phinx=phinx, 
         phiny=phiny, phinz=phinz, Fx_fact=Fx_fact, Fy_fact=Fy_fact, 
         Fz_fact=Fz_fact, params_names=params_names, params=params)

print("Saved as " + name)