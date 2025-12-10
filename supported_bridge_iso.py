# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as  mv

# Unités SI

L       = 0.205
w       = 0.035
h       = 0.010
rho     = 2700
beta    = 1/16 * (16 / 3 - 3.36 * h / w * (1 - 1/12 * h**4 / w**4))
K       = beta * h**3 * w
E       = 70e9
nu      = 0.3
G       = E/(2 * (1 + nu))
I       = w * h**3 / 12
J       = (w * h**3 + h * w**3) / 12

f_max = 15000
f_res = 1e-8
w_max = 2 * np.pi * f_max
w_res = 2 * np.pi * f_res

Delta_x = 1e-3
Delta_y = 1e-3
Delta_z = 1e-3

x = np.arange(0, L, Delta_x)
y = np.arange(-w/2, w/2, Delta_y)
z = np.arange(h/2, -h/2, -Delta_z)

X,Y,Z = np.meshgrid(x,y,z)

Nx, Ny, Nz = X.shape

#%% Flexion

n       = np.arange(1, L/np.pi*np.sqrt(w_max)*(rho*h*w/(E*I))**(1/4))
kappan  = n * np.pi / L

wn_fl           = np.sqrt(E * I / (rho * h * w)) * kappan**2 
fn_fl           = wn_fl / (2 * np.pi)

N_fl = wn_fl.size

phinx_fl = np.zeros((N_fl,Nx,Ny,Nz))
phiny_fl = np.zeros((N_fl,Nx,Ny,Nz))
phinz_fl = np.zeros((N_fl,Nx,Ny,Nz))

for i,kap in enumerate(kappan):
    phinx_fl[i]    = - Z * kap *  np.cos(kap * X)
    phinz_fl[i]    = np.sin(kap * X)

phin_fl = np.sqrt(phinx_fl**2 + phiny_fl**2 + phinz_fl**2)


#%% Torsion
 
n       = np.arange(1, np.sqrt(rho * J/ (G * K)) * L / np.pi * w_max)
N_to    = int(n[-1])
wn_to   = np.sqrt(G * K / (rho * J)) * n * np.pi / L 

phinx_to = np.zeros((N_to, Nx, Ny, Nz))
phiny_to = np.zeros((N_to, Nx, Ny, Nz))
phinz_to = np.zeros((N_to, Nx, Ny, Nz))

for i,nn in enumerate(n): 
    thetan  = np.sin(nn * np.pi / L * X) 
    phiny_to[i] = -Z * thetan
    phinz_to[i] = Y * thetan

phin_to = np.sqrt(phinx_to**2 + phiny_to**2 + phinz_to**2)


#%% Base modale unifiée

wn = np.concatenate((wn_to,
                     wn_fl))

phinx = np.concatenate((phinx_to,
                        phinx_fl))

phiny = np.concatenate((phiny_to,
                        phiny_fl))

phinz = np.concatenate((phinz_to,
                        phinz_fl))

idx_sort = wn.argsort()

wn      = wn[idx_sort]
phinx   = phinx[idx_sort]
phiny   = phiny[idx_sort]
phinz   = phinz[idx_sort]

phin    = np.sqrt(phinx**2 + phiny**2 + phinz**2) 

fn      =  wn / (2 * np.pi) 

#%% Plot box

mode    = 6
idx_z   = 0 

phin_plot   = phin[mode,:,:,idx_z]
X_plot      = X[:,:,idx_z]
Y_plot      = Y[:,:,idx_z]
print('Plane at z/h = ' + str(z[idx_z]/h))

plt.figure()
plt.pcolor(X_plot,Y_plot,phin_plot)
plt.axis('equal')
plt.title("f = "+ str(np.round(fn[mode],1)) + " Hz")
plt.colorbar()
