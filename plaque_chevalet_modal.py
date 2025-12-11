# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.io.wavfile import write
from scipy.linalg import null_space
import matplotlib.cm as cm
import time

#%% Load modal bases

dim3_coupling   = True
null            = False
null_null       = False

Dx = "4.100"
Dy = "3.500"

name = "Plaque_Chevalet_"+ null * "Zero_"+ null_null * "Zero_Zero_"+ "Modal_Dx_"+Dx+"_Dy_"+Dy+"_cm"+ \
        dim3_coupling * "_3D_coupled" 

file = np.load("Data/Bridge_"+ null * "Zero_"+ null_null * "Zero_Zero_"+"modal_basis_Dx_"+Dx+"_Dy_"+Dy+"_cm.npz")

x_c             = file['x']
y_c             = file['y']
z_c             = file['z']
wn_c            = file['wn']
mn_c            = file['mn']
kn_c            = file['kn']
cn_c            = file['cn']
phinx_c         = file['phinx']
phiny_c         = file['phiny']
phinz_c         = file['phinz']
Fx_factor_c     = file['Fx_fact']
Fy_factor_c     = file['Fy_fact']
Fz_factor_c     = file['Fz_fact']
params_names_c  = file["params_names"]
params_c        = file['params']

Nm_c    = wn_c.size
N_c     = x_c.size

file = np.load("Data/Plate_modal_basis_Dx_"+Dx+"_Dy_"+Dy+"_cm.npz")

x_p             = file['x']
y_p             = file['y']
z_p             = file['z']
wn_p            = file['wn']
mn_p            = file['mn']
kn_p            = file['kn']
cn_p            = file['cn']
phinx_p         = file['phinx']
phiny_p         = file['phiny']
phinz_p         = file['phinz']
Fx_factor_p     = file['Fx_fact']
Fy_factor_p     = file['Fy_fact']
Fz_factor_p     = file['Fz_fact']
params_names_p  = file["params_names"]
params_p        = file['params']

Nm_p                = wn_p.size
N_p                 = x_p.size

Lx  = params_p[0].real
Ly  = params_p[1].real
h   = params_p[2].real
Lb  = params_c[0].real
wb  = params_c[1].real
hb  = params_c[2].real

x_c = x_c + (Lx - Lb) / 2
y_c = y_c + Ly*0.7
z_c = z_c + (h + hb) / 2

Nm  = Nm_p + Nm_c
N   = N_p + N_c

#%% Concatenate 

x                   = np.concatenate((x_c, x_p))
y                   = np.concatenate((y_c, y_p))
z                   = np.concatenate((z_c, z_p))
phix                = np.zeros((N, Nm))
phix[:N_c,:Nm_c]    = phinx_c.T 
phix[N_c:,Nm_c:]    = phinx_p.T
phiy                = np.zeros((N, Nm))
phiy[:N_c,:Nm_c]    = phiny_c.T
phiy[N_c:,Nm_c:]    = phiny_p.T
phiz                = np.zeros((N, Nm))
phiz[:N_c,:Nm_c]    = phinz_c.T
phiz[N_c:,Nm_c:]    = phinz_p.T
Fx_factor           = np.concatenate((np.pad(Fx_factor_c.T, ((0,N_p), (0,0))), 
                                      np.pad(Fx_factor_p.T, ((N_c,0),(0,0)))), 
                                     axis=1)
Fy_factor           = np.concatenate((np.pad(Fy_factor_c.T, ((0,N_p), (0,0))), 
                                      np.pad(Fy_factor_p.T, ((N_c,0),(0,0)))), 
                                     axis=1)
Fz_factor           = np.concatenate((np.pad(Fz_factor_c.T, ((0,N_p), (0,0))), 
                                      np.pad(Fz_factor_p.T, ((N_c,0),(0,0)))), 
                                     axis=1)
M                   = np.diag(np.concatenate((mn_c, mn_p)))
C                   = np.diag(np.concatenate((cn_c, cn_p)))
K                   = np.diag(np.concatenate((kn_c, kn_p)))

phi = np.sqrt(phix**2 + phiy**2 + phiz**2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scat = ax.scatter(x, y, z, s=5, cmap=cm.winter, 
                  c=phi[:,-1])
ax.set(xlim=(x.min()*0.9, x.max()*1.1), 
       ylim=(y.min()*0.9, y.max()*1.1), 
       zlim=(-0.1, 0.1))
ax.set_aspect('equal')
ax.axis('off')

#%% Compute coupling surface

idx_coupl           = np.zeros((0,2), dtype=int)

for i in range(N_c):
    d = (x[N_c:] - x[i])**2 + (y[N_c:] - y[i])**2 + (z[N_c:] - z[i])**2
    if np.sqrt(d.min()) < 1e-5:
        idx_coupl = np.concatenate((idx_coupl, 
                                    np.array((i, N_c+d.argmin()))[np.newaxis]), 
                                   axis=0)
     
#%% Graphical check

# color = np.zeros(N, dtype=bool)
# for i in idx_coupl.flatten():
#     color[i] = True 
    
color = np.abs(Fz_factor[:,0])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scat = ax.scatter(x, y, z, s=5, cmap=cm.winter, 
                  c=color)
ax.set(xlim=(x.min()*0.9, x.max()*1.1),
       ylim=(y.min()*0.9, y.max()*1.1),
       zlim=(-0.1, 0.1))
ax.set_aspect('equal')
ax.axis('off')

#%% Constraint matrix formualtion A qdd = 0

Ax          = phix[idx_coupl[:,0]] - phix[idx_coupl[:,1]]
Ay          = phiy[idx_coupl[:,0]] - phiy[idx_coupl[:,1]]
Az          = phiz[idx_coupl[:,0]] - phiz[idx_coupl[:,1]]

if dim3_coupling:
    A = np.concatenate((Ax,Ay,Az),axis=0)
else:
    A = Az

N_s = A.shape[0]

#%% Find subspace such that q = L alpha always satisfies A qdd = 0
#   Then extract the coupled basis

L = null_space(A)

# Restrain the plate modal coupling

# for i in range(Nm_c,Nm):
#     idx         = np.argmax(L[i])
#     max_val     = L[i,idx] 
#     L[i]        = np.zeros(Nm - N_s)
#     L[i,idx]    = max_val

# Modal basis computation

Nm_tilda = L.shape[-1]

M_tilda             = np.diag(L.T @ M @ L)
C_tilda             = np.diag(L.T @ C @ L)
K_tilda             = np.diag(L.T @ K @ L)
wn_tilda            = np.zeros(Nm_tilda)
wn_tilda[M_tilda!=0]= np.sqrt(K_tilda[M_tilda!=0] / M_tilda[M_tilda!=0])
phix_tilda          = phix @ L
phiy_tilda          = phiy @ L
phiz_tilda          = phiz @ L
Fx_factor_tilda     = Fx_factor @ L
Fy_factor_tilda     = Fy_factor @ L
Fz_factor_tilda     = Fz_factor @ L

#%% Sort by modal frequency

idx = np.argsort(wn_tilda)

M_tilda             = M_tilda[idx]
C_tilda             = C_tilda[idx]
K_tilda             = K_tilda[idx]
wn_tilda            = wn_tilda[idx]
phix_tilda          = phix_tilda[:,idx]
phiy_tilda          = phiy_tilda[:,idx]
phiz_tilda          = phiz_tilda[:,idx]
Fx_factor_tilda     = Fx_factor_tilda[:,idx]
Fy_factor_tilda     = Fy_factor_tilda[:,idx]
Fz_factor_tilda     = Fz_factor_tilda[:,idx]

#%% Phi normalization

phi_tilda = np.sqrt(phix_tilda**2 + phiy_tilda**2 + phiz_tilda**2)

fact        = np.ones(Nm_tilda)

# fact              = 1 / np.abs(phi_tilda).max(axis=0)
fact[M_tilda!=0]    = 1 / np.sqrt(M_tilda[M_tilda!=0])

fact                = fact[np.newaxis]

phix_tilda      = fact * phix_tilda
phiy_tilda      = fact * phiy_tilda
phiz_tilda      = fact * phiz_tilda
Fx_factor_tilda = fact * Fx_factor_tilda
Fy_factor_tilda = fact * Fy_factor_tilda
Fz_factor_tilda = fact * Fz_factor_tilda

fact            = fact.flatten()

M_tilda       = fact**2 * M_tilda
K_tilda       = fact**2 * K_tilda
C_tilda       = fact**2 * C_tilda

#%% Save results

np.savez("Data/"+name+".npz", N=N, Nm=Nm, 
         dim3_coupling=dim3_coupling, mn=M_tilda, kn=K_tilda, cn=C_tilda, 
         wn=wn_tilda, x=x, y=y, z=z,
         phinx=phix_tilda.T, phiny=phiy_tilda.T, phinz=phiz_tilda.T,
         Fx_fact=Fx_factor_tilda, Fy_fact=Fy_factor_tilda, 
         Fz_fact=Fz_factor_tilda)
print("Simulation saved as " + name + ".npz")