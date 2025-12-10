# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

Dx = "0.5"
Dy = "0.5"
vert = True
name = "Plate_modal_basis_Dx_" + Dx + "_Dy_" + Dy + vert * "_cm_vert.npz"
file = np.load(name)

X           = file['X']
Y           = file['Y']
Z           = file['Z']
wn          = file['wn']
mn          = file['mn']
kn          = file['kn']
cn          = file['cn']
phinx       = file['phinx']
phiny       = file['phiny']
phinz       = file['phinz']
F_factor    = file['F_factor']

Nm          = wn.size
Ny, Nx, Nz  = X.shape
Dx          = np.abs(X[0,1,0] - X[0,0,0]) 
Dy          = np.abs(Y[1,0,0] - Y[0,0,0])
Dz          = np.abs(Z[0,0,1] - Z[0,0,0])

fmax    = 5e3 
df      = 1 
f       = np.arange(10,fmax,df) 
w       = 2 * np.pi * f 
Nf      = f.size 

if name[:5] == "Plate":
    pos_ext         = np.array([X.max()*0.5, Y.max()*0.6])
    idx_ext         = (int(np.round(pos_ext[1]/Dy)), 
                       int(np.round(pos_ext[0]/Dx)))
else:
    pos_ext         = np.array([0, 0])
    idx_ext         = (int(np.round(pos_ext[1]/Dy)), 
                       int(np.round(pos_ext[0]/Dx)))
f_ext           = np.zeros((Ny,Nx))
f_ext[idx_ext]  = 1
idx_z           = Z.argmax() 

FRF = np.zeros(Nf)

for n in range(Nm):
    F_exc   = np.sum(F_factor[n] * f_ext) * Dx * Dy * phinz[n,idx_ext[0],idx_ext[1],idx_z] 
    FRF     = FRF +  F_exc / (-w**2 * mn[n] + w * 1j * cn[n]+ kn[n])


FRF_dB = 20 * np.log10(np.abs(FRF) / np.abs(FRF).max()) 
plt.figure()
plt.plot(f, FRF_dB)
plt.xlabel("Fréquence (Hz)", fontsize=20)
plt.ylabel("FRF (dB)", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()