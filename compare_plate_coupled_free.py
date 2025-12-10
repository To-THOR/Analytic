# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from scipy.signal import ShortTimeFFT
import scipy.signal as sgn
from scipy.signal import find_peaks

#%%

dim3_coupling           = True
constraint_correction   = True

Dx = "4.100"
Dy = "3.500"


name = "Plate_modal_basis_Dx_"+Dx+"_Dy_"+Dy+"_cm"
name_2  = "Plaque_Chevalet_Zero_Dx_"+Dx+"_Dy_"+Dy+"_cm"+ dim3_coupling * "_3D_coupled" + constraint_correction * "_corrected" + "_ope_deform"
name_3  = "Plaque_Chevalet_Zero_Dx_"+Dx+"_Dy_"+Dy+"_cm"+ dim3_coupling * "_3D_coupled" + constraint_correction * "_corrected" + "_FRF"

file    = np.load(name + ".npz")
file_2  = np.load(name_2 + ".npz") 
file_3  = np.load(name_3 + ".npz") 

wn_mod      = file['wn']
m_mod       = file['mn']  
c_mod       = file['cn'] 
k_mod       = file['kn'] 
x_mod       = file['x']
y_mod       = file['y']
z_mod       = file['z']
X_mod       = file['phinx']
Y_mod       = file['phiny']
Z_mod       = file['phinz']
freq_mod    = wn_mod/(2*np.pi)
N_mod       = x_mod.size
Nm_mod      = wn_mod.size

x_ope       = file_2['x']
y_ope       = file_2['y']
z_ope       = file_2['z']
freq_ope    = file_2['freq']
X_ope       = file_2['X_op']
Y_ope       = file_2['Y_op']
Z_ope       = file_2['Z_op'] 
N_ope       = x_ope.size 
Nm_ope      = freq_ope.size

freq        = file_3["freq"]
sum_FRF     = file_3["sum_FRF"]
sum_FRFd    = file_3["sum_FRFd"]
sum_FRFdd   = file_3["sum_FRFdd"]
exc_FRF     = file_3["exc_FRF"]
exc_FRFd    = file_3["exc_FRFd"]
exc_FRFdd   = file_3["exc_FRFdd"]
x_exc       = file_3["x_exc"] 
y_exc       = file_3["y_exc"]
z_exc       = file_3["z_exc"]

om          = 2 * np.pi * freq 
idx_exc     = np.argmin((x_exc - x_mod)**2 + (z_exc - z_mod)**2 + 
                        (z_exc - z_mod)**2) 
modal_Q     = Z_mod[:,idx_exc][:,np.newaxis] / \
                     (-m_mod[:,np.newaxis] * om[np.newaxis]**2 + \
                      c_mod[:,np.newaxis] * 1j * om[np.newaxis] + \
                      k_mod[:,np.newaxis])
modal_sum_FRF   = (np.abs(Z_mod).sum(axis=1)[:,np.newaxis] * np.abs(modal_Q)).sum(axis=0)
modal_sum_FRFd  = np.abs(om)    * modal_sum_FRF
modal_sum_FRFdd = np.abs(om**2) * modal_sum_FRF
modal_exc_FRF   = (Z_mod[:,idx_exc][:,np.newaxis] * modal_Q).sum(axis=0)
modal_exc_FRFd  = 1j * om * modal_exc_FRF
modal_exc_FRFdd = -om**2 * modal_exc_FRF 

#%% Find the corresponding points

idx_ope = np.zeros(N_mod,dtype=int)
for i in range(N_mod):
    idx_ope[i] = np.argmin((x_ope-x_mod[i])**2 + 
                           (y_ope-y_mod[i])**2 + 
                           (z_ope-z_mod[i])**2)
x_ope = x_ope[idx_ope]
y_ope = y_ope[idx_ope]
z_ope = z_ope[idx_ope]
X_ope = X_ope[:,idx_ope]
Y_ope = Y_ope[:,idx_ope]
Z_ope = Z_ope[:,idx_ope]

#%% Compute the MAC

MAC = np.zeros((Nm_ope,Nm_mod))

for i in range(Nm_ope):
    for j in range(Nm_mod):
        Xope, Yope, Zope = X_ope[i], Y_ope[i], Z_ope[i]
        Xmod, Ymod, Zmod = X_mod[j], Y_mod[j], Z_mod[j]
        numer = np.abs((np.conjugate(Xmod) * Xope + \
                        np.conjugate(Ymod) * Yope + \
                        np.conjugate(Zmod) * Zope ).sum())**2
        denom = (np.abs(Xope)**2 + np.abs(Yope)**2 + np.abs(Zope)**2).sum() * \
                (np.abs(Xmod)**2 + np.abs(Ymod)**2 + np.abs(Zmod)**2).sum()
        MAC[i,j] = numer / denom
        
idx_mac = np.argmin(MAC,axis=1) 

#%%

idx = 0

z_scale     = 0.01
Z_ope_norm  = z_scale * np.abs(Z_ope[idx]) / np.abs(Z_ope[idx]).max()
Z_mod_norm  = z_scale * np.abs(Z_mod[idx_mac[idx]]) / np.abs(Z_mod[idx_mac[idx]]).max()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

scat = ax.scatter(x_ope, y_ope, z_ope, s=5, cmap=cm.winter, 
                  c=Z_ope_norm)
ax.set(xlim=(x_ope.min()*0.9, x_ope.max()*1.1), 
       ylim=(y_ope.min()*0.9, y_ope.max()*1.1), 
       zlim=(-z_scale, z_scale))
ax.set_aspect('equal')
ax.axis('off')


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

scat = ax.scatter(x_mod, y_mod, z_mod, s=5, cmap=cm.winter, 
                  c=Z_mod_norm)
ax.set(xlim=(x_mod.min()*0.9, x_mod.max()*1.1), 
       ylim=(y_mod.min()*0.9, y_mod.max()*1.1), 
       zlim=(-z_scale, z_scale))
ax.set_aspect('equal')
ax.axis('off')

#%% FRF compare

modal_exc_FRFdd_dB    = 20 * np.log10(np.abs(modal_exc_FRF) / np.abs(modal_exc_FRF).max()) 
exc_FRFdd_dB          = 20 * np.log10(np.abs(exc_FRF) / np.abs(exc_FRF).max())

plt.figure()
plt.plot(freq, modal_exc_FRFdd_dB, label="Plaque")
plt.plot(freq, exc_FRFdd_dB, label="Plaque + Chevalet")
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
