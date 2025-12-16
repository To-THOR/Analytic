# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


#%%

Dx = "1.025"
Dy = "3.500"

T   = "70"
mu  = "380"

coupled                 = True
guitar                  = False
system                  = "Plate"
modal                   = True
null                    = False
null_null               = False
dim3_coupling           = True
constraint_correction   = False

if system != "String":
    if coupled:
        name = "Plaque_Chevalet" + null * "_Zero" + null_null * "_Zero_Zero"  + + modal * "_Modal" + "_Dx_" + Dx + "_Dy_" + Dy + "_cm"+ \
                dim3_coupling * "_3D_coupled" + (not modal) * constraint_correction * "_corrected"
    elif guitar:
        name = "Guitar_modal_basis"
    else:
        name = system + "_modal_basis_Dx_" + Dx + "_Dy_" + Dy + "_cm"
else:
     name = system + "_modal_basis_Dy_" + Dy + "_cm" + "_T_" + T + "_N_mu_" + mu + "_mg.m-1"     

name = name
file = np.load("Data/" + name + ".npz" )
print("File " + name + ".npz" + " opened.")

x           = file['x']
y           = file['y']
z           = file['z']
phinx       = file['phinx']
phiny       = file['phiny']
phinz       = file['phinz']
wn          = file['wn']
mn          = file["mn"]
cn          = file["cn"]
kn          = file["kn"]
Fx_fact     = file["Fx_fact"]
Fy_fact     = file["Fy_fact"]
Fz_fact     = file["Fz_fact"] 

phin    = np.sqrt(phinx**2 + phiny**2 + phinz**2)

Nm      = mn.size

#%%

Lx  = x.max() - x.min()
Ly  = y.max() - y.min()

idx_exc = np.argmin((x - Lx * 0.1)**2 + (y - Ly * 0.1)**2)

Fe      = 5e3
Te      = 1/Fe
df      = 0.5
freq    = np.arange(0,Fe,df)
w       = 2*np.pi*freq
Nf      = freq.size 
Q       = Fz_fact[:,idx_exc] / (-mn[np.newaxis] * w[:,np.newaxis]**2 + \
                                   1j * w[:,np.newaxis] * cn[np.newaxis] + \
                                   kn[np.newaxis])
    
FRF_z   = (phinz[:,idx_exc][np.newaxis] * Q).sum(axis=1)
 
#%%

FRF_z_dB = 20 * np.log10(np.abs(FRF_z) / np.abs(FRF_z).max())

plt.figure()
plt.plot(freq, FRF_z_dB)
plt.xlabel("Fréquence (Hz)")
plt.ylabel("FRF (dB)")

#%%

np.savez("FRF/"+name+"_FRF_loc.npz", freq=freq, FRF=FRF_z)
print("Saved as FRF/"+name+"_FRF_loc.npz")