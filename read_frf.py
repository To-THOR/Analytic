# -*- coding: utf-8 -*-

#%%

import numpy as np
import matplotlib.pyplot as plt

#%%

Dx = "1.025"
Dy = "3.500"

T   = "70"
mu  = "380"

coupled                 = False
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
file = np.load("FRF/" + name + "_FRF_loc.npz" )
print("File " + name + "_FRF_loc.npz" + " opened.")

#%%

freq_1    = file['freq']
FRF_1     = file['FRF']

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
file = np.load("FRF/" + name + "_FRF_loc.npz" )
print("File " + name + "_FRF_loc.npz" + " opened.")

#%%

freq_2    = file['freq']
FRF_2     = file['FRF']

#%%

FRF_dB_1 = 20 * np.log10(np.abs(FRF_1) / np.abs(FRF_1).max())
FRF_dB_2 = 20 * np.log10(np.abs(FRF_2) / np.abs(FRF_2).max())

plt.figure()
plt.plot(freq_1, FRF_dB_1, label="Plaque seule")
plt.plot(freq_2, FRF_dB_2, label="Plaque et chevalet")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("FRF (dB)")
plt.legend()
