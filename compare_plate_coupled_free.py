# -*- coding: utf-8 -*-

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
file = np.load("Data/" + name + ".npz" )
print("File " + name + ".npz" + " opened.")

#%%

coupled                 = True
null_null               = True

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
file_2 = np.load("Data/" + name + ".npz" )
print("File " + name + ".npz" + " opened.")

#%%

x_f           = file['x']
y_f           = file['y']
z_f           = file['z']
phinx_f       = file['phinx']
phiny_f       = file['phiny']
phinz_f       = file['phinz']
wn_f          = file['wn']
mn_f          = file["mn"]
cn_f          = file["cn"]
kn_f          = file["kn"]
Fx_fact_f     = file["Fx_fact"]
Fy_fact_f     = file["Fy_fact"]
Fz_fact_f     = file["Fz_fact"] 

x_c           = file_2['x']
y_c           = file_2['y']
z_c           = file_2['z']
phinx_c       = file_2['phinx']
phiny_c       = file_2['phiny']
phinz_c       = file_2['phinz']
wn_c          = file_2['wn']
mn_c          = file_2["mn"]
cn_c          = file_2["cn"]
kn_c          = file_2["kn"]
Fx_fact_c     = file_2["Fx_fact"]
Fy_fact_c     = file_2["Fy_fact"]
Fz_fact_c     = file_2["Fz_fact"] 

Nm_p = mn_f.size
N_p  = x_f.size


#%% Find the corresponding points

idx_c = np.zeros(N_p,dtype=int)
for i in range(N_p):
    idx_c[i] = np.argmin((x_c-x_f[i])**2 + 
                           (y_c-y_f[i])**2 + 
                           (z_c-z_f[i])**2)
x_c = x_c[idx_c]
y_c = y_c[idx_c]
z_c = z_c[idx_c]

phinx_c       = phinx_c[:,idx_c]
phiny_c       = phiny_c[:,idx_c]
phinz_c       = phinz_c[:,idx_c]

#%% Compute the MAC

MAC = np.zeros((Nm_p,Nm_p))

for i in range(Nm_p):
    for j in range(Nm_p):
        MAC_x = np.abs(phinx_f[i] @ phinx_c[j])**2 / \
                ((phinx_f[i] @ phinx_f[i]) * (phinx_c[j] @ phinx_c[j]))
        MAC_y = np.abs(phiny_f[i] @ phiny_c[j])**2 / \
                ((phiny_f[i] @ phiny_f[i]) * (phiny_c[j] @ phiny_c[j]))
        MAC_z = np.abs(phinz_f[i] @ phinz_c[j])**2 / \
                ((phinz_f[i] @ phinz_f[i]) * (phinz_c[j] @ phinz_c[j]))
        MAC[i,j] = np.sqrt(MAC_x**2 + MAC_y**2 + MAC_z**2) / np.sqrt(3)

#%% Plot the MAC-matrix

idx_mac = np.argmax(MAC,axis=1) 

plt.matshow(MAC, vmin=0, vmax=1)
plt.title("MAC-matrix")
plt.colorbar()

#%%

idx     = np.argmax(MAC, axis=0)
mask    = np.zeros((Nm_p,Nm_p))
for i in range(Nm_p):
    mask[i,idx[i]] = 1

MAC_alt = mask * MAC

plt.matshow(MAC_alt, vmin=0, vmax=1)
plt.title("Selected MAC-matrix")
plt.colorbar()