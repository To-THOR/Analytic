# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from scipy.io.wavfile import write

#%%

Dx = "4.100"
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
                dim3_coupling * "_3D_coupled" + (not modal) * constraint_correction * "_corrected" + ".npz"
    elif guitar:
        name = "Guitar_modal_basis.npz"
    else:
        name = system + "_modal_basis_Dx_" + Dx + "_Dy_" + Dy + "_cm.npz"
else:
     name = system + "_modal_basis_Dy_" + Dy + "_cm" + "_T_" + T + "_N_mu_" + mu + "_mg.m-1.npz"     

name = name
file = np.load("Data/" + name)
print("File " + name + " opened.")

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

phin = np.sqrt(phinx**2 + phiny**2 + phinz**2)

zscale  = 0.02
    
phinx = zscale * phinx  / np.abs(phin).max(axis=-1)[:,np.newaxis]
phiny = zscale * phiny / np.abs(phin).max(axis=-1)[:,np.newaxis]
phinz = zscale * phinz / np.abs(phin).max(axis=-1)[:,np.newaxis]
phin  = np.sqrt(phinx**2 + phiny**2 + phinz**2)

idx_mode    = 0
N           = 25

T   = 2 *np.pi 
t   = np.linspace(0,T,N)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

scat = ax.scatter(x, y, z, s=5, cmap=cm.winter, 
                  c=phin[idx_mode])
ax.set(xlim=(x.min()*0.9, x.max()*1.1), 
       ylim=(y.min()*0.9, y.max()*1.1), 
       zlim=(-zscale, zscale))
ax.set_aspect('equal')
ax.axis('off')

def update(frame):
    time_fact = np.sin(t[frame])
    scat._offsets3d = (x + time_fact * phinx[idx_mode],
                      y + time_fact * phiny[idx_mode],
                      z + time_fact * phinz[idx_mode])
    colors = time_fact * phinz[idx_mode]
    return scat

ani = animation.FuncAnimation(fig=fig, func=update, frames=N, interval=1)
#plt.show()

ani.save("Animations/"+name[:-4]+"_mode_"+str(idx_mode)+".gif", writer="pillow")
print("Frequency = "+ str(np.round(1/(2*np.pi)*np.sqrt(kn[idx_mode]/mn[idx_mode])))+" Hz")

#%%

if system == "String":
    
    f = np.arange(1,20e3,1)
    w = 2 * np.pi * f
    
    Qn = Fz_fact[:,-1][:,np.newaxis] / (-w[np.newaxis]**2 * mn[:,np.newaxis] + \
                        1j * w[np.newaxis] * cn[:,np.newaxis] + \
                            kn[:,np.newaxis])
    
    Z       = (phinz[:,-1][:,np.newaxis] * Qn).sum(axis=0)
    
    Z_dB    = 20*np.log10(np.abs(Z)/np.abs(Z).max()) 
    
    plt.figure()
    plt.plot(f, Z_dB)
    plt.xlabel("Fréquence (Hz)", size=20)
    plt.ylabel("Admittance verticale à l'extrémité libre (dB)", size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    
    w0 = np.sqrt(kn/mn)
    alphan = cn/(2*mn)
    Fs = 40000
    t = np.arange(0,7,1/Fs)
    qn_t     =  Fz_fact[:,-1][:,np.newaxis] / \
        ((w0[:,np.newaxis]+1e-9) * mn[:,np.newaxis]) * \
            np.exp(-alphan[:,np.newaxis]*t[np.newaxis]) * \
                np.sin(w0[:,np.newaxis] * t[np.newaxis])
    z_t       = (phinz[:,-1][:,np.newaxis] * qn_t).sum(axis=0)
    
    plt.figure()
    plt.plot(t, z_t/z_t.max())
    plt.xlabel("Temps (s)", size=20)
    plt.ylabel("Déplacement vertical normalisé", size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    
    z_t_norm = z_t/np.max(z_t)
    z_t_norm = z_t_norm * np.iinfo('int32').max
    z_t_norm = z_t_norm.astype(np.int32)
    z_t_norm = np.pad(z_t_norm, (Fs,0))
    write("Audio/String_test.wav", Fs, z_t_norm)
    
    plt.figure()
    plt.plot(z_t_norm)