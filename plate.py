# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Unités SI

Lx      = 0.480
Ly      = 0.420
h       = 0.00319
rho     = 2700
eta     = 0.004
Ex      = 70e9 * (1 + 1j * eta)
Ey      = 70e9 * (1 + 1j * eta)
nuxy    = 0.3
nuyx    = 0.3
Gxy     = Ex / (2 * (1 + nuxy))

Lb      = 0.205 # Bridge length
wb      = 0.035 # Bridge width

D1 = Ex *           h**3 / (12  * (1 - nuxy * nuyx)) 
D2 = Ex * nuyx *    h**3 / (6   * (1 - nuxy * nuyx))
D3 = Ey *           h**3 / (12  * (1 - nuxy * nuyx))
D4 = Gxy * h**3 / 3

params_names = np.array(("Lx",
                        "Ly",
                        "h",
                        "rho",
                        "eta",
                        "Ex",
                        "Ey",
                        "nuxy",
                        "nuyx",
                        "Gxy",
                        "Lb",
                        "wb"))

params = np.array((Lx,
                   Ly,
                   h,
                   rho,
                   eta,
                   Ex,
                   Ey,
                   nuxy,
                   nuyx,
                   Gxy,
                   Lb,
                   wb))

Delta_x = 30e-3
Delta_y = 10e-3

f_max           = 5000
# w_max       = 2 * np.pi * f_max
kappa_max_x     = 2 * np.pi / Delta_x
kappa_max_y     = 2 * np.pi / Delta_y

xb = (Lx-Lb)/2
yb = Ly*0.7-wb/2

Nx = int(np.ceil(Lb / Delta_x))
Ny = int(np.ceil(wb / Delta_y))

x = np.linspace(0, Lb, Nx) + xb
y = np.linspace(0, wb, Ny) + yb
z = np.array((h/2))
# z = np.linspace(h/2, -h/2, 3)

Delta_x = x[1] - x[0]
Delta_y = y[1] - y[0]

X,Y,Z = np.meshgrid(x,y,z)

x = X.flatten()
y = Y.flatten()
z = Z.flatten()

x = np.insert(x, 0, Lx*0.1)
y = np.insert(y, 0, Ly*0.1)
z = np.insert(z, 0, h/2)

Nx_other = 30
Ny_other = 30

x_other = np.linspace(0, Lx, Nx_other)
y_other = np.linspace(0, Ly, Ny_other)
z_other = np.array(0)

Nz_other = z_other.size

X,Y,Z = np.meshgrid(x_other,y_other,z_other)

x_other = X.flatten()
y_other = Y.flatten()
z_other = Z.flatten()

x = np.concatenate((x,x_other))
y = np.concatenate((y,y_other))
z = np.concatenate((z,z_other))

Np = x.size

#%% Flexion

m_max   = int(np.ceil(kappa_max_x * Lx / np.pi))
n_max   = int(np.ceil(kappa_max_y * Ly / np.pi))
m, n    = np.arange(1,20), np.arange(1,20)
M, N    = np.meshgrid(m,n)
m,n     = M.flatten(), N.flatten()
wmn     = np.pi**2 * np.sqrt(1/(rho * h)) *\
    np.sqrt(D1 * m**4 / Lx**4 + 
            D3 * n**4/Ly**4 + 
            (D2 + D4) * m**2 * n**2 / (Lx**2 * Ly**2))

# idx = wmn/(2*np.pi) < f_max
# wmn = wmn[idx]
# m,n = m[idx], n[idx]

idx = wmn.argsort()
wmn = wmn[idx]
m,n = m[idx], n[idx]

fmn = wmn / (2 * np.pi)

Nm = m.size

phimnx = np.zeros((Nm, Np))
phimny = np.zeros((Nm, Np))
phimnz = np.zeros((Nm, Np))

for i in range(Nm):
    phimnz[i] = np.sin(m[i] * np.pi * x / Lx) * np.sin(n[i] * np.pi * y / Ly)
    phimnx[i] = -z * m[i] * np.pi / Lx * phimnz[i]
    phimny[i] = -z * n[i] * np.pi / Ly * phimnz[i]
    
phimn = np.sqrt(phimnx**2 + phimny**2 + phimnz**2)

mmn     = np.ones(Nm) * rho * h * 1/4 * Lx * Ly
kmn     = np.ones(Nm) * (D1*(m*np.pi/Lx)**4 + D3*(n*np.pi/Ly)**4 +
                         (D2+D4)*(m*np.pi/Lx)**2*(n*np.pi/Ly)**2) * \
                          1/4 * Lx * Ly * (1 + 1j * eta)
cmn                     = np.imag(kmn) / np.real(wmn)
kmn                     = np.real(kmn)
Fx_fact                 = np.zeros((Nm,Np))
Fy_fact                 = np.zeros((Nm,Np))
Fz_fact                 = np.zeros((Nm,Np))
Fz_fact[:,z==z.max()]   = phimnz[:,z==z.max()]

# Normalization

fact        = 1 / np.sqrt(mmn)[:,np.newaxis]

phinx       = fact * phimnx
phiny       = fact * phimny
phinz       = fact * phimnz
Fx_fact     = fact * Fx_fact
Fy_fact     = fact * Fy_fact
Fz_fact     = fact * Fz_fact

fact        = fact.flatten()

mmn         = fact**2 * mmn
kmn         = fact**2 * kmn
cmn         = fact**2 * cmn

#%% Plot box

mode    = 9
z_val   = z.max()
idx_z   = z == z_val

phimn_plot   = phimn[mode,idx_z]
x_plot      = x[idx_z]
y_plot      = y[idx_z]
# print('Plane at z/h = ' + str(z[idx_z]/h))

plt.figure()
plt.scatter(x_plot,y_plot,c=phimn_plot)
plt.axis('equal')
plt.xlim((0,Lx))
plt.ylim((0,Ly))
plt.title("f = "+ str(np.round(fmn[mode].real,1)) + " Hz")
plt.colorbar()

#%%

# for i in range(fmn.size):
#     print(m[i],'\t',n[i],' \t',np.round(fmn[i],1))
    
# %% Save modal basis

name = "Data/Plate_modal_basis_Dx_"+ f"{(np.round(Delta_x*100,3)):.3f}" +"_Dy_"+\
        f"{(np.round(Delta_y*100,3)):.3f}"+"_cm"
        
np.savez(name, x=x, y=y, z=z, wn=wmn, mn=mmn, kn=kmn, cn=cmn, phinx=phimnx, 
         phiny=phimny, phinz=phimnz, Fx_fact=Fx_fact, Fy_fact=Fy_fact, 
         Fz_fact=Fz_fact, params_names=params_names, params=params)

print("Saved as " + name)