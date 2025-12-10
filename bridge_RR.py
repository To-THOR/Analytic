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
Ex              = 13.3e9 * (1 + 1j * eta)
Ey              = 1.7e9  * (1 + 1j * eta)
nux             = 0.38
nuy             = 0.38 
G               = 0.93e9 * (1 + 1j * eta)
I               = w * h**3 / 12
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
                   Ex,
                   Ey,
                   G,
                   nux,
                   nuy,
                   fric))

Delta_x = 40e-3
Delta_y = 15e-3

f_max       = 5000
f_res       = 1e-8
w_res       = 2 * np.pi * f_res
kappax_max  = 4 * np.pi / Delta_x
kappay_max  = 4 * np.pi / Delta_y

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

Nx_other = 15
Ny_other = 5
Nz_other = 2

x_other   = np.linspace(0, L, Nx_other)
y_other   = np.linspace(-w/2, w/2, Ny_other)
z_other   = np.linspace(0, h/2, Nz_other) 

X,Y,Z = np.meshgrid(x_other,y_other,z_other)

x_other = X.flatten()
y_other = Y.flatten()
z_other = Z.flatten()

x = np.concatenate((x,x_other))
y = np.concatenate((y,y_other))
z = np.concatenate((z,z_other))

Np          = Nx*Ny*Nz + Nx_other*Ny_other*Nz_other 

#%% Flexion

kappanL          = np.insert(np.load("beam_kappaL.npy"),0,0)
kappanx          = kappanL / L
kappany          = kappanL / w
kappanx          = kappanx[kappanx < kappax_max]
kappany          = kappany[kappany < kappay_max]

Nx_fl       = kappanx.size
Ny_fl       = kappany.size
N_fl        = Nx_fl * Ny_fl - 1

m,n         = np.arange(Nx_fl), np.arange(Ny_fl)
m,n         = np.meshgrid(m,n)
m,n         = m.flatten()[1:], n.flatten()[1:]

xphinx_fl   = np.zeros((Nx_fl,Np))
xphiny_fl   = np.ones((Nx_fl,Np))
xphinz_fl   = np.ones((Nx_fl,Np))

yphinx_fl   = np.ones((Ny_fl,Np))
yphiny_fl   = np.zeros((Ny_fl,Np))
yphinz_fl   = np.ones((Ny_fl,Np))

kapx = kappanx[1]
alphan          = (np.sin(kapx * L) - np.sinh(kapx * L)) / \
                    (np.cosh(kapx * L) - np.cos(kapx * L))
xphinx_fl[1]     = - z * kapx * \
    ( np.cosh(kapx * x) + np.cos(kapx * x) + \
                     alphan * ( np.sinh(kapx * x) - np.sin(kapx * x) ))
xphinz_fl[1]     = np.sinh(kapx * x) + np.sin(kapx * x) + \
    alphan * (np.cosh(kapx * x) + np.cos(kapx * x))
    
kapy = kappany[1]
alphan          = (np.sin(kapy * w) - np.sinh(kapy * w)) / \
                    (np.cosh(kapy * w) - np.cos(kapy * w))
yphiny_fl[1]     = - z * kapy * \
    ( np.cosh(kapy * (y+w/2)) + np.cos(kapy * (y+w/2)) + \
                     alphan * ( np.sinh(kapy * (y+w/2)) - np.sin(kapy * (y+w/2)) ))
yphinz_fl[1]     = np.sinh(kapy * (y+w/2)) + np.sin(kapy * (y+w/2)) + \
    alphan * (np.cosh(kapy * (y+w/2)) + np.cos(kapy * (y+w/2)))

for i,kapx in enumerate(kappanx[2:],2):
    xphinx_fl[i]    = - z * kapx * (np.exp(-kapx*x) + \
                                  np.exp(kapx*(x-L)) * np.sin(kapx * L) +\
                                  np.sin(kapx * x) + np.cos(kapx * x))
    xphinz_fl[i]    = -np.exp(-kapx*x) + np.exp(kapx*(x-L)) * np.sin(kapx*L) + \
                        np.sin(kapx*x) - np.cos(kapx*x)

for i,kapy in enumerate(kappany[2:],2):
    yphiny_fl[i]    = - z * kapy * (np.exp(-kapy*(y+w/2)) + \
                                  np.exp(kapy*(y-w/2)) * np.sin(kapy * w) +\
                                  np.sin(kapy * (y+w/2)) + np.cos(kapy * (y+w/2)))
    yphinz_fl[i]    = -np.exp(-kapy*(y+w/2)) + np.exp(kapy*(y-w/2)) * np.sin(kapy*w) + \
                        np.sin(kapy*(y+w/2)) - np.cos(kapy*(y+w/2))

phinx_fl = xphinx_fl[m] * yphinx_fl[n]
phiny_fl = xphiny_fl[m] * yphiny_fl[n]
phinz_fl = xphinz_fl[m] * yphinz_fl[n]

phin_fl = np.sqrt(phinx_fl**2 + phiny_fl**2 + phinz_fl**2)


Intx_fl     = np.zeros(Nx_fl)
Inty_fl     = np.zeros(Ny_fl)
dIntx_fl    = np.zeros(Nx_fl)
dInty_fl    = np.zeros(Ny_fl)
ddIntx_fl   = np.zeros(Nx_fl)
ddInty_fl   = np.zeros(Ny_fl)
oddIntx_fl  = np.zeros(Nx_fl)
oddInty_fl  = np.zeros(Ny_fl)

file        = np.load("Integral_beam_mode_0.npz")

Intx_fl[0]  = L
Intx_fl[1]  = 1/kappanx[1] * file["I_phi_phi"] 
Intx_fl[2:] = L
    
Inty_fl[0]   = w 
Inty_fl[1]   = 1/kappany[1] * file["I_phi_phi"] 
Inty_fl[2:]  = w

dIntx_fl[1]     = kappanx[1] * file["I_dphi_dphi"]
dIntx_fl[2:]    = kappanx[2:] * (6 + L * kappanx[2:])

dInty_fl[1]     = kappany[1] * file["I_dphi_dphi"]
dInty_fl[2:]    = kappany[2:] * (6 + w * kappany[2:])

ddIntx_fl[1]    = kappanx[1]**3 * file["I_ddphi_ddphi"] 
ddIntx_fl[2:]   = kappanx[2:]**4 * L

ddInty_fl[1]    = kappany[1]**3 * file["I_ddphi_ddphi"]
ddInty_fl[2:]   = kappany[2:]**4 * w

oddIntx_fl[1]   = kappanx[1] * file["I_phi_ddphi"]
oddIntx_fl[2:]  = kappanx[2:] * (2 - L * kappanx[2:])

oddInty_fl[1]   = kappany[1] * file["I_phi_ddphi"]
oddInty_fl[2:]  = kappany[2:] * (2 - w * kappany[2:])

Ex_star = Ex * h**3 / 12 * (1 - nux * nuy)
Ey_star = Ey * h**3 / 12 * (1 - nux * nuy)
G_star  = G * h**3 / 12
H0      = Ey_star * nux + 2 * G_star

m_fl                        = rho * h * Intx_fl[m] * Inty_fl[n]
k_fl                        = H0 / L**4 * (\
    Ex_star * ddIntx_fl[m] * Inty_fl[n] + \
    Ey_star * nux * (L/w)**2 * 2 * oddIntx_fl[m] * oddInty_fl[n] + \
    2 * (H0 - nux * Ey_star) * (L/w)**2 * dIntx_fl[m] * dInty_fl[n] +\
    Ey_star * (L/w)**4 * Intx_fl[m] * ddInty_fl[n])
wn_fl                       = np.sqrt(k_fl / m_fl) 
c_fl                        = np.imag(k_fl) / np.real(wn_fl)
k_fl                        = np.real(k_fl)
wn_fl                       = np.sqrt(k_fl / m_fl) 
Fx_fact_fl                  = np.zeros((N_fl,Np))
Fy_fact_fl                  = np.zeros((N_fl,Np))
Fz_fact_fl                  = np.zeros((N_fl,Np))
Fz_fact_fl[:,z==z.max()]    = phinz_fl[:,z==z.max()]
Fz_fact_fl[:,z==z.min()]    = phinz_fl[:,z==z.min()]


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

# Rotation en y

phinx_ry    = z[np.newaxis]
phiny_ry    = np.zeros((1,Np))
phinz_ry    = -x[np.newaxis]
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

# Rotation en z

phinx_rz    = -y[np.newaxis]
phiny_rz    = x[np.newaxis]
phinz_rz    = np.zeros((1,Np))
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

#%% Base modale unifiée

wn = np.concatenate((wn_tx,
                     wn_ty,
                     wn_tz,
                     wn_rx,
                     wn_ry,
                     wn_rz,
                     wn_fl))

mn = np.concatenate((m_tx,
                     m_ty,
                     m_tz,
                     m_rx,
                     m_ry,
                     m_rz,
                     m_fl))

cn = np.concatenate((c_tx,
                     c_ty,
                     c_tz,
                     c_rx,
                     c_ry,
                     c_rz,
                     c_fl)) 

kn = np.concatenate((k_tx,
                     k_ty,
                     k_tz,
                     k_rx,
                     k_ry,
                     k_rz,
                     k_fl))

phinx = np.concatenate((phinx_tx,
                        phinx_ty,
                        phinx_tz,
                        phinx_rx,
                        phinx_ry,
                        phinx_rz,
                        phinx_fl))

phiny = np.concatenate((phiny_tx,
                        phiny_ty,
                        phiny_tz,
                        phiny_rx,
                        phiny_ry,
                        phiny_rz,
                        phiny_fl))

phinz = np.concatenate((phinz_tx,
                        phinz_ty,
                        phinz_tz,
                        phinz_rx,
                        phinz_ry,
                        phinz_rz,
                        phinz_fl))

Fx_fact = np.concatenate((Fx_fact_tx,
                          Fx_fact_ty,
                          Fx_fact_tz,
                          Fx_fact_rx,
                          Fx_fact_ry,
                          Fx_fact_rz,
                          Fx_fact_fl))

Fy_fact = np.concatenate((Fy_fact_tx,
                          Fy_fact_ty,
                          Fy_fact_tz,
                          Fy_fact_rx,
                          Fy_fact_ry,
                          Fy_fact_rz,
                          Fy_fact_fl))

Fz_fact = np.concatenate((Fz_fact_tx,
                          Fz_fact_ty,
                          Fz_fact_tz,
                          Fz_fact_rx,
                          Fz_fact_ry,
                          Fz_fact_rz,
                          Fz_fact_fl))

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

name = "BridgeRR_modal_basis_Dx_"+ f"{(np.round(Delta_x*100,3)):.3f}" +"_Dy_"+\
        f"{(np.round(Delta_y*100,3)):.3f}"+"_cm"

np.savez(name, x=x, y=y, z=z, wn=wn, mn=mn, kn=kn, cn=cn, phinx=phinx, 
         phiny=phiny, phinz=phinz, Fx_fact=Fx_fact, Fy_fact=Fy_fact, 
         Fz_fact=Fz_fact, params_names=params_names, params=params)

print("Saved as " + name)