# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# -------------- Fonctions --------------

def reshape_3D(a : np.ndarray, Nx, Ny, Nz):
    b = np.empty((a.shape[0], Nz, Ny, Nx))
    for i in range(Nz):
        b[:,i] = a[:,::Nz].reshape((a.shape[0], Ny, Nx)) 
    return b

# -------------- Paramètres matériaux --------------

# Plaque

pl = {
      "Lx"  : 480e-3,
      "Ly"  : 420e-3,
      "h"   : 3.19e-3,
      "rho" : 2740,
      "E"   : 70e9,
      "D"   : 210.5,
      "eta" : 0.4e-2,
      } 

# Chevalets

# Palissandre scallopé
pc = {
      "L"   : 172e-3,
      "w"   : 41e-3,
      "h"   : 10e-3,
      "rho" : 859,
      "E"   : 13300e6,
      "G"   : 812e6,
      "eta" : 1e-2,
      }

# Palissandre non scallopé
ps = {
      "L"   : 205e-3,
      "w"   : 34e-3,
      "h"   : 10e-3,
      "rho" : 859,
      "E"   : 13300e6,
      "G"   : 812e6,
      "eta" : 1e-2,
      }

# Ebène scallopé
ec = {
      "L"   : 154e-3,
      "w"   : 38e-3,
      "h"   : 4e-3,
      "rho" : 1207,
      "E"   : 23600e6,
      "G"   : 1880e6,
      "eta" : 1e-2,
      }

# Ebène non scallopé
es = {
      "L"   : 183e-3,
      "w"   : 35e-3,
      "h"   : 10e-3,
      "rho" : 1207,
      "E"   : 23600e6,
      "G"   : 1880e6,
      "eta" : 1e-2,
      }

# -------------- Géométrie --------------

Delta   = 1e-3
Nz      = 3
xc      = pl['Lx']/2
yc      = pl['Ly']*0.6

# -------------- Autres -------------- 

kappa_lim = np.pi/(5*Delta) 

# -------------- Initialisation --------------

br = ps 

# -------------- Maillage --------------


x = np.arange(0,pl["Lx"],Delta)
x = x + (pl["Lx"] - x[-1]) / 2
y = np.arange(0,pl["Ly"],Delta)
y = y + (pl["Ly"] - y[-1]) / 2
z = np.linspace(-1,1,Nz)

pl['Nx'], pl['Ny'], pl['Nz'] = x.size, y.size, z.size
br['Nz']                     = z.size

X,Y,Z = np.meshgrid(x,y,z)
x,y,z = X.flatten(), Y.flatten(), Z.flatten()

pl["x"], pl["y"], pl["z"] = x[np.newaxis], y[np.newaxis], pl['h']/2*z[np.newaxis]
pl["Delta"] = Delta
pl["Delta_z"] = pl['z'].flatten()[1] - pl['z'].flatten()[0]

idx_x = np.abs(x - xc) <= br["L"]/2
idx_y = np.abs(y - yc) <= br["w"]/2

idx_br = np.logical_and(idx_x, idx_y)

pl['idx_br'] = np.where(idx_br)

br['x']             = x[np.newaxis,idx_br] - xc + br['L']/2
br['y']             = y[np.newaxis,idx_br] - yc 
br['Nx'], br['Ny']  = np.unique(br['x']).size, np.unique(br['y']).size
br['z']             = br['h']/2*z[np.newaxis, idx_br]

br["Delta"] = Delta
br["Delta_z"] = br["z"].flatten()[1] - br["z"].flatten()[0]

#%%
# -------------- Modes de plaque -------------- 

m   = np.arange(1,15)
n   = np.arange(1,15)
M,N = np.meshgrid(m,n) 
m,n = M.flatten(), N.flatten()

kappa   =  np.sqrt(np.pi**2 * ( (m/pl['Lx'])**2 + (n/pl['Ly'])**2 ))
idx     = kappa < kappa_lim 

kappa   = kappa[idx]
m       = m[idx] 
n       = n[idx]

idx = np.argsort(kappa)

kappa   = kappa[idx]
m       = m[idx,np.newaxis] 
n       = n[idx,np.newaxis]

pl['Nm'] = n.size

om = np.real(np.sqrt(pl['D']*(1+1j*pl['eta']) / \
                              (pl['rho']*pl['h'])))*kappa**2
    
pl['om'] = om.flatten()
    
phi_x = -pl['z'] * m * np.pi / pl['Lx'] * \
    np.cos(m*np.pi*pl['x']/pl['Lx']) * np.sin(n*np.pi*pl['y']/pl['Ly'])
    
phi_y = -pl['z'] * n * np.pi / pl['Ly'] * \
    np.sin(m*np.pi*pl['x']/pl['Lx']) * np.cos(n*np.pi*pl['y']/pl['Ly'])
    
phi_z = np.sin(m*np.pi*pl['x']/pl['Lx']) * np.sin(n*np.pi*pl['y']/pl['Ly'])
    
pl['phi_x'] = phi_x
pl['phi_y'] = phi_y
pl['phi_z'] = phi_z

plt.figure()
plt.pcolormesh(reshape_3D(phi_x, pl['Nx'], pl['Ny'], pl['Nz'])[2,1])
plt.colorbar()
plt.axis('equal')

phi_norm2 = reshape_3D(phi_x, pl['Nx'], pl['Ny'], pl['Nz'])**2 + \
            reshape_3D(phi_y, pl['Nx'], pl['Ny'], pl['Nz'])**2 + \
            reshape_3D(phi_z, pl['Nx'], pl['Ny'], pl['Nz'])**2

m = np.trapz(phi_norm2, axis=-1, dx=pl['Delta']) 
m = np.trapz(m, axis=-1, dx=pl['Delta'])
m = pl['rho'] * np.trapz(m, axis=-1, dx=pl['Delta_z']) 

pl['M'] = np.diag(m) 
pl['K'] = pl['M'] * pl['om']**2
pl['C'] = pl['M'] * pl['om'] * pl['eta']

#%%

# -------------- Modes de flexion normal --------------

kappa   = np.linspace(2/br['L'],kappa_lim,100000)
f       = np.cos(kappa * br['L']) * np.cosh(kappa * br['L'])
sf      = np.sign(f)
dsf     = sf[:-1] - sf[1:]
idx     = np.where(dsf != 0)[0] 
zeros   = (f[idx+1]*kappa[idx] - f[idx]*kappa[idx+1])/(f[idx+1] - f[idx])
kappa   = zeros[:,np.newaxis] 

br['Nm'] = kappa.size

idx_no_approx_zero = kappa*br['L']<20

coeff = (np.sin(kappa * br['L']) - np.sinh(kappa * br['L'])) / \
    (np.cosh(kappa * br['L']) - np.cos(kappa * br['L']))  
phi_x = -br['z'] * kappa *\
    (idx_no_approx_zero * np.cosh(kappa * br['x']) + np.cos(kappa * br['x']) + \
        coeff * (idx_no_approx_zero * np.sinh(kappa * br['x']) - \
                 np.sin(kappa * br['x']))) 
phi_y = np.zeros((kappa.size, br['Nx']*br['Ny']*br['Nz'])) 
phi_z = (idx_no_approx_zero * np.sinh(kappa * br['x']) + np.sin(kappa * br['x'])) + \
    coeff * (idx_no_approx_zero * np.cosh(kappa * br['x']) + np.cos(kappa * br['x'])) 
    
br['phi_x'] = phi_x
br['phi_y'] = phi_y
br['phi_z'] = phi_z
    
plt.figure()
plt.pcolormesh(reshape_3D(phi_x, br['Nx'], br['Ny'], br['Nz'])[-1,1])
plt.colorbar()
plt.axis('equal')

br['om'] = kappa**2 * \
    np.real(np.sqrt(br['E'] * (1 + 1j*br['eta']) * br['h']**2 / (12*br['rho'])))
br['om'] = br['om'].flatten()

# -------------- Modes de flexion parallèle --------------
    
br['Nm'] = br['Nm'] + kappa.size

om = kappa**2 * \
    np.real(np.sqrt(br['E'] * (1 + 1j*br['eta']) * br['w']**2 / (12*br['rho'])))
    
br['om'] = np.concatenate((br['om'],om.flatten()))

br['phi_z'] = np.concatenate((br['phi_z'],phi_y))
br['phi_y'] = np.concatenate((br['phi_y'],phi_z))
br['phi_x'] = np.concatenate((br['phi_x'],phi_x))

#%%

# -------------- Modes de torsion --------------

n = np.arange(1,20)

kappa = n*np.pi/br['L']
idx = kappa < kappa_lim
kappa = kappa[idx, np.newaxis]

br['Nm'] = br['Nm'] + kappa.size

om = kappa * np.real(np.sqrt(br['G']*(1+1j*br['eta'])/br['rho']))

br['om'] = np.concatenate((br['om'],om.flatten()))

theta = np.cos(kappa*br['x'])

phi_z = -br['z'] * (1 - np.cos(theta)) - \
    br['y'] * np.sin(theta)
    
phi_y = br['z'] * np.sin(theta) - br['y'] * (1 - np.cos(theta))

phi_x = np.zeros(phi_z.shape)

br['phi_z'] = np.concatenate((br['phi_z'],phi_z))
br['phi_y'] = np.concatenate((br['phi_y'],phi_y))
br['phi_x'] = np.concatenate((br['phi_x'],phi_x))

plt.figure()
plt.pcolormesh(reshape_3D(phi_z, br['Nx'], br['Ny'], br['Nz'])[-1,1])
plt.colorbar()
plt.axis('equal')

#%%
# -------------- Modes rigides --------------

br['Nm'] = br['Nm'] + 3

om = np.zeros(3)

br['om'] = np.concatenate((br['om'],om.flatten()))

phi_z = np.concatenate((np.ones((1,br['Nx']*br['Ny']*br['Nz'])),
                        np.zeros((1,br['Nx']*br['Ny']*br['Nz'])),
                        np.zeros((1,br['Nx']*br['Ny']*br['Nz'])),
                        np.zeros((1,br['Nx']*br['Ny']*br['Nz'])),
                        br['x']/br['L'],
                        br['y']/br['w'])) 
phi_y = np.concatenate((np.zeros((1,br['Nx']*br['Ny']*br['Nz'])),
                        np.ones((1,br['Nx']*br['Ny']*br['Nz'])),
                        np.zeros((1,br['Nx']*br['Ny']*br['Nz'])),
                        br['x']/(br['L'])*np.ones((1,br['Nx']*br['Ny']*br['Nz'])),
                        np.zeros((1,br['Nx']*br['Ny']*br['Nz'])),
                       -br['z']/(br['w'])*np.ones((1,br['Nx']*br['Ny']*br['Nz']))))
phi_x = np.concatenate((np.zeros((1,br['Nx']*br['Ny']*br['Nz'])),
                        np.zeros((1,br['Nx']*br['Ny']*br['Nz'])),
                        np.ones((1,br['Nx']*br['Ny']*br['Nz'])),
                        -br['y']/br['L'],
                        -br['z']/(br['L'])*np.ones((1,br['Nx']*br['Ny']*br['Nz'])),
                       np.zeros((1,br['Nx']*br['Ny']*br['Nz']))))

br['phi_z'] = np.concatenate((br['phi_z'],phi_z))
br['phi_y'] = np.concatenate((br['phi_y'],phi_y))
br['phi_x'] = np.concatenate((br['phi_x'],phi_x))

plt.figure()
plt.pcolormesh(reshape_3D(phi_y, br['Nx'], br['Ny'], br['Nz'])[3,1])
plt.colorbar()
plt.axis('equal')

#%%
# -------------- Union des modes de chevalet -------------- 

idx = np.argsort(br['om'])

br['om']    = br['om'][idx]
br['phi_z'] = br['phi_z'][idx]
br['phi_y'] = br['phi_y'][idx]
br['phi_x'] = br['phi_x'][idx]
            
br['M'] = np.empty((br['Nm'],br['Nm']))

print('------- Computation of the M matrix starting -------')
for i in range(br['Nm']):
    print(i+1,'over',br['Nm'])
    idx = (np.arange(i,br['Nm']), np.arange(br['Nm']-i))
    phi_norm2 = reshape_3D(br['phi_x'], br['Nx'], br['Ny'], br['Nz'])[idx[0]] * \
        reshape_3D(br['phi_x'], br['Nx'], br['Ny'], br['Nz'])[idx[1]] + \
                reshape_3D(br['phi_y'], br['Nx'], br['Ny'], br['Nz'])[idx[0]] * \
        reshape_3D(br['phi_y'], br['Nx'], br['Ny'], br['Nz'])[idx[1]] + \
                reshape_3D(br['phi_z'], br['Nx'], br['Ny'], br['Nz'])[idx[0]] * \
        reshape_3D(br['phi_z'], br['Nx'], br['Ny'], br['Nz'])[idx[1]]
    m = np.trapz(phi_norm2, axis=-1, dx=br['Delta']) 
    m = np.trapz(m, axis=-1, dx=br['Delta']) 
    m = br['rho'] * np.trapz(m, axis=-1, dx=br['Delta_z']) 
    br['M'][idx[0],idx[1]]    = m  
    br['M'][idx[1],idx[0]]    = m
print('------- Computation of the M matrix finished -------')

br['K'] = br['M'] * br['om']**2
br['C'] = br['M'] * br['om'] * br['eta']

#%% Display

plt.figure()
plt.pcolormesh(reshape_3D(phi_y, br['Nx'], br['Ny'], br['Nz'])[3,1])
plt.colorbar()
plt.axis('equal')

br['M'][br['M']<=0] = 0

plt.figure()
plt.imshow(10*np.log10((br['M']+1e-50)**2/np.max(br['M']**2)))
plt.clim((-50,0))
plt.colorbar()

#%% Display

# M = np.zeros((pl['Nm']+br['Nm'], pl['Nm']+br['Nm']))
# M[:pl['Nm'],:pl['Nm']] = pl['M']
# M[pl['Nm']:pl['Nm']+br['Nm'],pl['Nm']:pl['Nm']+br['Nm']] = br['M']
# K = np.zeros((pl['Nm']+br['Nm'], pl['Nm']+br['Nm']))
# K[:pl['Nm'],:pl['Nm']] = pl['K']
# K[pl['Nm']:pl['Nm']+br['Nm'],pl['Nm']:pl['Nm']+br['Nm']] = br['K']
# C = np.zeros((pl['Nm']+br['Nm'], pl['Nm']+br['Nm']))
# C[:pl['Nm'],:pl['Nm']] = pl['C']
# C[pl['Nm']:pl['Nm']+br['Nm'],pl['Nm']:pl['Nm']+br['Nm']] = br['C']
    
M = br['M']
C = br['C']
K = br['K']

# plt.figure()
# plt.pcolormesh(reshape_3D(phi_y, br['Nx'], br['Ny'], br['Nz'])[3,1])
# plt.colorbar()
# plt.axis('equal')

br['M'][br['M']<=0] = 0

plt.figure()
plt.imshow(10*np.log10((M+1e-50)**2/np.max(M**2)))
plt.clim((-50,0))
plt.colorbar()

#%%

# -------------- Matrices UK -------------- 

# Diagonalize the M, C and K matrices

eigval, P   = np.linalg.eig(M)    # Md = (P^-1)MP
Md          = np.diag(eigval)
#Md          = np.linalg.inv(P) @ M @ P
Cd          = np.linalg.inv(P) @ C @ P
Kd          = np.linalg.inv(P) @ K @ P

plot_matrix = Md
plt.figure()
plt.imshow(10*np.log10((plot_matrix+1e-50)**2/np.max(plot_matrix**2)))
plt.clim((-200,0))
plt.colorbar()

# -------------- Main UK --------------

# -------------- Affichage des résultats --------------