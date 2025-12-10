# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#%% Torsion 

n           = 30
u           = np.linspace(0,n*np.pi,100000)
du          = u[1] - u[0] 
integrand   =  (np.cos(u)**2)
I           = 1/(n*np.pi) * integrand.sum() * du

plt.figure()
plt.plot(u,integrand)

I_litt      = 1/2 
print(I)
print(I_litt)
#%% Flexion : fonction asymptotique

values      = np.load("Data/beam_kappaL.npy")
mode        = 0
u           = np.linspace(0,values[mode],10000)
du          = u[1] - u[0] 

alphan      = (np.sin(u[-1]) - np.sinh(u[-1])) / \
              (np.cosh(u[-1]) - np.cos(u[-1]))
alphan_litt =  -1  + 2 * np.exp(-u[-1]) * np.sin(u[-1])

phin        = np.sinh(u) + np.sin(u) + alphan * (np.cosh(u) + np.cos(u))
dphin       = np.cosh(u) + np.cos(u) + alphan * (np.sinh(u) - np.sin(u))
ddphin      = np.sinh(u) - np.sin(u) + alphan * (np.cosh(u) - np.cos(u))  
phinx       = -(np.cosh(u) + np.cos(u) + alphan * (np.sinh(u) - np.sin(u)))
phin_litt   = -np.exp(-u) + np.sin(u) - np.cos(u) + np.sin(u[-1]) * np.exp(u-u[-1])
phinx_litt  = -(np.exp(-u) + np.sin(u) + np.cos(u) + np.sin(u[-1]) * np.exp(u-u[-1]))

plt.figure()
plt.plot(u,phin,label='Ground truth')
plt.plot(u,phin_litt, label='Approximate')
plt.legend()

plt.figure()
plt.plot(u,phinx,label='Ground truth')
plt.plot(u,phinx_litt, label='Approximate')
plt.legend()
        
#%% Flexion : intégrale

int_        = np.sum(phin**2) * du
int_litt    = np.sum(phin_litt**2) * du 
int_approx  = 2 + u[-1]
int_approx_bis = 1/2 * (np.sin(u[-1])**2 - 2 * np.sin(2*u[-1]) - 1)

print(int_, int_litt, int_approx, int_approx_bis)
print(np.abs((phin.sum()- phin_litt.sum())**2) / (phin.sum()**2))

#%%

A = np.exp(-2*u)
B = np.exp(2*(u-u[-1])) * np.sin(u)**2
C = np.ones(u.size)
D = - 2 * np.exp(-u) * np.sin(u)
E = 2 * np.exp(-u) * np.cos(u)
F = 2 * np.exp(u-u[-1]) * np.sin(u) * np.sin(u[-1])
G = - 2 * np.exp(u-u[-1]) * np.cos(u) * np.sin(u[-1])
H = - 2 * np.cos(u) * np.sin(u)
I = - 2 * np.exp(-u[-1])

I1 = 1/2 
I2 = 1/2 * np.sin(u[-1])**2
I3 = u[-1]
I4 = - 1
I5 = 1
I6 = np.sin(u[-1]) * (np.sin(u[-1]) - np.cos(u[-1]))
I7 = -np.sin(u[-1]) * (np.sin(u[-1]) + np.cos(u[-1]))
I8 = -1

II  = I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8
III = 1/2 * (np.sin(u[-1])**2 - 2 * np.sin(2*u[-1])- 1 + 2 * u[-1])

phin2_approx    = A + B + C + D + E + F + G + H + I
phin2_approx_2  = A + B + C + D + E + F + G + H + 0*I

plt.figure()
plt.plot(phin**2, label='Ground truth')
plt.plot(phin2_approx, label='Approx 1')
plt.plot(phin2_approx_2, label='Approx 2')
plt.legend()

#%%

plt.figure()
plt.plot(u,A)
plt.plot(u,np.exp(-2*u))

print('-------------------')
print(A.sum() * du, I1, np.exp(-2*u).sum() * du)
print(B.sum() * du, I2)
print(C.sum() * du, I3)
print(D.sum() * du, I4)
print(E.sum() * du, I5)
print(F.sum() * du, I6)
print(G.sum() * du, I7)
print(H.sum() * du, I8)

I_true      = np.sum(phin**2) * du
Id_true     = np.sum(dphin * dphin) * du
Iodd_true    = np.sum(ddphin * phin) * du
Idd_true    = np.sum(ddphin * ddphin) * du
print(I_true, 
      np.sum(phin2_approx) * du,
      np.sum(phin2_approx_2) * du, 
      II,
      III)

print(np.abs(I_true - III) / I_true * 100, "%")

#%%

if mode==0:
    np.savez("Data/Integral_beam_mode_0.npz", I_phi_phi = I_true, 
             I_dphi_dphi = Id_true, I_phi_ddphi = Iodd_true, I_ddphi_ddphi = Idd_true)
    
#%% Orthogonalité

L       = 1
kappanL = np.load("Data/beam_kappaL.npy")
kappan  = kappanL / L
Nm      = kappan.size
Nx      = 10000
x       = np.linspace(0,L,Nx)
dx      = x[1] - x[0]

PHI     = np.zeros((Nm, Nx))
DPHI    = np.zeros((Nm, Nx))
DDPHI   = np.zeros((Nm, Nx))

kap     = kappan[0]
alphan          = (np.sin(kap * L) - np.sinh(kap * L)) / \
                    (np.cosh(kap * L) - np.cos(kap * L))
PHI[0]      = np.sinh(kap * x) + np.sin(kap * x) + \
                     alphan * (np.cosh(kap * x) + np.cos(kap * x))
DPHI[0]     = kap * (np.cosh(kap * x) + np.cos(kap * x) + \
                     alphan * (np.sinh(kap * x) - np.sin(kap * x)))
DDPHI[0]    = kap**2 * (np.sinh(kap * x) - np.sin(kap * x) + \
                     alphan * (np.cosh(kap * x) - np.cos(kap * x)))
for i,kap in enumerate(kappan[1:],1):
    PHI[i]      = -np.exp(-kap*x) + np.exp(kap*(x-L)) * np.sin(kap*L) + \
                        np.sin(kap*x) - np.cos(kap*x)
    DPHI[i]     = kap * (np.exp(-kap*x) + np.exp(kap*(x-L)) * np.sin(kap*L) + \
                        np.sin(kap*x) + np.cos(kap*x))
    DDPHI[i]    = kap**2 * (-np.exp(-kap*x) + np.exp(kap*(x-L)) * np.sin(kap*L) - \
                        np.sin(kap*x) + np.cos(kap*x))
                        
MAC     = np.zeros((Nm, Nm))
MACD    = np.zeros((Nm, Nm))
MACDD   = np.zeros((Nm, Nm))
MACDD_  = np.zeros((Nm, Nm))

for i in range(Nm):
    for j in range(i,Nm):
        temp_int    = dx * (PHI[i] * PHI[j]).sum()
        MAC[i,j]    = temp_int
        MAC[j,i]    = temp_int
        temp_int    = dx * (DPHI[i] * DPHI[j]).sum()
        MACD[i,j]   = temp_int
        MACD[j,i]   = temp_int
        temp_int    = dx * (DDPHI[i] * DDPHI[j]).sum()
        MACDD[i,j]  = temp_int
        MACDD[j,i]  = temp_int
        temp_int    = dx * (DDPHI[i] * PHI[j]).sum()
        MACDD_[i,j] = temp_int
        temp_int    = dx * (PHI[i] * DDPHI[j]).sum()
        MACDD_[j,i] = temp_int
        
plt.matshow(20 * np.log10(np.abs(MAC) / np.abs(MAC).max()), vmin=-20, vmax=0)
plt.colorbar()
plt.matshow(20 * np.log10(np.abs(MACD) / np.abs(MACD).max()), vmin=-20, vmax=0)
plt.colorbar()
plt.matshow(20 * np.log10(np.abs(MACDD) / np.abs(MACDD).max()), vmin=-20, vmax=0)
plt.colorbar()
plt.matshow(20 * np.log10(np.abs(MACDD_) / np.abs(MACDD_).max()), vmin=-20, vmax=0)
plt.colorbar()

int_value   = 1 / (2 * kappan) * \
    (np.sin(kappan*L)**2 - 2 * np.sin(2*kappan*L) + 2 * kappan * L - 1)
int_value_bis = L
intod_value = kappan / 2 * \
    (3 + np.sin(kappan*L)**2 - 2 * L * kappan)
intod_value_bis = kappan * (2 - L * kappan)
intd_value  = kappan / 2 * \
    (7 + 5 * np.sin(kappan*L)**2 + 2 * L * kappan) 
intd_value_bis  = kappan * (6 + L * kappan)
intdd_value = kappan**3 / 2 * \
    (-1 + np.sin(kappan*L)**2 + 2 * np.sin(2*kappan*L) + 2 * L * kappan) 
intdd_value_bis = kappan**4 * L

error       = np.abs(int_value - np.diag(MAC)) / np.abs(np.diag(MAC)) 

error_bis   = np.abs(int_value_bis - np.diag(MAC)) / np.abs(np.diag(MAC))

errorod     = np.abs(intod_value - np.diag(MACDD_)) / np.abs(np.diag(MACDD_))

errorod_bis = np.abs(intod_value_bis - np.diag(MACDD_)) / np.abs(np.diag(MACDD_))

errord      = np.abs(intd_value - np.diag(MACD)) / np.abs(np.diag(MACD))

errord_bis  = np.abs(intd_value_bis - np.diag(MACD)) / np.abs(np.diag(MACD))

errordd     = np.abs(intdd_value - np.diag(MACDD)) / np.abs(np.diag(MACDD))

errordd_bis = np.abs(intdd_value_bis - np.diag(MACDD)) / np.abs(np.diag(MACDD))

print("\n------ Error phi * phi ------")
print(error)
print(error_bis)

print("\n------ Error phi * ddphi ------")
print(errorod)
print(errorod_bis)

print("\n------ Error dphi * dphi ------")
print(errord)
print(errord_bis)

print("\n------ Error ddphi * ddphi ------")
print(errordd)
print(errordd_bis)