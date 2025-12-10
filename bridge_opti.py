# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def w(E,I,rho,w,h,L):
    return np.sqrt(E * I/(rho * w * h)) * kappa**2    

def dw(E,I,rho,w,h,L):
    1/2 * np.sqrt(I / (rho * w * h * E)) * kappa**2

def f(E,I,rho,w,h,L,w_exp):
    return np.sum((w(E,I,rho,w,h,L) - w_exp)**2)
    
def df(E,I,rho,w,h,L,w_exp):
    return np.sum(2 * (w(E,I,rho,w,h,L) - w_exp) * ())

# Unités SI

L       = 0.205
w       = 0.035
h       = 0.010
rho     = 750
G       = 0.93e9
beta    = 1/16 * (16 / 3 - 3.36 * h / w * (1 - 1/12 * h**4 / w**4))
K       = beta * h**3 * w
E       = 13.3e9
I       = w * h**3 / 12
J       = (w * h**3 + h * w**3) / 12

f_max = 5000
f_res = 1e-8
w_max = 2 * np.pi * f_max
w_res = 2 * np.pi * f_res

Delta_x = 1e-3
Delta_y = 1e-3
Delta_z = 1e-3

x = np.arange(0, L, Delta_x)
y = np.arange(-w/2, w/2, Delta_y)
z = np.arange(h/2, -h/2, -Delta_z)

X,Y,Z = np.meshgrid(x,y,z)

Nx, Ny, Nz = X.shape

#%% Flexion

kappa_max   = (rho * h * w / (E * I))**(1/4) * np.sqrt(w_max)
kappa_res   = (rho * h * w / (E * I))**(1/4) * np.sqrt(w_res)
kappa       = np.arange(0, kappa_max, kappa_res)

funca           = np.cos(kappa*L)
funcb           = np.cosh(kappa*L)
func            = funca * funcb - 1
idx_zero_cross  = (np.abs(np.sign(func[1:]) - np.sign(func[:-1]))) == 2
idx_a           = np.append(idx_zero_cross, False)
idx_b           = np.append(False, idx_zero_cross) 
alphan          = (func[idx_a] - func[idx_b]) / (kappa[idx_a] - kappa[idx_b]) 
betan           = func[idx_a] - alphan * kappa[idx_a]
kappan          = - betan / alphan

#%%

wn_expe = np.array([2*np.pi*1001.999, 2*np.pi*2595.23])

E_opti = rho * h * w / I * (np.sum(kappan**2 / wn_expe) / np.sum(kappan**4 / wn_expe**2))**2

wn      = np.sqrt(E * I/(rho * w * h)) * kappan**2   
wn_opti = np.sqrt(E_opti * I/(rho * w * h)) * kappan**2

error_wn = np.sum((wn - wn_expe)**2 / wn_expe**2)
error_wn_opti = np.sum((wn_opti - wn_expe)**2 / wn_expe**2) 

print('Error ratio = '+str(np.round(error_wn_opti / error_wn * 100,1)) + ' %')