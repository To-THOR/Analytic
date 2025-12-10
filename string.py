# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

L           = 0.65
T_kg        = 8.07 
T           = 8.07 * 9.81
E4_f        = 329.6276
mu          = T / (2 * L * E4_f)**2

f_max = 5000

c = np.sqrt(T/mu)
f = c/(2*L)

n   = np.arange((4*L/c*f_max - 1) / 2)
kn  = np.pi / (2 * L) * (1 + 2 * n)
fn  = kn * c / (2 * np.pi) 

x    = np.linspace(0,L,1000)

phin = np.sin(kn[:,np.newaxis] * x[np.newaxis,:])
