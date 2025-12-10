# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

kappaL_max  = 100
kappaL_res  = 1e-5
kappaL      = np.arange(0, kappaL_max, kappaL_res)

funca           = np.cos(kappaL)
funcb           = np.cosh(kappaL)
func            = funca * funcb - 1
idx_zero_cross  = (np.abs(np.sign(func[1:]) - np.sign(func[:-1]))) == 2
idx_a           = np.append(idx_zero_cross, False)
idx_b           = np.append(False, idx_zero_cross) 
alphan          = (func[idx_a] - func[idx_b]) / (kappaL[idx_a] - kappaL[idx_b]) 
betan           = func[idx_a] - alphan * kappaL[idx_a]
kappanL         = - betan / alphan

print(kappanL)

np.save("Data/beam_kappaL.npy", kappanL)