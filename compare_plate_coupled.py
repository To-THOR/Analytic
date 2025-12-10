# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from scipy.signal import ShortTimeFFT
import scipy.signal as sgn
from scipy.signal import find_peaks

#%% User's input

dim3_coupling_1             = True
constraint_correction_1     = True
null_1                      = False
Dx_1 = "4.100"
Dy_1 = "3.500"

dim3_coupling_2             = True
constraint_correction_2     = True
null_2                      = False
Dx_2 = "3.417"
Dy_2 = "3.500"

#%%

name_1    = "Plaque_Chevalet_"+null_1*"Null_"+"Dx_"+Dx_1+"_Dy_"+Dy_1+"_cm"+ dim3_coupling_1 * "_3D_coupled" + constraint_correction_1 * "_corrected" + "_FRF"
name_2    = "Plaque_Chevalet_"+null_2*"Null_"+"Dx_"+Dx_2+"_Dy_"+Dy_2+"_cm"+ dim3_coupling_1 * "_3D_coupled" + constraint_correction_2 * "_corrected" + "_FRF"

file_1    = np.load(name_1 + ".npz")
file_2    = np.load(name_2 + ".npz")

#%%
freq_1      = file_1["freq"]
sum_FRF_1   = file_1["sum_FRF"]
sum_FRFd_1  = file_1["sum_FRFd"]
sum_FRFdd_1 = file_1["sum_FRFdd"]
exc_FRF_1   = file_1["exc_FRF"]
exc_FRFd_1  = file_1["exc_FRFd"]
exc_FRFdd_1 = file_1["exc_FRFdd"]
x_exc_1     = file_1["x_exc"] 
y_exc_1     = file_1["y_exc"]
z_exc_1     = file_1["z_exc"]

freq_2      = file_2["freq"]
sum_FRF_2   = file_2["sum_FRF"]
sum_FRFd_2  = file_2["sum_FRFd"]
sum_FRFdd_2 = file_2["sum_FRFdd"]
exc_FRF_2   = file_2["exc_FRF"]
exc_FRFd_2  = file_2["exc_FRFd"]
exc_FRFdd_2 = file_2["exc_FRFdd"]
x_exc_2     = file_2["x_exc"] 
y_exc_2     = file_2["y_exc"]
z_exc_2     = file_2["z_exc"]

#%%

exc_FRFdd_dB_1  = 20 * np.log10(np.abs(exc_FRF_1) / np.abs(exc_FRF_1).max())
exc_FRFdd_dB_2  = 20 * np.log10(np.abs(exc_FRF_2) / np.abs(exc_FRF_2).max())

plt.figure()
plt.plot(freq_1, exc_FRFdd_dB_1, label=name_1)
plt.plot(freq_2, exc_FRFdd_dB_2, label=name_2)
plt.legend()