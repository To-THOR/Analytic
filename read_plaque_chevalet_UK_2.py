# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from scipy.signal import ShortTimeFFT
import scipy.signal as sgn
from scipy.signal import find_peaks

#%%
         

dim3_coupling           = True
constraint_correction   = True

Dx = "2.050"
Dy = "3.500"

name = "Plaque_Chevalet_Zero_Dx_"+Dx+"_Dy_"+Dy+"_cm"+ \
        dim3_coupling * "_3D_coupled" + constraint_correction * "_corrected"

file = np.load(name + ".npz")

N                   = int(file["N"])
Nm                  = int(file["Nm"])
dim3_coupling       = bool(file["dim3_coupling"])
q                   = file["q"]
qd                  = file["qd"]
qdd                 = file["qdd"]
F_ext               = file["F_ext"]
F_c                 = file["F_c"]
const_violation     = file["const_violation"]
t                   = file["t"]
M                   = file["M"]
C                   = file["C"]
K                   = file["K"]
A                   = file["A"]
b                   = file["b"]
B                   = file["B"]
W                   = file["W"]
phix                = file["phinx"]
phiy                = file["phiny"]
phiz                = file["phinz"]
x                   = file["x"]
y                   = file["y"]
z                   = file["z"]
idx_exc             = int(file["idx_exc"])

#%%

Te  = t[1] - t[0]
Fe  = 1 / Te 
T   = t[-1]
N_t = t.size

#%% Check results

idx_mode = 50

plt.figure()
plt.plot(t, np.real(q[idx_mode]))    
plt.xlabel("t (s)")
plt.ylabel("q(t) for mode n°"+str(idx_mode))

plt.figure()
plt.plot(t, F_ext[idx_mode].real)
plt.xlabel("t (s)")
plt.ylabel("Fext(t) for mode n°"+str(idx_mode))

plt.figure()
plt.plot(t, F_c[idx_mode].real)
plt.xlabel("t (s)")
plt.ylabel("Fc(t) for mode n°"+str(idx_mode))

z_phys      = (phiz[idx_exc,:] @ q)
zd_phys     = (phiz[idx_exc,:] @ qd)
zdd_phys    = (phiz[idx_exc,:] @ qdd)

plt.figure()
plt.plot(t, np.real(z_phys))
plt.xlabel("t (s)")
plt.ylabel("z(t)")

plt.figure()
plt.plot(t, np.real(const_violation))
plt.xlabel("t (s)")
plt.ylabel("Mean square constraint violation for all constraints")

#%%

Fext_freq       = np.fft.rfft(F_ext[idx_mode].real)
z_phys_freq     = np.fft.rfft(z_phys.real)
zd_phys_freq    = np.fft.rfft(zd_phys.real)
zdd_phys_freq   = np.fft.rfft(zdd_phys.real)
freq            = np.fft.rfftfreq(N_t, Te)

freq_idx        = freq < 5000
Fext_freq       = Fext_freq[freq_idx]
z_phys_freq     = z_phys_freq[freq_idx]
zd_phys_freq    = zd_phys_freq[freq_idx]
zdd_phys_freq   = zdd_phys_freq[freq_idx]
freq            = freq[freq_idx]

Fext_freq_dB        = 20 * np.log10(np.abs(Fext_freq)/np.abs(Fext_freq).max())
z_phys_freq_dB      = 20 * np.log10(np.abs(z_phys_freq)/np.abs(z_phys_freq).max())
zd_phys_freq_dB     = 20 * np.log10(np.abs(zd_phys_freq)/np.abs(zd_phys_freq).max())
zdd_phys_freq_dB    = 20 * np.log10(np.abs(zdd_phys_freq)/np.abs(zdd_phys_freq).max())

#%% Sum_frf

phiz_sum    = np.abs(phiz).sum(axis=0)
Q           = np.fft.rfft(q.real, axis=-1)[:,freq_idx]
Qd          = np.fft.rfft(qd.real, axis=-1)[:,freq_idx]
Qdd         = np.fft.rfft(qdd.real, axis=-1)[:,freq_idx]

sum_FRF_Z   = phiz_sum @ Q
sum_FRF_Zd  = phiz_sum @ Qd
sum_FRF_Zdd = phiz_sum @ Qdd

sum_FRF_Z_dB    = 20 * np.log10(np.abs(sum_FRF_Z) / 
                                np.abs(sum_FRF_Z).max())
sum_FRF_Zd_dB   = 20 * np.log10(np.abs(sum_FRF_Zd) / 
                                np.abs(sum_FRF_Zd).max()) 
sum_FRF_Zdd_dB  = 20 * np.log10(np.abs(sum_FRF_Zdd) / 
                                np.abs(sum_FRF_Zdd).max()) 

peaks_idx       = find_peaks(sum_FRF_Zdd_dB)[0]
peaks_freq      = freq[peaks_idx]
peaks_FRF_Z     = sum_FRF_Z_dB[peaks_idx]
peaks_FRF_Zd    = sum_FRF_Zd_dB[peaks_idx]
peaks_FRF_Zdd   = sum_FRF_Zdd_dB[peaks_idx]
peaks_zdd       = zdd_phys_freq_dB[peaks_idx]
peaks_z         = z_phys_freq_dB[peaks_idx]

#%% Plot

plt.figure()
plt.plot(freq, sum_FRF_Z_dB)
plt.scatter(peaks_freq, peaks_FRF_Z, color='r')
plt.xlabel("f (Hz)")
plt.ylabel("Sum FRF Z (dB)")

plt.figure()
plt.plot(freq, sum_FRF_Zdd_dB)
#plt.scatter(peaks_freq, peaks_FRF_Zdd, color='r')
plt.xlabel("f (Hz)", size=15)
plt.ylabel("dB", size=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.figure()
plt.plot(freq, Fext_freq_dB)
plt.xlabel("f (Hz)")
plt.ylabel("Fext (dB)")

plt.figure()
plt.plot(freq, z_phys_freq_dB)
plt.scatter(peaks_freq, peaks_z, color='r')
plt.xlabel("f (Hz)")
plt.ylabel("z (dB)")

plt.figure()
plt.plot(freq, zdd_phys_freq_dB)
plt.scatter(peaks_freq, peaks_zdd, color='r')
plt.xlabel("f (Hz)")
plt.ylabel("zdd (dB)")

#%% Save sum-FRF and FRF at excitation

FRF_name = name + "_FRF.npz"
np.savez("Data/"+FRF_name, freq = freq ,sum_FRF = sum_FRF_Z, sum_FRFd = sum_FRF_Zd, 
         sum_FRFdd = sum_FRF_Zdd, exc_FRF = z_phys_freq, 
         exc_FRFd = zd_phys_freq, exc_FRFdd = zdd_phys_freq, 
         x_exc = x[idx_exc], y_exc = y[idx_exc], z_exc = z[idx_exc])

print("FRF saved as " + FRF_name)

#%% Operational deform

mode_idx        = 0 
mode_freq_idx   = peaks_idx[mode_idx]

N_anim  = 25
T_anim  = 1 / (freq[mode_freq_idx] + (freq[mode_freq_idx]==0)) 
t_anim  = np.linspace(0,T_anim,N_anim)

z_scale = 0.05

Z_op            = phiz @ Q[:,mode_freq_idx]
Zdd_op          = phiz @ Qdd[:,mode_freq_idx]
Z_op_norm       = z_scale * np.abs(Z_op) / np.abs(Z_op).max()
Zdd_op_norm     = np.abs(Zdd_op) / np.abs(Zdd_op).max()

X_op            = phix @ Q[:,mode_freq_idx]  
Xdd_op          = phix @ Qdd[:,mode_freq_idx]
X_op_norm       = z_scale * np.abs(X_op) / np.abs(Z_op).max()
Xdd_op_norm     = np.abs(Xdd_op) / np.abs(Zdd_op).max()

Y_op            = phiy @ Q[:,mode_freq_idx]  
Ydd_op          = phiy @ Qdd[:,mode_freq_idx]
Y_op_norm       = z_scale * np.abs(Y_op) / np.abs(Z_op).max()
Ydd_op_norm     = np.abs(Ydd_op) / np.abs(Zdd_op).max()

phase_X           = np.angle(X_op) 
phase_Y           = np.angle(Y_op)
phase_Z           = np.angle(Z_op)  

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

scat = ax.scatter(x, y, z, s=5, cmap=cm.winter, 
                  c=Z_op_norm)
ax.set(xlim=(x.min()*0.9, x.max()*1.1), 
       ylim=(y.min()*0.9, y.max()*1.1), 
       zlim=(-z_scale, z_scale))
ax.set_aspect('equal')
ax.axis('off')

def update(frame):
    freq_temp   = 2*np.pi*freq[mode_freq_idx] + (freq[mode_freq_idx]==0)
    time_fact_X = np.sin(freq_temp * t_anim[frame] + phase_X)
    time_fact_Y = np.sin(freq_temp * t_anim[frame] + phase_Y)
    time_fact_Z = np.sin(freq_temp * t_anim[frame] + phase_Z)
    
    scat._offsets3d = (x + time_fact_X * X_op_norm,
                      y + time_fact_Y * Y_op_norm,
                      z + time_fact_Z * Z_op_norm)
    colors = time_fact_Z * Z_op_norm[idx_mode]
    return scat

ani = animation.FuncAnimation(fig=fig, func=update, frames=N_anim, interval=1)

ani_name = "Animations"+name+"_ope_deform_"+str(mode_idx)+".gif"
ani.save(ani_name, writer="pillow")
print("Animation saved as " + ani_name)

#%% Save all ope deforms

N_peaks = peaks_idx.size

X_op    = np.zeros((N_peaks, N), dtype=complex)
Y_op    = np.zeros((N_peaks, N), dtype=complex)
Z_op    = np.zeros((N_peaks, N), dtype=complex)

for i in range(N_peaks):
    Z_op[i] = phiz @ Q[:,peaks_idx[i]]
    X_op[i] = phix @ Q[:,peaks_idx[i]]  
    Y_op[i] = phiy @ Q[:,peaks_idx[i]]  

np.savez("Data"+name+"_ope_deform.npz", x=x, y=y, z=z, freq=peaks_freq,\
        X_op=X_op, Y_op=Y_op, Z_op=Z_op)
    
print("Operational deforms saved as " + name + "_ope_deform.npz")