# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.io.wavfile import write
from scipy.signal import ShortTimeFFT
import matplotlib.cm as cm
import time

#%% Load modal bases

dim3_coupling           = True
constraint_correction   = True

Dx = "2.050"
Dy = "3.500"

name = "Plaque_Chevalet_Dx_"+Dx+"_Dy_"+Dy+"_cm"+ \
        dim3_coupling * "_3D_coupled" + constraint_correction * "_corrected"

file = np.load("Bridge_modal_basis_Dx_"+Dx+"_Dy_"+Dy+"_cm.npz")

x_c             = file['x']
y_c             = file['y']
z_c             = file['z']
wn_c            = file['wn']
mn_c            = file['mn']
kn_c            = file['kn']
cn_c            = file['cn']
phinx_c         = file['phinx']
phiny_c         = file['phiny']
phinz_c         = file['phinz']
Fx_factor_c     = file['Fx_fact']
Fy_factor_c     = file['Fy_fact']
Fz_factor_c     = file['Fz_fact']
params_names_c  = file["params_names"]
params_c        = file['params']

Nm_c    = wn_c.size
N_c     = x_c.size

file = np.load("Plate_modal_basis_Dx_"+Dx+"_Dy_"+Dy+"_cm.npz")

x_p             = file['x']
y_p             = file['y']
z_p             = file['z']
wn_p            = file['wn']
mn_p            = file['mn']
kn_p            = file['kn']
cn_p            = file['cn']
phinx_p         = file['phinx']
phiny_p         = file['phiny']
phinz_p         = file['phinz']
Fx_factor_p     = file['Fx_fact']
Fy_factor_p     = file['Fy_fact']
Fz_factor_p     = file['Fz_fact']
params_names_p  = file["params_names"]
params_p        = file['params']

Nm_p                = wn_p.size
N_p                 = x_p.size

Lx  = params_p[0].real
Ly  = params_p[1].real
h   = params_p[2].real
Lb  = params_c[0].real
wb  = params_c[1].real
hb  = params_c[2].real

x_c = x_c + (Lx - Lb) / 2
y_c = y_c + Ly*0.7
z_c = z_c + (h + hb) / 2

Nm  = Nm_p + Nm_c
N   = N_p + N_c

#%% Concatenate 

x                   = np.concatenate((x_c, x_p))
y                   = np.concatenate((y_c, y_p))
z                   = np.concatenate((z_c, z_p))
phix                = np.zeros((N, Nm))
phix[:N_c,:Nm_c]    = phinx_c.T 
phix[N_c:,Nm_c:]    = phinx_p.T
phiy                = np.zeros((N, Nm))
phiy[:N_c,:Nm_c]    = phiny_c.T
phiy[N_c:,Nm_c:]    = phiny_p.T
phiz                = np.zeros((N, Nm))
phiz[:N_c,:Nm_c]    = phinz_c.T
phiz[N_c:,Nm_c:]    = phinz_p.T
Fx_factor           = np.concatenate((np.pad(Fx_factor_c.T, ((0,N_p), (0,0))), np.pad(Fx_factor_p.T, ((N_c,0),(0,0)))), axis=1)
Fy_factor           = np.concatenate((np.pad(Fy_factor_c.T, ((0,N_p), (0,0))), np.pad(Fy_factor_p.T, ((N_c,0),(0,0)))), axis=1)
Fz_factor           = np.concatenate((np.pad(Fz_factor_c.T, ((0,N_p), (0,0))), np.pad(Fz_factor_p.T, ((N_c,0),(0,0)))), axis=1)
M                   = np.diag(np.concatenate((mn_c, mn_p)))
C                   = np.diag(np.concatenate((cn_c, cn_p)))
K                   = np.diag(np.concatenate((kn_c, kn_p)))

phi = np.sqrt(phix**2 + phiy**2 + phiz**2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scat = ax.scatter(x, y, z, s=5, cmap=cm.winter, 
                  c=phi[:,-1])
ax.set(xlim=(x.min()*0.9, x.max()*1.1), 
       ylim=(y.min()*0.9, y.max()*1.1), 
       zlim=(-0.1, 0.1))
ax.set_aspect('equal')
ax.axis('off')

#%% Compute coupling surface

idx_coupl           = np.zeros((0,2), dtype=int)

for i in range(N_c):
    d = (x[N_c:] - x[i])**2 + (y[N_c:] - y[i])**2 + (z[N_c:] - z[i])**2
    if np.sqrt(d.min()) < 1e-5:
        idx_coupl = np.concatenate((idx_coupl, np.array((i, N_c+d.argmin()))[np.newaxis]), axis=0)
     
#%% Graphical check

# color = np.zeros(N, dtype=bool)
# for i in idx_coupl.flatten():
#     color[i] = True 
    
color = np.abs(Fz_factor[:,0])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scat = ax.scatter(x, y, z, s=5, cmap=cm.winter, 
                  c=color)
ax.set(xlim=(x.min()*0.9, x.max()*1.1),
       ylim=(y.min()*0.9, y.max()*1.1),
       zlim=(-0.1, 0.1))
ax.set_aspect('equal')
ax.axis('off')

#%% UK initial matrices formulation

Ax          = phix[idx_coupl[:,0]] - phix[idx_coupl[:,1]]
Ay          = phiy[idx_coupl[:,0]] - phiy[idx_coupl[:,1]]
Az          = phiz[idx_coupl[:,0]] - phiz[idx_coupl[:,1]]

if dim3_coupling:
    A = np.concatenate((Ax,Ay,Az),axis=0)
else:
    A = Az

N_s = A.shape[0]

A_plus      = np.linalg.pinv(A)
b           = np.zeros(N_s)
M_rt_inv    = np.diag(np.concatenate((mn_c**(-1/2),mn_p**(-1/2))))
B           = A @ M_rt_inv
B_plus      = np.linalg.pinv(B)
W           = np.eye(Nm) - M_rt_inv @ B_plus @ A
V           = M_rt_inv @ B_plus @ b
M_inv       =  np.diag(np.concatenate((1/mn_c,1/mn_p)))

#%% Time-domain parameters

Fe      = 500e3
Te      = 1/Fe
T       = 1
t       = np.arange(0,T,Te)
N_t     = t.size

#%% UK excitation

def give_F_gauss_ponctual(t, idx_exc, Fz_factor):
    T_exc   = 1e-5 
    Fz_exc   = np.exp(-(t-5*T_exc)**2/(2*T_exc**2))
    F_mod   = Fz_exc * Fz_factor[idx_exc,:]
    return F_mod

def give_F_lin_ponctual(t, idx_exc, Fz_factor):
    T_exc       = 0.005
    Fz_exc      = 0
    if t<T_exc:
        Fz_exc = t/T_exc
    F_mod   = Fz_exc * Fz_factor[idx_exc,:]
    return F_mod

idx_exc = ((x[z==h/2])**2 + (y[z==h/2])**2).argmin()
idx_exc = np.arange(N)[z==h/2][idx_exc]

exc_func = give_F_gauss_ponctual

#%% UK initial conditions / vector initialization

q               = np.zeros((Nm,N_t), dtype=complex)
qd              = np.zeros((Nm,N_t), dtype=complex)
qdd             = np.zeros((Nm,N_t), dtype=complex)
F_c             = np.zeros((Nm,N_t), dtype=complex)
F_ext           = np.zeros((Nm,N_t), dtype=complex)
const_violation = np.zeros((N_t), dtype=complex)

#%% UK loop

start = time.time()
print("----------- Begin UK simulation -----------")
for i in range(N_t-1):
    if i%(N_t//(100/1))==0 and i!=0: 
        print(i)
        time_diff = time.time() - start
        remaining_time = (100 - i//(N_t//(100/1))) * time_diff/(i//(N_t//(100/1)))
        print(str(i//(N_t//(100/1)))+"% (time = "+str(np.round(time_diff))+
              " s)\tEstimated time before end : "+str(int(remaining_time//60))+" min "
              + str(int(remaining_time%60))+" s")
    
    q[:,i+1]                = q[:,i] + Te * qd[:,i] + 0.5 * Te**2 * qdd[:,i]
    q[:,i+1]                = q[:,i+1] - constraint_correction * A_plus @ (A @ q[:,i+1] - b)
    const_violation[i+1]    = np.abs(A @ q[:,i+1] - b).sum(axis=0) / N_s
    qd_half                 = qd[:,i] + 0.5 * Te * qdd[:,i]
    
    F_ext[:,i]  = exc_func(t[i+1], idx_exc, Fz_factor)
    F           = - K @ q[:,i+1] - C @ qd_half + F_ext[:,i]
    
    qudd        = M_inv @ F
    
    qdd[:,i+1]  = W @ qudd + V
    
    qd[:,i+1]   = qd[:,i] + 0.5 * Te * (qdd[:,i] + qdd[:,i+1])
    qd[:,i+1]   = qd[:,i+1] - constraint_correction * A_plus @ (A @ qd[:,i+1] - b)
    
    F_c[:,i]    = M_inv @ B_plus @ (b - A @ qudd)
   
print("\nTemps total : " + str(int(time_diff//60)) + " min " + (str(int(time_diff%60)) + ' s'))
print("------------ End UK simulation ------------")
    
#%% Save results

np.savez(name+".npz", N=N, Nm=Nm, 
         dim3_coupling=dim3_coupling, q=q, qd=qd, qdd=qdd, F_ext=F_ext, 
         F_c=F_c, const_violation=const_violation, t=t, M=M, C=C, K=K, A=A, 
         b=b, B=B, W=W, x=x, y=y, z=z, phinx=phix, phiny=phiy,
         phinz=phiz, idx_exc=idx_exc)
print("Simulation saved as " + name + ".npz")

#%% Check results

idx_mode = 0
plt.figure()
plt.plot(t, np.real(q[idx_mode]))    
plt.xlabel("t (s)")
plt.ylabel("q(t) for mode n°"+str(idx_mode))

plt.figure()
plt.plot(t, np.real(F_ext[idx_mode]))
plt.xlabel("t (s)")
plt.ylabel("Fext(t) for mode n°"+str(idx_mode))

plt.figure()
plt.plot(t, np.real(F_c[idx_mode]))
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

#%% Save sound

undersample = int(np.floor(Fe/40e3))
rate        = int(np.round(Fe / undersample)) 
scaled      = np.asarray(np.real(zd_phys[::undersample]), dtype=np.float32)
scaled      = scaled / ( np.max(np.abs(scaled[100:])) +  (np.max(np.abs(scaled))==0) ) 
scaled      = scaled - np.mean(scaled) 
write(name+".wav", rate, np.pad(scaled[100:], (int(1*rate),0)))

#%%

plt.figure()
plt.plot(np.arange(0,T,1/rate)[-scaled[100:].size:], scaled[100:])

#%%

T_win       = 0.05
T_win       = int(Fe * T_win) * Te
hop_fact    = 0.9
win         = np.ones(int(Fe * T_win))
hop         = int(np.round(Fe * T_win * (1 - hop_fact)))
SFT         = ShortTimeFFT(win,hop,Fe)

Sx      = SFT.stft(zd_phys.real) 
t_Sx    = np.arange(0,hop*Te*Sx.shape[1],hop*Te)
f_Sx    = np.arange(0, Fe/win.size*Sx.shape[0] , Fe/win.size)

T_Sx, F_Sx = np.meshgrid(t_Sx, f_Sx) 

idx_f_Sx = f_Sx < 5000 

plt.figure()
plt.pcolor(T_Sx[idx_f_Sx,:], F_Sx[idx_f_Sx,:], 20 * \
           np.log10(np.abs(Sx[idx_f_Sx,:]) / np.max(np.abs(Sx[idx_f_Sx,:]))), vmin = -80, vmax=0)
# plt.pcolor(T_Sx[idx_f_Sx,:], F_Sx[idx_f_Sx,:], np.abs(Sx[idx_f_Sx,:]) / np.max(np.abs(Sx[idx_f_Sx,:])))    
cb = plt.colorbar()
cb.set_label('TFCT (dB)', size=30)
cb.ax.tick_params(labelsize=20)
plt.xlabel('Temps (s)', size=30)
plt.ylabel('Fréquence (Hz)', size=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

#%%

z_fft       = np.fft.rfft(z_phys.real)
freq        = np.fft.rfftfreq(N_t, d=Te)

idx_fft     = freq < 5e3
z_fft       = z_fft[idx_fft]
freq        = freq[idx_fft]

z_fft_dB    = 20 * np.log10(np.abs(z_fft)/np.abs(z_fft).max()) 

plt.figure()
plt.plot(freq, z_fft_dB)
plt.xlabel("f (Hz)")
plt.ylabel("FRF")