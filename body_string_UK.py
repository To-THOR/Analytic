# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from scipy.io.wavfile import write

#%% Load the body's modal basis

Dx = "1.025"
Dy = "3.500"

coupled                 = True
guitar                  = False
system                  = "Plate"
modal                   = True
null                    = False
null_null               = False
dim3_coupling           = True
constraint_correction   = False

if coupled:
    name = "Plaque_Chevalet" + null * "_Zero" + null_null * "_Zero_Zero"  + + modal * "_Modal" + "_Dx_" + Dx + "_Dy_" + Dy + "_cm"+ \
            dim3_coupling * "_3D_coupled" + (not modal) * constraint_correction * "_corrected"
elif guitar:
    name = "Guitar_modal_basis"
else:
    name = system + "_modal_basis_Dx_" + Dx + "_Dy_" + Dy + "_cm"

name    = name
file_b  = np.load("Data/" + name + ".npz" )
print("File " + name + ".npz" + " opened.")

#%% Loads the strings' modal bases

Dy = 2.031e-2

string_chosen   = "E2"
 
string_mu       = {"E4" : 0.38e-3,
                   "B3" : 0.52e-3,
                   "G3" : 0.90e-3,
                   "D3" : 1.95e-3,
                   "A2" : 3.61e-3,
                   "E2" : 6.24e-3}

string_T        = {"E4" : 70.3,
                   "B3" : 53.4,
                   "G3" : 58.3,
                   "D3" : 71.2,
                   "A2" : 73.9,
                   "E2" : 71.6} 
 
T       = string_T[string_chosen]
mu      = string_mu[string_chosen]

coupled                 = False
guitar                  = False
system                  = "Plate"
modal                   = True
null                    = False
null_null               = False

name = "String_modal_basis_Dy_"+ string_chosen + f"_{(np.round(Dy*100,3)):.3f}" +"_cm_T_"+\
        f"{(np.round(T,0)):.0f}"+"_N_mu_"+f"{(np.round(mu*1e6,0)):.0f}"+"_mg.m-1"

name    = name
file_s  = np.load("Data/" + name + ".npz" )
print("File " + name + ".npz" + " opened.")

#%% File's name

dim3_coupling           = True
constraint_correction   = True

name = "Corps_Cordes_Dy_" + str(Dy) + "_cm"+ \
        dim3_coupling * "_3D_coupled" + constraint_correction * "_corrected"
        
#%%

F_max = 20e3
W_max = 2 * np.pi * F_max 

x_b             = file_b['x']
y_b             = file_b['y']
z_b             = file_b['z']
wn_b            = file_b['wn']
mn_b            = file_b['mn']
kn_b            = file_b['kn']
cn_b            = file_b['cn']
phinx_b         = file_b['phinx']
phiny_b         = file_b['phiny']
phinz_b         = file_b['phinz']
Fx_factor_b     = file_b['Fx_fact']
Fy_factor_b     = file_b['Fy_fact']
Fz_factor_b     = file_b['Fz_fact']

idx = wn_b < W_max

wn_b        = wn_b[idx]
mn_b        = mn_b[idx]
kn_b        = kn_b[idx]
cn_b        = cn_b[idx]
phinx_b     = phinx_b[idx]
phiny_b     = phiny_b[idx]
phinz_b     = phinz_b[idx]
Fx_factor_b = Fx_factor_b[idx]
Fy_factor_b = Fy_factor_b[idx]
Fz_factor_b = Fz_factor_b[idx]


Nm_b            = wn_b.size
N_b             = x_b.size

x_s             = file_s['x']
y_s             = file_s['y']
z_s             = file_s['z']
wn_s            = file_s['wn']
mn_s            = file_s['mn']
kn_s            = file_s['kn']
cn_s            = file_s['cn']
phinx_s         = file_s['phinx']
phiny_s         = file_s['phiny']
phinz_s         = file_s['phinz']
Fx_factor_s     = file_s['Fx_fact']
Fy_factor_s     = file_s['Fy_fact']
Fz_factor_s     = file_s['Fz_fact']

idx = wn_s < W_max

wn_s        = wn_s[idx]
mn_s        = mn_s[idx]
kn_s        = kn_s[idx]
cn_s        = cn_s[idx]
phinx_s     = phinx_s[idx]
phiny_s     = phiny_s[idx]
phinz_s     = phinz_s[idx]
Fx_factor_s = Fx_factor_s[idx]
Fy_factor_s = Fy_factor_s[idx]
Fz_factor_s = Fz_factor_s[idx]


Nm_s            = wn_s.size
N_s             = x_s.size

Nm  = Nm_b + Nm_s
N   = N_b + N_s

#%% Concatenate 

x_coupling  = x_b[z_b == z_b.max()]
y_coupling  = y_b[z_b == z_b.max()]
x_s         = x_s - x_s.min() + x_coupling[0]
y_s         = y_s - y_s.max() +  y_coupling[0] 
z_s         = np.ones(N_s) * z_b.max()

x                   = np.concatenate((x_b, x_s))
y                   = np.concatenate((y_b, y_s))
z                   = np.concatenate((z_b, z_s))
phix                = np.zeros((N, Nm))
phix[:N_b,:Nm_b]    = phinx_b.T 
phix[N_b:,Nm_b:]    = phinx_s.T
phiy                = np.zeros((N, Nm))
phiy[:N_b,:Nm_b]    = phiny_b.T
phiy[N_b:,Nm_b:]    = phiny_s.T
phiz                = np.zeros((N, Nm))
phiz[:N_b,:Nm_b]    = phinz_b.T
phiz[N_b:,Nm_b:]    = phinz_s.T
Fx_factor           = np.concatenate((np.pad(Fx_factor_b.T, ((0,N_s), (0,0))), np.pad(Fx_factor_s.T, ((N_b,0),(0,0)))), axis=1)
Fy_factor           = np.concatenate((np.pad(Fy_factor_b.T, ((0,N_s), (0,0))), np.pad(Fy_factor_s.T, ((N_b,0),(0,0)))), axis=1)
Fz_factor           = np.concatenate((np.pad(Fz_factor_b.T, ((0,N_s), (0,0))), np.pad(Fz_factor_s.T, ((N_b,0),(0,0)))), axis=1)
M                   = np.diag(np.concatenate((mn_b, mn_s)))
C                   = np.diag(np.concatenate((cn_b, cn_s)))
K                   = np.diag(np.concatenate((kn_b, kn_s)))

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

dist    = (x_s[-1] - x_b)**2 + (y_s[-1] - y_b)**2 + (y_s[-1] - y_b)**2 
idx     = np.arange(N_b)[dist==0]
idx     = idx[0]

idx_coupl = np.array([[idx, N-1]])

#%% Graphical check

# color = np.zeros(N, dtype=bool)
# for i in idx_coupl.flatten():
#     color[i] = True 
    
color = np.zeros(N)
color[idx_coupl[:,0]] = np.ones(idx_coupl.shape[0])
color[idx_coupl[:,1]] = np.ones(idx_coupl.shape[0])

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
Az          = phiz[idx_coupl[:,0]] - phiz[idx_coupl[:,1]]

if dim3_coupling:
    A = np.concatenate((Ax,Az),axis=0)
else:
    A = Az

N_s = A.shape[0]

A_plus      = np.linalg.pinv(A)
b           = np.zeros(N_s)
M_rt_inv    = np.diag(np.concatenate((mn_b**(-1/2),mn_s**(-1/2))))
B           = A @ M_rt_inv
B_plus      = np.linalg.pinv(B)
W           = np.eye(Nm) - M_rt_inv @ B_plus @ A
V           = M_rt_inv @ B_plus @ b
M_inv       =  np.diag(np.concatenate((1/mn_b,1/mn_s)))

#%% Time-domain parameters

Fe      = 100e3
Te      = 1/Fe
T       = 0.1
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

idx_exc = int(N-N_s/2)

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

np.savez("Data/" + name+".npz", N=N, Nm=Nm, 
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
write("Audio/" + name + ".wav", rate, np.pad(scaled[100:], (int(1*rate),0)))

#%%

plt.figure()
plt.plot(np.arange(0,T,1/rate)[-scaled[100:].size:], scaled[100:])

#%%

# T_win       = 0.05
# T_win       = int(Fe * T_win) * Te
# hop_fact    = 0.9
# win         = np.ones(int(Fe * T_win))
# hop         = int(np.round(Fe * T_win * (1 - hop_fact)))
# SFT         = ShortTimeFFT(win,hop,Fe)

# Sx      = SFT.stft(zd_phys.real) 
# t_Sx    = np.arange(0,hop*Te*Sx.shape[1],hop*Te)
# f_Sx    = np.arange(0, Fe/win.size*Sx.shape[0] , Fe/win.size)

# T_Sx, F_Sx = np.meshgrid(t_Sx, f_Sx) 

# idx_f_Sx = f_Sx < 5000 

# plt.figure()
# plt.pcolor(T_Sx[idx_f_Sx,:], F_Sx[idx_f_Sx,:], 20 * \
#            np.log10(np.abs(Sx[idx_f_Sx,:]) / np.max(np.abs(Sx[idx_f_Sx,:]))), vmin = -80, vmax=0)
# # plt.pcolor(T_Sx[idx_f_Sx,:], F_Sx[idx_f_Sx,:], np.abs(Sx[idx_f_Sx,:]) / np.max(np.abs(Sx[idx_f_Sx,:])))    
# cb = plt.colorbar()
# cb.set_label('TFCT (dB)', size=30)
# cb.ax.tick_params(labelsize=20)
# plt.xlabel('Temps (s)', size=30)
# plt.ylabel('Fréquence (Hz)', size=30)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)

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
