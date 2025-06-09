#%% 
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:55:44 2025

@author: Widmark Cardoso
"""
import numpy as np
# import plotly.io as pio
# import plotly.tools as tls
import matplotlib.pyplot as plt
import matplotlib

from scipy.interpolate import interp1d
from src.noise import farfield
from matplotlib.gridspec import GridSpec
from string import Template
from pathlib import Path

matplotlib.use('TkAgg')

#%% Paths
path = Path('validate_data', 'ALM')

if not path.exists():
    path = Path('..', 'validate_data', 'ALM')

#%% Parameters for simulation
tsr = 7.55
rho = 1.222
Vtip = 80
Uinf = Vtip/tsr
B = 3

#%% Geometry
radius, twist, chord = np.loadtxt(path.joinpath('geo_NREL5MW.txt'), delimiter=',', unpack=True, 
                                  usecols=(0, 1, 2))

R = radius[-1]
D = 2*R
chord_fn = interp1d(radius, chord,fill_value="extrapolate") # type: ignore

# MCA = np.full_like(chord, 0)

fig = plt.figure(figsize=(10, 6))
gs = GridSpec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])


ax1.plot(radius,chord,'k')

ax1.set_xlabel('Radius [m]')
ax1.set_ylabel('Chord [m]')
ax1.grid()

ax2.plot(radius, twist, 'k')

ax2.set_xlabel('Radius [m]')
ax2.set_ylabel('Twist [deg]')
ax2.grid()


le = chord/2
te =-chord/2

ax3.plot(radius, le, 'k')
ax3.plot(radius, te, 'k')
ax3.fill_between(radius, le, te, color = 'gray')

for i in range(len(le)):
    ax3.plot([radius[i]]*2, [le[i], te[i]], 'k--')


ax3.set_xlabel('Radius [m]')
ax3.set_xlim((0, R))
ax3.set_ylim((-3, 3))


plt.tight_layout()
# plt.show()
plt.show(block = False)

# plotly_fig = tls.mpl_to_plotly(fig)
# pio.write_html(plotly_fig,'figure.html')


#%% Parameters in Nek for writing file
period=2*np.pi/7.55
Tmax=10
# Tmax=1.5
nt=20
indexTmax=(Tmax*nt+1) +1

nacl=22


#%% load 
data = np.loadtxt(path.joinpath('ACL11.dat'))

datanekfull = np.reshape(data, (-1,nacl, 11))
# datanekfull = np.permute_dims(datanekfull, (1, 0, 2))
datanek = datanekfull[0:indexTmax, : , :]


t_T = datanek[:, :, 0] # Tempo varia na direção i
r_R = datanek[:, :, 1] # discretização do raio varia na direção j

t_T = datanek[:, 0, 0] # Tempo varia na direção i
r_R = datanek[0, :, 1] # discretização do raio varia na direção j

chord_nek = chord_fn(r_R*R)


# Velocities
Vt = datanek[:, :, 3]
Vn = datanek[:, :, 4]

# Forces
fn = datanek[:, :, 9]
ft = datanek[:, :, 10]

# Angle
phi = np.arctan2(Vn, -Vt)

# Lift and drag
fl = fn*np.cos(phi) + ft*np.sin(phi)
fd = fn*np.sin(phi) - ft*np.cos(phi)
dLdr = fl*(rho*Uinf**2*R)
dDdr = fd*(rho*Uinf**2*R)

# Thrust and Torque
dT_dr = B*(fn*rho*Uinf**2*R)
dQ_dr = (r_R*R) *B*(ft*rho*Uinf**2*R)
T = np.trapezoid(dT_dr, r_R*R)
Q = np.trapezoid(dQ_dr, r_R*R)


#%% Plots
legend = r'$ALM^d \ (\varepsilon = 3.5 \Delta x)$'
xlabel = r'$r/R$'

subs = ['f_n','f_t', 'f_l', 'f_d']
ylabel = [r"$\dfrac{" + i + r"}{\rho U^2 R}$" for i in subs]  

fig = plt.figure(figsize=(10, 6))
gs = GridSpec(2, 2)

# Normal Force
ax1 = fig.add_subplot(gs[0, 0])

ax1.plot(r_R[1:-1], fn[-1, 1:-1],'go--' ,label = legend)

ax1.set_xlabel(xlabel)
ax1.set_ylabel(ylabel[0])

ax1.set_xticks(np.arange(0, 1.1, 0.1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax1.set_xlim((0, 1))
ax1.set_ylim(0, 0.9)

ax1.legend()
ax1.grid()

# Tangential force
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(r_R[1:-1], ft[-1, 1:-1],'go--' ,label = legend)

ax2.set_xlabel(xlabel)
ax2.set_ylabel(ylabel[1])

ax2.set_xticks(np.arange(0, 1.1, 0.1))
# plt.yticks(np.arange(0, 1, 0.1))

ax2.set_xlim((0, 1))
ax2.set_ylim(-0.02, 0.1)

ax2.legend()
ax2.grid()


# Lift force
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(r_R[1:-1], fl[-1, 1:-1],'go--' ,label = legend)

ax3.set_xlabel(xlabel)
ax3.set_ylabel(ylabel[2])

ax3.set_xticks(np.arange(0, 1.1, 0.1))
ax3.set_yticks(np.arange(-0.1, 1, 0.1))

ax3.set_xlim((0, 1))
ax3.set_ylim(-0.1, 0.9)

ax3.legend()
ax3.grid()


# Drag force
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(r_R[1:-1], fd[-1, 1:-1],'go--' ,label = legend)

ax4.set_xlabel(xlabel)
ax4.set_ylabel(ylabel[3])

ax4.set_xticks(np.arange(0, 1.1, 0.1))
# plt.yticks(np.arange(0, 0.04, 0.1))

ax4.set_xlim((0, 1))
ax4.set_ylim(0, 0.04)

ax4.legend()
ax4.grid()


plt.tight_layout()
# plt.show()
plt.show(block = False)

#%% Plot Thrust and Torque
plt.close('all')

fig = plt.figure(figsize=(11, 4))
gs = GridSpec(1, 2)
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(r_R, dT_dr[-1], 'k--')

ax1.set_xlabel('r/R')
ax1.set_ylabel('T')
ax1.grid()


ax2 = fig.add_subplot(gs[0,1])
ax2.plot(r_R, dQ_dr[-1], 'k--')

ax2.set_xlabel('r/R')
ax2.set_ylabel('Q')

ax2.grid()
    
plt.tight_layout()
# plt.show()
plt.show(block = False)


#%% Load thrust and torque from reference
Tdata = np.loadtxt(path.joinpath('thrust.txt'))
Qdata = np.loadtxt(path.joinpath('torque.txt'))

fig = plt.figure(figsize=(11, 5))

plt.plot(Tdata[:,0], Tdata[:,1], 'm', label = 'Thrust [kN]')
plt.plot(Qdata[:,0], Qdata[:,1], 'y', label = 'Torque [kN.m]')
plt.plot(Uinf, T[-1]/1e3, 'ko', label = 'Simulated Thrust')
plt.plot(Uinf, Q[-1]/1e3, 'ks', label = 'Simulated Torque')


plt.axvline(Uinf, ls = '--', c = 'k', label = r'$U_{inf}$' + f' = {Uinf:.1f} m/s (tsr = {tsr:.2f})')

plt.xticks(np.arange(3, 26))
plt.yticks(np.arange(0, 5.5, 0.5)*1e3)
plt.xlim([3, 25])
plt.ylim([0, 5000])


plt.xlabel('Wind Speed [m/s]')

plt.legend()

plt.grid(ls = '--')
plt.tight_layout()
# plt.show()
plt.show(block = False)

#%% Noise Computation
plt.close('all')
# Initial Conditions
nmics = 100
Pref = 2e-5
mics = np.array([
    np.linspace(-5*D, 5*D, nmics),
    2*D*np.ones(nmics),
    np.zeros(nmics)
    ]).T

case = farfield(microphones=mics)
Mx = Uinf/case.sound_speed
Mt = Vtip/case.sound_speed

# Values for distribuied method
# pos = (dQ_dr[-1] >=0 ) * (dT_dr[-1] >= 0)
# dQdr_valid = dQ_dr[-1, pos]
# dTdr_valid = dT_dr[-1, pos]
# r_R_valid = r_R[pos]
# chord_nek_valid = chord_nek[pos]

dTdr = dT_dr[-1]
dQdr = dQ_dr[-1]

MCA = np.full_like(chord_nek, 0)
prmsH = case.hansonSteady(
    number_of_harmonics=1, 
    number_of_blades=B,
    Mt = Mt,
    Mx=Mx,
    rtip=R,
    z = r_R,
    b = chord_nek,
    MCA=MCA,
    loading=[dTdr, dQdr]
)

prmsHLD = case.hansonSteady_liftDrag(
    number_of_harmonics=1, 
    number_of_blades=B,
    Mt = Mt,
    Mx=Mx,
    rtip=R,
    z = r_R,
    b = chord_nek,
    MCA=MCA,
    loading=[dLdr[-1], dDdr[-1]]
)

# Values to Effective Radius method
Reff = 0.8*R
Mrot = 0.8*Mt

prmsGWe = case.garrickWatkinsReff(
    number_of_harmonics=1,
    number_of_blades=B,
    reff=Reff,
    loading=np.array([T[-1], Q[-1]]),
    Mrot=Mrot,
    Mx=Mx
)

prmsHe = case.hansonReff(
    number_of_harmonics=1,
    number_of_blades=B, 
    Mt = Mt,
    rtip=R,
    BD = chord_fn(0.8*R)/D,
    loading=[T[-1], Q[-1]],
    Mx = Mx,
    phase=True
)


splGWe = 20*np.log10(prmsGWe/Pref)
splHe = 20*np.log10(prmsHe/Pref)
splH = 20*np.log10(prmsH/Pref)
splHLD = 20*np.log10(prmsHLD/Pref)


fig = plt.figure(figsize=(10, 6))
gs = GridSpec(2, 1)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

ax1.plot(mics[:, 0]/D, prmsGWe, 'k', label = 'Garrick-Watkins (Eff. Radius)')
ax1.plot(mics[:, 0]/D, prmsHe, 'ro', label = 'Hanson (Eff. Radius)', markersize = 4)
ax1.plot(mics[:, 0]/D, prmsH, 'k--', label = 'Hanson')
ax1.plot(mics[:, 0]/D, prmsHLD, 'k-.', label = 'Hanson - LiftDrag')
ax1.set_xlabel('x/D')
ax1.set_ylabel('Prms')
ax1.set_xlim(-5, 5)

ax1.legend()
ax1.grid()

ax2.plot(mics[:, 0]/D, splGWe, 'k', label = 'Garrick-Watkins')
ax2.plot(mics[:, 0]/D, splHe, 'ro', label = 'Hanson', markersize = 4)
ax2.plot(mics[:, 0]/D, splH, 'k--', label = 'Hanson')
ax2.plot(mics[:, 0]/D, splHLD, 'k-.', label = 'Hanson - LiftDrag')
ax2.set_xlabel('x/D')
ax2.set_ylabel('SPL [dB]')
ax2.set_xlim(-5, 5)
ax2.set_ylim(-40, 80)

ax2.legend()
ax2.grid()

plt.tight_layout()

plt.show(block = False)


# %%
