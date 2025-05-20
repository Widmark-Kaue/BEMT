#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from src.noise import farfield
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

#%% Paths
path = Path('validate_data', 'noise')
if not path.exists():
    path = Path('..','validate_data','noise')


#%% Load Garrick data
table_path = path.joinpath('example_prop.xlsx')
table = pd.read_excel(table_path)
Mvec1 = table['M'].to_numpy()
Tvec1 = table['T'].to_numpy()
Jvec1 = table['V/nD'].to_numpy()

Mvec = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9])
valid_points = np.isin(Mvec1 , Mvec)
Tvec = Tvec1[valid_points]
Jvec = Jvec1[valid_points]

Mtip = np.pi*Mvec[1]/Jvec[1]
MrotReff = 0.8*np.pi*Mvec[1]/Jvec[1]


Q = 2680 #ft-lb


# Geometric constants
n = 101                         # Number of points
B = 2
R = 5 #ft
D = 2*R
Reff = 0.8*R
phi = np.arange(0, 2*np.pi, 2*np.pi/(n-1))
dphi =  2*np.pi/(n-1)


# Noise Constants
m = 1
k = 0.29686

# Microphone Coordinates
mic2D = np.array([
    np.linspace(-5*D,5*D, 200), # x coord
    2*D * np.ones(200),           # y coord
    np.zeros(200)                   # z coord
    ]).T


# Moving Force coordinate
y1 = Reff*np.cos(phi)
z1 = Reff*np.sin(phi)
x1 = np.zeros(len(y1))


case = farfield(microphones=mic2D)

fig = plt.figure(figsize=(12, 5))
tableau_colors = plt.get_cmap('tab10').colors

gs = GridSpec(2, 3)

 
## Garrick Watkins Reff
for i in range(len(Mvec)):
    ax = fig.add_subplot(gs[0, i]) if i <=2 else fig.add_subplot(gs[1, i-3])
    
    T = Tvec[i]
    Mx = Mvec[i]
    
    prms1 = case.garrickWatkinsReff(
                            number_of_harmonics=1,
                            number_of_blades=B,
                            reff=Reff,
                            loading=(T*478.8, Q*478.8),
                            Mrot=MrotReff,
                            Mx=Mx
                        )


    prms2 = case.hansonReff(
                        number_of_harmonics=1,
                        number_of_blades=B,
                        Mt=Mtip,
                        # Mx=Mx,
                        rtip=D/2,
                        zeff=0.8,
                        BD = 0.05,
                        loading=(T*478.8, Q*478.8),
                        rms=True,
                        include_imag_part=True
                        )
    prms3 = case.hansonReff(
                        number_of_harmonics=1,
                        number_of_blades=B,
                        Mt=Mtip,
                        Mx=Mx,
                        rtip=D/2,
                        zeff=0.8,
                        BD = 0.05,
                        loading=(T*478.8, Q*478.8),
                        rms=True,
                        include_imag_part=True
                        )


    ax.plot(mic2D[:,0]/D, prms1, 'r--', label = fr'Garrick&Watkins', linewidth = 1, alpha = 0.8)
    ax.plot(mic2D[:,0]/D, prms2, 'k-',  label = fr'Hanson', linewidth = 1)  
    ax.plot(mic2D[:,0]/D, prms3, 'k:',  label = fr'Hanson - With Mx', linewidth = 1)  

    
    if i == 0:
        ax.legend(loc = 'best')
    
    ax.set_title(fr'M = {Mx}')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 480)
    ax.grid(which='both', ls='--') 



fig.supxlabel('$x/D$')
fig.supylabel('$P_{rms}$ [Pa]')

plt.tight_layout()
plt.show()

#%% Load data from Casalino, et al. (2021) and Carvalho, et al. (2023)
patternName = 'casalino_2021.txt'

J_CT = np.loadtxt(path.joinpath('JxCT_'+patternName))
J_CP = np.loadtxt(path.joinpath('JxCP_'+patternName))
  
J = np.linspace(J_CT[0,0], J_CT[-1,0], 100)
J = J[(J >= J_CP[0,0]) * (J <= J_CP[-1, 0])]

CP = interp1d(J_CP[:,0], J_CP[:,1])
CT = interp1d(J_CT[:,0], J_CT[:,1])

plt.plot(J_CP[::3,0], J_CP[::3,1],'ro', label = 'CP')
plt.plot(J_CT[::3,0], J_CT[::3,1],'bo', label = 'CT')

plt.plot(J, CP(J), 'r--')
plt.plot(J, CT(J), 'b--')

plt.title('Casalino, et al. (2021)')
plt.xlabel('J [-]')
plt.ylabel(r'$C_{T,P} \times 10^1$' + '[-]')

plt.legend()
plt.grid()
plt.show()  

#%%


# Set Constants
D = 0.3             # m
Reff = D/2 * 0.8    # m
RPM = 5000          # RPM
n = RPM/60          # Hz
Omega =  n*2*np.pi  # rad/s
Mt = 0.23
Pref = 2e-5         # Pa

SPL  = lambda prms: 20* np.log10(prms/Pref)

# Microphones positions
mics = np.array([
    np.linspace(-0.9, 0.9, 13), # x coord
    -4*D * np.ones(13),          # y coord
    np.zeros(13)                # z coord
]).T

# Noise case
case = farfield(microphones = mics)
rho = case.density
c = case.sound_speed
theta = np.rad2deg(case.microphones_to_polar[:,1])


# Figure
fig = plt.figure(figsize=[12, 4.8])
gs = GridSpec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# Noise Evaluate
# kwargs = dict(
#     number_of_harmonics = 1,
#     number_of_blades = 2,
#     rtip = D/2,
#     zeff = 0.8,
#     Mt = Mt,
#     Mx = 0,
#     BD = 0.05,
#     loading = np.zeros(2),
#     rms = True,
#     include_imag_part = True
# )


## ###########################
# J  = 0.4 -> V0 = 10 m/s
## ###########################
df_J04 = pd.read_excel(path.joinpath('thetaxSPL_carvalho2023.xlsx'), sheet_name='J04')

V0 = 10
Mx = V0/c
print(f'Mx = {Mx}')



CT_J04 = CT(0.4)*1e-1
CP_J04 = CP(0.4)*1e-1

T_J04 = CT_J04 * rho * D**4 * n**2          
W_J04 = CP_J04 * rho * D**5 * n**3
Q_J04 = W_J04/Omega

print(f'Q_J04 = {Q_J04:.2f} N m')
print(f'T_J04 = {T_J04:.2f} N')

# kwargs['loading'] = np.array([T_J04, Q_J04])

prms1 = case.garrickWatkinsReff(
    number_of_harmonics=1,
    number_of_blades=B,
    reff=Reff,
    loading=(T_J04, Q_J04),
    Mrot=0.8*Mt,
    # Mtip=Mt,
    Mx=Mx
)


prms2 = case.hansonReff(
    number_of_harmonics = 1,
    number_of_blades = 2,
    rtip = D/2,
    zeff = 0.8,
    Mt = Mt,
    Mx = 0,
    BD = 0.05,
    loading = (T_J04, Q_J04),
    rms = True,
    include_imag_part = True
)

prms3 = case.hansonReff(
    number_of_harmonics = 1,
    number_of_blades = 2,
    rtip = D/2,
    zeff = 0.8,
    Mt = Mt,
    Mx = Mx,
    BD = 0.05,
    loading = (T_J04, Q_J04),
    rms = True,
    include_imag_part = True
)



ax1.plot(theta, SPL(prms1), 'k', label = 'Garrick-Watkins')
ax1.plot(theta, SPL(prms2), 'k--', label = 'Hanson')
ax1.plot(theta, SPL(prms3), 'k:', label = 'Hanson - Mx')
ax1.plot(df_J04['Exp-Theta'], df_J04['Exp-SPL'], 'ko',markersize = 4, label = 'Casalino,2021')


ax1.set_title('J = 0.4')
ax1.set_xlabel(r'$\theta$ [deg]')
ax1.set_ylabel('SPL [dB]')

# ax1.set_ylim([52, 67])

ax1.legend(loc = 'lower right')
ax1.grid()

## ###########################
# J  = 0.6 -> V0 = 15 m/s
## ###########################
df_J06 = pd.read_excel(path.joinpath('thetaxSPL_carvalho2023.xlsx'), sheet_name='J06')

V0 = 15
Mx = V0/c
print(f'Mx = {Mx}')

CT_J06 = CT(0.6)*1e-1
CP_J06 = CP(0.6)*1e-1

T_J06 = CT_J06 * rho * D**4 * n**2          
W_J06 = CP_J06 * rho * D**5 * n**3
Q_J06 = W_J06/Omega

print(f'Q_J06 = {Q_J06:.2f} N m')
print(f'T_J06 = {T_J06:.2f} N')

prms1 = case.garrickWatkinsReff(
    number_of_harmonics=1,
    number_of_blades=B,
    reff=Reff,
    loading=(T_J06, Q_J06),
    Mrot=0.8*Mt,
    Mx=Mx 
)


prms2 = case.hansonReff(
    number_of_harmonics = 1,
    number_of_blades = 2,
    rtip = D/2,
    zeff = 0.8,
    Mt = Mt,
    Mx = 0,
    BD = 0.05,
    loading = (T_J06, Q_J06),
    rms = True,
    include_imag_part = True
)

prms3 = case.hansonReff(
    number_of_harmonics = 1,
    number_of_blades = 2,
    rtip = D/2,
    zeff = 0.8,
    Mt = Mt,
    Mx = Mx,
    BD = 0.05,
    loading = (T_J06, Q_J06),
    rms = True,
    include_imag_part = True
)



ax2.plot(theta, SPL(prms1), 'k', label = 'Garrick-Watkins')
ax2.plot(theta, SPL(prms2), 'k--', label = 'Hanson')
ax2.plot(theta, SPL(prms3), 'k:', label = 'Hanson - Mx')
ax2.plot(df_J04['Exp-Theta'], df_J04['Exp-SPL'], 'ko',markersize = 4, label = 'Casalino,2021')


ax2.set_title('J = 0.6')
ax2.set_xlabel(r'$\theta$ [deg]')
ax2.set_ylabel('SPL [dB]')

ax2.grid()
plt.show()
# %%
