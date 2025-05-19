#%% Lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import matplotlib.ticker as ticker
import matplotlib.cm as cm

from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from pathlib import Path
from src.noise import farfield

#%% Load data from Casalino, et al. (2021)
patternName = 'casalino_2021.txt'
path = Path('validate_data', 'noise')

if not path.exists():
    path = Path('..', 'validate_data', 'noise')


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


# %% Hanson Method (Effectiv Radius) - Test With Carvalho, 2023 Data

# Create figure object
fig = plt.figure(figsize=[12, 4.8])
gs = GridSpec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# Set Constants
D = 0.3             # m
RPM = 5000          # RPM
n = RPM/60          # Hz
Omega =  n*2*np.pi  # rad/s
Mt = 0.23
Pref = 2e-5         # Pa

# Microphones positions
mics = np.array([
    np.linspace(-0.9, 0.9, 13), # x coord
    4*D * np.ones(13),          # y coord
    np.zeros(13)                # z coord
]).T

# Noise case
case = farfield(microphones = mics)
rho = case.density


# Noise Evaluate
kwargs = dict(
    number_of_harmonics = 1,
    number_of_blades = 2,
    rtip = D/2,
    zeff = 0.8,
    Mt = Mt,
    Mx = 0,
    BD = 0.05,
    loading = np.zeros(2),
    rms = True,
    include_imag_part = True
)


## ###########################
# J  = 0.4 -> V0 = 10 m/s
## ###########################
df_J04 = pd.read_excel(path.joinpath('thetaxSPL_carvalho2023.xlsx'), sheet_name='J04')


CT_J04 = CT(0.4)*1e-1
CP_J04 = CP(0.4)*1e-1

T_J04 = CT_J04 * rho * D**4 * n**2          
W_J04 = CP_J04 * rho * D**5 * n**3
Q_J04 = W_J04/Omega

print(f'Q_J04 = {Q_J04:.2f} N m')
print(f'T_J04 = {T_J04:.2f} N')

kwargs['loading'] = np.array([T_J04, Q_J04])

prms = case.hansonReff(**kwargs)
spl = 20* np.log10(prms/Pref)

theta = np.rad2deg(case.microphones_to_polar[:,1])

ax1.plot(theta, spl, 'b', label = 'Implemented')
ax1.plot(df_J04['Model-Theta'], df_J04['Model-SPL'], 'k', label = 'Carvalho,2023')
ax1.plot(df_J04['Exp-Theta'], df_J04['Exp-SPL'], 'ko', label = 'Casalino,2021')


ax1.set_title('J = 0.4')
ax1.set_xlabel(r'$\theta$ [deg]')
ax1.set_ylabel('SPL [dB]')

ax1.set_ylim([52, 67])

ax1.legend(loc = 'lower right')
ax1.grid()
# plt.show()

## ###########################
# J  = 0.6 -> V0 = 15 m/s
## ###########################
df_J06 = pd.read_excel(path.joinpath('thetaxSPL_carvalho2023.xlsx'), sheet_name='J06')


CT_J06 = CT(0.6)*1e-1
CP_J06 = CP(0.6)*1e-1

T_J06 = CT_J06 * rho * D**4 * n**2          
W_J06 = CP_J06 * rho * D**5 * n**3
Q_J06 = W_J06/Omega

print(f'Q_J06 = {Q_J06:.2f} N m')
print(f'T_J06 = {T_J06:.2f} N')

kwargs['loading'] = np.array([T_J06, Q_J06])

prms = case.hansonReff(**kwargs)
spl = 20* np.log10(prms/Pref)

ax2.plot(theta, spl, 'b', label = 'Implemented')
ax2.plot(df_J06['Model-Theta'], df_J06['Model-SPL'], 'k', label = 'Carvalho,2023')
ax2.plot(df_J06['Exp-Theta'], df_J06['Exp-SPL'], 'ko', label = 'Casalino,2021')


ax2.set_title('J = 0.6')
ax2.set_xlabel(r'$\theta$ [deg]')
ax2.set_ylabel('SPL [dB]')

ax2.set_ylim([52, 67])
# ax2.legend(loc = 'lower right')
ax2.grid()

plt.tight_layout()
plt.show()
# %%
