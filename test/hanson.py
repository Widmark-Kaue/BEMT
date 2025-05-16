#%% Lib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm

from scipy.interpolate import interp1d
from pathlib import Path
from src.noise import farfield

#%% Load data from Casalino, et al. (2021)
patternName = 'casalino_2021.txt'
path = Path()

if not path.joinpath('CTxJ_'+patternName).exists():
    path = Path('test')

J_CT = np.loadtxt(path.joinpath('CTxJ_'+patternName))
J_CP = np.loadtxt(path.joinpath('CPxJ_'+patternName))
  
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
# Set Constants
D = 0.3         # m
RPM = 5000        # RPM
n = RPM/60




## J  = 0.4 -> V0 = 10 m/s
Re = 99700
kinematic_vicosity = D**2 * np.pi*n/(2*Re) # m^2/s
viscosity = 1.81e-5                        # Pa.s
density = viscosity/kinematic_vicosity     # kg/m^3


CT_J04 = CT(0.4)*1e-1
CP_J04 = CP(0.4)*1e-1

# T_J04 =                 


# Noise Model
model = TonalNoiseModel(microphones = mics, density = density)

Mt = D*np.pi*n / model.sound_speed

# Microphones positions
mics = np.array([
    np.linspace(-0.9, 0.9, 13), # x coord
    4*D * np.ones(13),          # y coord
    np.zeros(13)                # z coord
])

kwargs = dict(
    number_of_blades = 2,
    rtip = D/2,
    zeff = 0.8,
    Mt = (n * 2*np.pi/60 * D/2) /model.sound_speed,
    Mx = 0,
    BD = 0.05,
    loading = np.array([0.2, 0.2]),
    rms = True
)
# %%
