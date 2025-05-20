#%% Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


from pathlib import Path
from src.noise import farfield

#%% Colors
colors =[   [0,0,1.0],
            [0,0.5,1.0],
            [0,1.0,1.0],
            [0.5,1.0,0.5],
            [1.0,1.0,0],
            [1.0,0.5,0] ]


#%% Load data and Set constants
table_path = Path('validate_data','noise','example_prop.xlsx')
if not table_path.exists():
    table_path = Path('..','validate_data','noise','example_prop.xlsx')
table = pd.read_excel(table_path)
Mvec1 = table['M'].to_numpy()
Tvec1 = table['T'].to_numpy()
Jvec1 = table['V/nD'].to_numpy()


Mvec = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9])
valid_points = np.isin(Mvec1 , Mvec)
Tvec = Tvec1[valid_points]
Jvec = Jvec1[valid_points]
MtipVec = 0.8*np.pi*Mvec/Jvec
MtipVec[0] = MtipVec[1]
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
mic = np.array([
    np.arange(-0.5*D,0.5*D, (D/(n-1))), # x coord
    0.6*D * np.ones(n - 1),           # y coord
    np.zeros(n - 1)                   # z coord
    ]).T



mic2D = np.array([
    np.linspace(-5*D,5*D, 200), # x coord
    2*D * np.ones(200),           # y coord
    np.zeros(200)                   # z coord
    ]).T

# mic2D = mic.copy()
# mic2D[:, 1] = 2*D*np.ones(n-1)

# Moving Force coordinate
y1 = Reff*np.cos(phi)
z1 = Reff*np.sin(phi)
x1 = np.zeros(len(y1))



#%% Plot mics
ax = plt.figure().add_subplot(projection='3d')
ax.plot(x1, y1, z1, color='b')
ax.plot(mic[:,0], mic[:,1], mic[:,2], 'k')
ax.plot(mic2D[:,0], mic2D[:,1], mic2D[:,2], 'r')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=15., azim=25, roll=0)
plt.tight_layout()
# plt.show()

#%% Compute Prms for Near Field
prms = np.zeros((mic.shape[0], len(Mvec)))
fig, ax  = plt.subplots()
for i, M in enumerate(Mvec):
    T = Tvec[i]
    beta = np.sqrt(1-M**2)
    for mic_i in range(n-1):
        # Mic Coord
        x, y, z = mic[mic_i]

        S = np.sqrt( (x - x1)**2 + beta**2*( (y - y1)**2 + (z - z1)**2) )
        sigma = (M*(x - x1) + S)/(beta**2)
        
        # Integrand parts
        ab = T * k/(beta**2) *(M + x/S) - Q * m*B/(Reff**2)
        arg = m*B*phi + k*sigma
        
        
        AA = (T * x/(S**2)  * np.cos(arg) + ab * np.sin(arg))/S
        BB = (-T * x/(S**2)  * np.sin(arg) + ab * np.cos(arg))/S
        
        AA =  np.sum(AA*dphi)
        BB =  np.sum(BB*dphi)
        
        prms[mic_i, i] = np.sqrt(2)/(8*np.pi**2) * np.sqrt((AA*478.8)**2 + (BB*478.8)**2)
    
    ax.plot(mic[:,0]/D, prms[:,i], label = f'M = {M}', color = colors[i], linewidth = 1)


ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.set_xticks(np.arange(-0.5, 0.6, 0.1))
ax.set_xlim(-0.5, 0.5)


ax.set_xlabel('$x/D$')
ax.set_ylabel('$P_{rms}$')
ax.legend()
ax.grid(which='both', ls='--') 
plt.tight_layout()
plt.show()        

 
#%% Compute Prms for Far Field
case = farfield(microphones=mic2D)

fig, ax  = plt.subplots()
for i in range(len(Mvec)):
    T = Tvec[i]
    Mtip = MtipVec[i]
    Mx = Mvec[i]
    
    prms = case.garrickWatkinsReff(
                        number_of_harmonics=1,
                        number_of_blades=B,
                        reff=Reff,
                        loading=(T*480, Q*480),
                        Mrot=Mtip,
                        Mx=Mx
                    )

    ax.plot(mic2D[:,0]/D, prms, color = colors[i],   label = fr'M = {Mx}', linewidth = 1)
    
ax.set_xlim(-4, 4)
ax.set_ylim(0, 480)
ax.set_xlabel('$x/D$')
ax.set_ylabel('$P_{rms}$')
ax.legend()
ax.grid(which='both', ls='--') 
plt.tight_layout()
plt.show()  


# %%
