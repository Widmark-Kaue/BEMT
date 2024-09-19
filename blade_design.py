#%% Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from read_files import process_file

#%% Paths
image_path = Path('images')
image_path.mkdir(exist_ok=True)

airfoil_path = Path('airfoil')
#%% Plot airfoil
x, y = np.loadtxt(airfoil_path.joinpath('S834.dat'), skiprows=1, unpack=True)

plt.plot(x, y)
plt.axis('equal')
plt.grid()
plt.savefig(image_path.joinpath('airfoil.png'), format = 'png', dpi = 720)
plt.show()

# %% Load c_lft file and plot Cl x alpha
c_lift = process_file(airfoil_path.joinpath('s834_c_lft.txt'))
cols = c_lift.columns

for col in cols:
    re = float(col.split()[-1])
    plt.plot(c_lift[col][0][:,  0], c_lift[col][0][:, 1], label=f'Re = {re:.3e}')

plt.title(r'Cl x $\alpha$')
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel('Cl')

plt.legend()
plt.grid()
plt.savefig(image_path.joinpath('cl_alpha.png'), format = 'png', dpi = 720)
plt.show()

# %% Load c_drg file and plot Cl/Cd x alpha
c_lift_drag = process_file(airfoil_path.joinpath('s834_c_drg.txt'))
cols = c_lift_drag.columns

for col in cols:
    re = float(col.split()[-1])
    LD = c_lift_drag[col][0][:, 1]/c_lift_drag[col][0][:, 2]
    plt.plot(c_lift_drag[col][0][:,  0], LD , label=f'Re = {re:.3e}')

plt.title(r'L/D x $\alpha$')
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel('Cl/Cd')

plt.legend()
plt.grid()
plt.savefig(image_path.joinpath('LD_alpha.png'), format = 'png', dpi = 720)
plt.show()
# %% Flow Angle and Pitch Angle
alpha_opt = 5

# Tip Speed Ratio
TSR = 7

# axial induction factor
a = np.arange(0.1, 0.33334, 1e-5)

# tangential induction factor
a_line = (1 - 3 *a)/(4*a - 1)

# local rotational speed ratio
num = a*( 1 - a)
den = a_line*(1 + a_line)
x = np.sqrt(num/den)

# Flow angle
phi = np.rad2deg(np.arctan((1 - a)/(1 + a_line)/x))

# Pitch angle
theta_opt = phi - alpha_opt

df = pd.DataFrame({
    'a':  a, 
    'a_line': a_line, 
    'x': x, 
    'r/R': x/TSR,  
    'phi': phi, 
    'theta': theta_opt
    } )

# Plotting
locs = theta_opt > 0
plt.plot(x[locs]/TSR, theta_opt[locs],'k', label = 'Optimun pitch angle')
plt.plot(x[locs]/TSR, phi[locs], 'k--', label = 'Flow angle')

plt.xlabel('r/R')
plt.ylabel(r'$\theta$, $\phi$ [deg]')

plt.xlim([0,1])

plt.grid()
plt.legend()
plt.show()








# %%
 