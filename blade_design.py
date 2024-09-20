#%% Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import BASE_COLORS
from pathlib import Path
from read_files import process_file

#%% Colors
colors = list(BASE_COLORS.keys())

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
# plt.savefig(image_path.joinpath('cl_alpha.png'), format = 'png', dpi = 720)
plt.show()

# %% Load c_drg file and plot Cl/Cd x alpha
c_lift_drag = process_file(airfoil_path.joinpath('s834_c_drg.txt'))
cols = c_lift_drag.columns

alpha_opt = []
cl_opt  = []
cd_opt = []
re = []
for i, col in enumerate(cols):
    re.append(float(col.split()[-1]))
    
    LD = c_lift_drag[col][0][:, 1]/c_lift_drag[col][0][:, 2]
    alpha = c_lift_drag[col][0][:,  0]
    
    opt_loc = np.where(LD == max(LD))[0][0]
    alpha_opt.append(alpha[opt_loc])
    cl_opt.append(c_lift_drag[col][0][opt_loc, 1])
    cd_opt.append(c_lift_drag[col][0][opt_loc, 2])
    
    plt.plot(alpha, LD, colors[i], label=f'Re = {re[-1]:.3e},'+ r' $\alpha_{opt} =$'+ f'{alpha_opt[-1]:.1f}°')
    plt.plot(alpha_opt[-1], max(LD), f'{colors[i]}o')

plt.title(r'L/D x $\alpha$')
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel('Cl/Cd')

plt.legend()
plt.grid()
# plt.savefig(image_path.joinpath('LD_alpha.png'), format = 'png', dpi = 720)
plt.show()

df_opt = pd.DataFrame({
    'Re': re,
    'alpha_opt': alpha_opt,
    'cl_opt': cl_opt,
    'cd_opt': cd_opt,
})

print(df_opt)
# %% Flow Angle and Pitch Angle
line_max_re = df_opt.loc[df_opt['Re'].idxmax()]
print(line_max_re)

alpha_opt = line_max_re['alpha_opt']

# Tip Speed Ratio
TSR = 7

# axial induction factor
a = np.arange(0.26, 0.3333, 1e-5)

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

df_sections = pd.DataFrame({
    'a':  a, 
    'a_line': a_line, 
    'x': x, 
    'r/R': x/TSR,  
    'phi': phi, 
    'theta': theta_opt
    } )

# Plotting
locs = df_sections['r/R'] <= 1 #valid points


plt.plot(x[locs]/TSR, theta_opt[locs],'k', label = 'Optimun pitch angle')
plt.plot(x[locs]/TSR, phi[locs], 'k--', label = 'Flow angle')

plt.xlabel('r/R')
plt.ylabel(r'$\theta$, $\phi$ [deg]')

plt.grid()
plt.legend()
plt.show()
# %% Chord Distribution
B = 2           # number of blades
phi_rad = np.deg2rad(phi[locs])

# Tip Prandtl's correction function
f = B/2/np.sin(phi_rad) * (TSR/x[locs] - 1)
F = 2/np.pi * np.arccos(np.exp(-f))

# Tangential force coefficient
Ct = line_max_re['cl_opt']*np.sin(phi_rad) - line_max_re['cd_opt']*np.cos(phi_rad)

# Solidity
sigma = 4*x[locs]*a_line[locs] * np.sin(phi_rad)**2/(1 - a[locs])/Ct

# Chord Distributio
c_R = 2*np.pi*sigma*x[locs]/(B*TSR)
    
plt.plot(x[locs]/TSR, c_R, 'k')
plt.plot(x[locs]/TSR, c_R*F, 'k--', label = 'With Tip Prandtl Correction')

plt.title('Chord Distribution')
plt.xlabel(r'$r/R$')
plt.ylabel(r'$c/R$')

plt.legend()
plt.grid()
plt.show()
