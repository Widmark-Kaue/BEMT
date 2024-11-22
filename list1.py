#%% libs
from src.rotor import Rotor
import numpy as np 

from scipy.integrate import simpson
#%% 2. c)
rotor  = Rotor(
    number_of_blades=3,
    number_of_sections=100,
    tip_speed_ratio=8.5,
    airfoil_name='s834'
)
rotor.load_airfoil_prop(plot=False)
rotor.blade_design(r0_R=0.05, filter_invalid_solidity=False, plot=False)

rho = 1.225
V0 = 10.59
P  =15e6
R = np.sqrt(P/(0.5*rho*V0**3 *rotor.CP_opt*np.pi))
print(f'{rotor.CP_opt.round(3):=}')
print(f'{R.round(2):=} m')

#%% 2. d)
Cl_Cd = 120

a = rotor.sections['a'].to_numpy()
a_line = rotor.sections['a_line'].to_numpy()
x = rotor.sections['x'].to_numpy()
R_r = 1/rotor.sections['r_R'].to_numpy()
lamb = rotor.tip_speed_ratio

term = (1 - a)/(1+a_line)/x
Cn = Cl_Cd + term
Ct = Cl_Cd*term -1

dCP = 8*a*x*Ct*(1-a)/(Cn*R_r*lamb)

CP = simpson(dCP, x = x)
print(f'{CP.round(4):=}')

R = np.sqrt(P/(0.5*rho*V0**3 *CP*np.pi))
print(f'{R.round(2):=} m')
