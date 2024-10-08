import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import root
from blade_design import blade_design
from utils import *

def bemt(TSR:float, rotor:pd.DataFrame, airfoil_name:str, number_of_blades: int, threeD_correction: bool = False, tip_correction_model:str = 'Prandtl', iter:int = 100, tol:float = 1e-3, D: float= 0.05):
    
    # Read  airfoil data
    c_lift_drag = process_file(airfoil_path.joinpath(f'{airfoil_name}_c_drg.txt'))
    c_lift = process_file(airfoil_path.joinpath(f'{airfoil_name}_c_lft.txt'))
    
    re_drg = list(c_lift_drag.keys())[-1]
    re_lft = list(c_lift.keys())[-2:]

    Cd = c_lift_drag[re_drg][0][:, 2]
    alpha_cd = c_lift_drag[re_drg][0][:, 0]

    Cl = np.concatenate([c_lift[re][0][:,1] for re in re_lft])  
    alpha_cl = np.concatenate([c_lift[re][0][:,0] for re in re_lft])  

    arg = np.argsort(alpha_cl)
    alpha_cl = np.sort(alpha_cl)
    Cl = Cl[arg]

    Cl_mat = np.concatenate((Cl.reshape(-1,1), alpha_cl.reshape(-1,1)), axis = 1)
    Cd_mat = np.concatenate((Cd.reshape(-1,1), alpha_cd.reshape(-1,1)), axis = 1)
   
    # Extrapolation of  coefficients for large angles of attack
    coeff_extra = coefficients_extrapolation(Cl_mat,Cd_mat, alpha_shift=5, interpolate_type='quadratic')
    Cl_extra = interp1d(coeff_extra['alpha'],  coeff_extra['Cl'])
    Cd_extra = interp1d(coeff_extra['alpha'],  coeff_extra['Cd'])

    
    # Step 0 - Prepare geometric parameters (Using intermediate points to avoid edge effects)
    r_R = rotor['r/R'].to_numpy()[1:-1]
    sigma = rotor['sigma'].to_numpy()[1:-1] * rotor['Tip Correction'].to_numpy()[1:-1]
    theta = np.deg2rad(rotor['theta'].to_numpy()[1:-1])
    x = r_R*TSR                   

    # Step 1 - Initialize the BEMT parameters
    a =  np.zeros(len(x))
    a_line = np.zeros(len(x))
    a_new =  np.zeros(len(x))
    a_line_new = np.zeros(len(x))
    
    phi = np.zeros(len(x))
    alpha = np.zeros(len(x))
    Ct = np.zeros(len(x))
    Cn = np.zeros(len(x))

    error = []
    
    for i in range(iter):
        # Step 2 - Compute flow angle
        phi = np.arctan((1-a)/(1+a_line)/x)
        
        # Step 2.5 - Tip Correction
        F = tip_correction(phi,1/r_R, number_of_blades, model= tip_correction_model)
        
        # Step 3 - Compute local angle of attack
        alpha = phi - theta 
        
        # Step 4 - Compute local lift and drag coefficients
        Cl = Cl_extra(alpha)
        Cd = Cd_extra(alpha)
        
        # Step 5 - Compute local Cn and Ct
        Cn = Cl*np.cos(phi) + Cd*np.sin(phi)
        Ct = Cl*np.sin(phi) - Cd*np.cos(phi)
        
        # Step 6 - Update induction factors
        # Update axial induction factor
        if np.all(a <= 1/3):
            a_new = 4*F *np.sin(phi)**2/(sigma * Cn) + 1
            a_new = 1/a_new
            
        else:
            locs = a <=  1/3
            locs_root = a > 1/3
            
            a_new[locs] = 1/((4*F[locs] * np.sin(phi[locs])**2/(sigma[locs] * Cn[locs])) + 1)
            # a_line_new[locs] = 1/((4*F[locs] * np.sin(phi[locs])* np.cos(phi[locs])/(sigma[locs] * Ct[locs])) - 1)
            
            # Glauert Correction for high values of a
            term1 = lambda a: (1 - a)**2 * sigma[locs_root]*Cn[locs_root]/(4*F[locs_root]*np.sin(phi[locs_root])**2)
            term2 = lambda a: a*(1 - 1/4 * (5 - 3*a)*a)
            func = lambda a: term1(a) - term2(a)
            
            # K = sigma[locs_root]*Cn[locs_root]/(np.sin(phi[locs_root])**2)
            # func = lambda a: -K + a*(1+ 4*F[locs_root] + 2*K) - a**2*(5*F[locs_root] + K)+3*F[locs_root]*a**3
            sol = root(func, (2/3)*np.ones(len(a[locs_root])), method = 'lm')
            a_new[locs_root] = sol.x
            # a_line_new[locs_root] = (1 - 3*a_new[locs_root])/(4*a_new[locs_root] - 1)
        
        # Update tangential induction factor
        a_line_new = 4*F*np.sin(phi) *  np.cos(phi)/(sigma * Ct) - 1
        a_line_new = 1/a_line_new
        
        #  Step 7 - Check convergence: Root Mean Square Error
        error_a = np.sqrt(np.sum((a_new - a)**2)/len(a))
        error_a_line = np.sqrt(np.sum((a_line_new - a_line)**2)/len(a_line))
        error.append([error_a, error_a_line])
      
        if  np.all(np.array(error[-1])< tol):
            a = a_new.copy()
            a_line = a_line_new.copy()
            del a_new
            del a_line_new
            break
        else:
            # Update the induction factors
            a = a_new.copy()
            a_line = a_line_new.copy()
            a_new = np.zeros(len(x))
            a_line_new = np.zeros(len(x))

    # Step 8 - Compute local thrust and power
    # Step 9 - Compute total thrust and power
    pass
if __name__ == '__main__':
    airfoil_name = 's834'
    rotor = blade_design('s834', 7, 2, number_of_sections=50,plot=False)
    # plt.plot(rotor['r/R'], rotor['c/R'])
    # plt.show()
    bemt(7, rotor, airfoil_name, 2, iter=500, D=0.1, tol=1e-6)