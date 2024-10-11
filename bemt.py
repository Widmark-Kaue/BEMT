import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import simpson
from scipy.optimize import root
from rotor import Rotor
from utils import *

from dataclasses import dataclass

@dataclass
class BEMTResult:
    rotor: Rotor
    TSR: float
    C_P: float
    C_T: float
    dC_P: np.ndarray
    dC_T: np.ndarray
    iterations:int
    converged:bool
    error:np.ndarray

def bemt(rotor: Rotor,TSR:float, threeD_correction: bool = False, tip_correction_model:str = 'Prandtl', iter:int = 100, tol:float = 1e-3) -> BEMTResult:
    
    if rotor.airfoil_name is not None:
        # Read  airfoil data
        c_lift_drag = process_file(airfoil_path.joinpath(f'{rotor.airfoil_name}_c_drg.txt'))
        c_lift = process_file(airfoil_path.joinpath(f'{rotor.airfoil_name}_c_lft.txt'))
        
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
    else:
        Cl_extra = lambda alpha: rotor.Cl_opt
        Cd_extra = lambda alpha: rotor.Cd_opt
        
    
    # Step 0 - Prepare geometric parameters (Using intermediate points to avoid edge effects)
    r_R = rotor.sections['r_R'].to_numpy()[1:-1]
    c_R = rotor.sections['c_R'].to_numpy()[1:-1]
    sigma = rotor.sections['sigma'].to_numpy()[1:-1] * rotor.sections['tip_correction'].to_numpy()[1:-1]
    theta = np.deg2rad(rotor.sections['theta_opt'].to_numpy()[1:-1])
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
    converged = False
    for i in range(iter):
        # Step 2 - Compute flow angle
        phi = np.arctan((1-a)/(1+a_line)/x)
        
        # Step 2.5 - Tip Correction
        F = tip_correction(phi,1/r_R, rotor.number_of_blades, model= tip_correction_model)
        
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
            
            # Glauert Correction for high values of a
            term1 = lambda a: (1 - a)**2 * sigma[locs_root]*Cn[locs_root]/(4*F[locs_root]*np.sin(phi[locs_root])**2)
            term2 = lambda a: a*(1 - 1/4 * (5 - 3*a)*a)
            func = lambda a: term1(a) - term2(a)
            
            sol = root(func, (2/3)*np.ones(len(a[locs_root])), method = 'lm')
            a_new[locs_root] = sol.x
        
        # Update rotational induction factor
        a_line_new = 4*F*np.sin(phi) *  np.cos(phi)/(sigma * Ct) - 1
        a_line_new = 1/a_line_new
        
        #  Step 7 - Check convergence: Root Mean Square Error
        error_a = np.sqrt(np.sum((a_new - a)**2)/len(a))
        error_a_line = np.sqrt(np.sum((a_line_new - a_line)**2)/len(a_line))
        error.append([error_a, error_a_line])
      
        if  np.all(np.array(error[-1])< tol):
            a = a_new.copy()
            a_line = a_line_new.copy()
            converged = True
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
    dC_T = 4 * np.pi * sigma**2 * Cn /(c_R * rotor.number_of_blades * TSR)
    dC_P = 4 * np.pi * sigma**2 * Ct * x/(c_R * rotor.number_of_blades * TSR)
    
    # Include boundary conditions
    dC_T = np.concatenate(([0], dC_T, [0]))
    dC_P = np.concatenate(([0], dC_P, [0]))

    
    # Step 9 - Compute total thrust and power
    x_full = rotor.sections['r_R'].to_numpy()*TSR
    C_T = simpson(dC_T, x = x_full)
    C_P = simpson(dC_P, x = x_full)
    
    results = BEMTResult(rotor, TSR, C_P, C_T, dC_P, dC_T, i, converged, error)
    
    return results
if __name__ == '__main__':
    rotor = Rotor(number_of_blades=2, number_of_sections=50, tip_speed_ratio=7, airfoil_name='s834')
    rotor.load_airfoil_prop(plot=False)
    rotor.blade_design(plot=False)
    bemt(rotor=rotor, TSR=7, iter=500, tol=1e-6)