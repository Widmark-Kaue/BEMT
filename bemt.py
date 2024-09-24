import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from blade_design import process_file, airfoil_path

def coefficients_extrapolation(df_coeff:pd.DataFrame, smooth:bool = False) -> pd.DataFrame:
    # Model for Large Angle of Attacks - From modified Hoerner flat plat coefficients
    alpha_extra_pos = np.linspace(25, 180, 200)
    alpha_extra_neg = np.linspace(-180, -25, 200)
    
    Cl_extra_pos =  np.sin(2*np.deg2rad(alpha_extra_pos))
    Cl_extra_neg =  np.sin(2*np.deg2rad(alpha_extra_neg))
    
    Cd_extra_pos =  1.3*np.sin(np.deg2rad(alpha_extra_pos))**2
    Cd_extra_neg =  1.3*np.sin(np.deg2rad(alpha_extra_neg))**2
    
    df_extra = pd.DataFrame(
        {
            'alpha': np.concatenate((alpha_extra_neg, df_coeff[:,0] , alpha_extra_pos)),
            'Cl':  np.concatenate((Cl_extra_neg, df_coeff[:,1], Cl_extra_pos)),
            'Cd':  np.concatenate((Cd_extra_neg, df_coeff[:,2], Cd_extra_pos))

        }
    )    
    return  df_extra

def bemt(rotor:pd.DataFrame, airfoil_name:str, threeD_correction: bool = False, tip_correction:str = '', iter:int = 100, tol:float = 1e-3):
    # Read  airfoil data
    c_lift_drag = process_file(airfoil_path.joinpath(f'{airfoil_name}_c_drg.txt'))
   
    # Extrapolation of  coefficients for large angles of attack
    re = max(c_lift_drag, key = lambda x: float(x.split()[-1])) # get the greater reynolds number case
    coeff_extra = coefficients_extrapolation(c_lift_drag[re][0])
    Cl_inter = interp1d(np.deg2rad(coeff_extra['alpha']),  coeff_extra['Cl'], kind='cubic')
    Cd_inter = interp1d(np.deg2rad(coeff_extra['alpha']),  coeff_extra['Cd'], kind='cubic')

    
    # Step 1 - Initialize the BEMT parameters
    a = a_line = np.zeros(len(rotor['x']))
    a_new = a_line_new = np.zeros(len(rotor['x']))
    
    phi = np.zeros(len(rotor['x']))
    alpha = np.zeros(len(rotor['x']))
    Ct = np.zeros(len(rotor['x']))
    Cn = np.zeros(len(rotor['x']))
    
    
    for i in range(iter):
        # Step 2 - Compute flow angle
        phi = np.arctan((1-a)/(1+a_line)/rotor['x'])
        
        # Step 3 - Compute local angle of attack
        alpha = phi - np.deg2rad(rotor['theta'])
        
        # Step 4 - Compute local lift and drag coefficients
        
        
        # Step 5 - Compute local Cn and Ct
        
        # Step 6 - Update induction factors
        a_new = 1/((4*rotor['Tip Correction'] * np.sin(phi)**2/(rotor['sigma'] * Cn)) + 1)
        a_line_new = 1/((4*rotor['Tip Correction'] * np.sin(phi)* np.cos(phi)/(rotor['sigma'] * Cn)) - 1)
        
        #  Step 7 - Check convergence
        if  np.allclose(a, a_new, atol=tol):
            break

         
     