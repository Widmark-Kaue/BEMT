import numpy as np
import pandas as pd

from blade_design import process_file, airfoil_path

def coefficients_extrapolation(df_coeff:pd.DataFrame, rotor:pd.DataFrame) -> pd.DataFrame:
    pass

def bemt(rotor:pd.DataFrame, airfoil_name:str, threeD_correction: bool = False, iter:int = 100, tol:float = 1e-3):
    # Read  airfoil data
    c_lift_drag = process_file(airfoil_path.joinpath(f'{airfoil_name}_c_drg.txt'))

    
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
        
        #  Step 4 - Compute local lift and drag coefficients
        
        # Step 5 - Compute local Cn and Ct
        
        # Step 6 - Update induction factors
        a_new = 1/((4*rotor['Tip Correction'] * np.sin(phi)**2/(rotor['sigma'] * Cn)) + 1)
        a_line_new = 1/((4*rotor['Tip Correction'] * np.sin(phi)* np.cos(phi)/(rotor['sigma'] * Cn)) - 1)
        
        #  Step 7 - Check convergence
        if  np.allclose(a, a_new, atol=tol):
            break

         
     