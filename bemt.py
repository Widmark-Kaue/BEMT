import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.optimize import root
from blade_design import process_file, airfoil_path, tip_correction, blade_design

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

def bemt(TSR:float, rotor:pd.DataFrame, airfoil_name:str, number_of_blades: int, threeD_correction: bool = False, tip_correction_model:str = 'Prandtl', iter:int = 100, tol:float = 1e-3):
    
    # Read  airfoil data
    c_lift_drag = process_file(airfoil_path.joinpath(f'{airfoil_name}_c_drg.txt'))
   
    # Extrapolation of  coefficients for large angles of attack
    re = max(c_lift_drag, key = lambda x: float(x.split()[-1])) # get the greater reynolds number case
    coeff_extra = coefficients_extrapolation(c_lift_drag[re][0])
    Cl_interp = interp1d(np.deg2rad(coeff_extra['alpha']),  coeff_extra['Cl'], kind='cubic')
    Cd_interp = interp1d(np.deg2rad(coeff_extra['alpha']),  coeff_extra['Cd'], kind='cubic')

    
    # Step 0 - Prepare geometric parameters (Using intermediate points to avoid edge effects)
    r_R = rotor['r/R'].to_numpy()[1:-1]
    sigma = rotor['sigma'].to_numpy()[1:-1]
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
        Cl = Cl_interp(alpha)
        Cd = Cd_interp(alpha)
        
        # Step 5 - Compute local Cn and Ct
        Cn = Cl*np.cos(phi) + Cd*np.sin(phi)
        Ct = Cl*np.sin(phi) - Cd*np.cos(phi)
        
        # Step 6 - Update induction factors
        if np.all(a <= 1/3):
            a_new = 4*F *np.sin(phi)**2/(sigma * Cn) + 1
            a_new = 1/a_new
            
            a_line_new = 4*F*np.sin(phi) *  np.cos(phi)/(sigma * Ct) - 1
            a_line_new = 1/a_line_new
        else:
            locs = a <=  1/3
            locs_root = a > 1/3
            
            a_new[locs] = 1/((4*F[locs] * np.sin(phi[locs])**2/(sigma[locs] * Cn[locs])) + 1)
            a_line_new[locs] = 1/((4*F[locs] * np.sin(phi[locs])* np.cos(phi[locs])/(sigma[locs] * Ct[locs])) - 1)
            
            K = sigma[locs_root]*Cn[locs_root]/(np.sin(phi[locs_root])**2)
            func = lambda a: -K + a*(1+ 4*F[locs_root] + 2*K) - a**2*(5*F[locs_root] + K)+3*F[locs_root]*a**3
            sol = root(func, (2/3)*np.ones(len(a[locs_root])))
            a_new[locs_root] = sol.x
            a_line_new[locs_root] = (1 - 3*a_new[locs_root])/(4*a_new[locs_root] - 1)
        
        #  Step 7 - Check convergence: Root Mean Square Error
        error.append(np.sqrt(np.sum((a_new - a)**2)/len(a)))
        print(f'a_new = {len(a_new[a_new < 0])}')
        print(f'a_line_new ={len(a_line_new[a_line_new < 0])}')
        if error[-1] < tol:
            break
        else:
            a = a_new.copy()
            a[a < 0] = 0
            a_line = a_line_new.copy()
            a_line[a_line  < 0] = 0

    pass
if __name__ == '__main__':
    airfoil_name = 's834'
    rotor = blade_design('s834', 7, 2, number_of_sections=50,plot=False)
    bemt(6, rotor, airfoil_name, 2, iter=500)