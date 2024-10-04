import numpy as np
import pandas as pd

from pathlib import Path

#%% Paths
image_path = Path('images')
image_path.mkdir(exist_ok=True)

airfoil_path = Path('airfoil')
#%% Functions
def process_file(file_path) -> pd.DataFrame:
    save_data = False
    df = pd.DataFrame()
    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        for line in lines:
            if 'Average Reynolds #' in line:
                reynolds  = float(lines[i+1])
            
            if 'Number of angles of attack:' in line:
                number_of_aoa = int(lines[i+1])
            
            if 'alpha' in line:
                data = np.array([[float(a.split()[0]), float(a.split()[1]), float(a.split()[2])] for a in lines[i+1: i+1+number_of_aoa]])
                df[f'Re = {reynolds}'] = [data]
            
            i+=1
    
    return df

def tip_correction(phi:np.ndarray, R_r:np.ndarray, number_of_blades:int, model:str = '') -> np.ndarray:
    match model:
        case 'Prandtl':
            # Tip Prandtl's correction function
            f = number_of_blades/2/np.sin(phi) * (R_r - 1)
            F = 2/np.pi * np.arccos(np.exp(-f))
        case _:
            F = np.ones(len(R_r))
    
    return F

def coefficients_extrapolation(df_coeff_cl:pd.DataFrame, df_coeff_cd:pd.DataFrame, smooth:bool = False) -> pd.DataFrame:
    # Model for Large Angle of Attacks - From modified Hoerner flat plat coefficients
    alpha_extra_pos = np.linspace(25, 180, 200)
    alpha_extra_neg = np.linspace(-180, -25, 200)
    
    Cl_extra_pos =  np.sin(2*np.deg2rad(alpha_extra_pos))
    Cl_extra_neg =  np.sin(2*np.deg2rad(alpha_extra_neg))
    
    Cd_extra_pos =  1.3*np.sin(np.deg2rad(alpha_extra_pos))**2
    Cd_extra_neg =  1.3*np.sin(np.deg2rad(alpha_extra_neg))**2
    
    if smooth:
        # alpha_d = max(np.abs([df_coeff_cl['alpha'][0], df_coeff_cl['alpha'])[-1l]])) + 1
        # alpha_shift = np.deg2rad()
        # delta_alpha = np.deg2rad(10)
        # g = lambda alpha: 0.5*(1+np.tanh((alpha_d + alpha_shift  - np.abs(alpha))/delta_alpha))
        pass
    
    df_extra = pd.DataFrame(
        {
            'alpha': np.concatenate((alpha_extra_neg, df_coeff_cd[:,0] , alpha_extra_pos)),
            'Cl':  np.concatenate((Cl_extra_neg, df_coeff_cd[:,1], Cl_extra_pos)),
            'Cd':  np.concatenate((Cd_extra_neg, df_coeff_cd[:,2], Cd_extra_pos))

        }
    )    
    return  df_extra