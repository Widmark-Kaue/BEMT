import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from pathlib import Path

#%% Paths
image_path = Path('images')
image_path.mkdir(exist_ok=True)

airfoil_path = Path('airfoil')
validate_path = Path('validate')
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

def coefficients_extrapolation(Cl_mat:np.ndarray, Cd_mat:np.ndarray, smooth:bool = True, alpha_shift:float = 5, delta_alpha:float =5, delta_alpha_d:float = 0, interpolate_type:str = 'linear') -> pd.DataFrame:
    
    #Eliminate duplicates angle of attack values
    _, idx = np.unique(Cl_mat[:, 1], return_index=True)
    Cl_mat = Cl_mat[idx, :]

    _, idx = np.unique(Cd_mat[:, 1], return_index=True)
    Cd_mat = Cd_mat[idx, :]
    
    # funcs of experimental data - C._known
    Cl_known = interp1d(np.deg2rad(Cl_mat[:,1]), Cl_mat[:,0],  kind=interpolate_type, fill_value="extrapolate")
    Cd_known = interp1d(np.deg2rad(Cd_mat[:,1]), Cd_mat[:,0], kind=interpolate_type, fill_value="extrapolate")
    
    # Model for Large Angle of Attacks - From modified Hoerner flat plat coefficients
    alpha_d_cl_pos = np.deg2rad(Cl_mat[-1,1] + delta_alpha_d)
    alpha_d_cl_neg = np.deg2rad(abs(Cl_mat[0,1]) + delta_alpha_d)
    
    alpha_d_cd_pos = np.deg2rad(Cd_mat[-1,1] + delta_alpha_d)    
    alpha_d_cd_neg = np.deg2rad(abs(Cd_mat[0,1]) + delta_alpha_d)    
    
    alpha_extra = np.deg2rad(np.linspace(-180, 180, 1000))
    alpha_extra_pos = alpha_extra[alpha_extra >= 0]
    alpha_extra_neg = alpha_extra[alpha_extra < 0]
    
    Cl_high = lambda alpha: np.sin(2*alpha)
    Cd_high =  lambda alpha: 1.3*np.sin(alpha)**2

    if smooth:
        g = lambda alpha, alpha_d, alpha_shift = np.deg2rad(alpha_shift), delta_alpha = np.deg2rad(delta_alpha): 0.5*(1+np.tanh((alpha_d + alpha_shift  - np.abs(alpha))/delta_alpha))
        
        g_cl_pos = lambda alpha: g(alpha, alpha_d = alpha_d_cl_pos)
        g_cl_neg = lambda alpha: g(alpha, alpha_d = alpha_d_cl_neg)
        
        g_cd_pos = lambda alpha: g(alpha, alpha_d = alpha_d_cd_pos)
        g_cd_neg = lambda alpha: g(alpha, alpha_d = alpha_d_cd_neg)
    
    df_extra = pd.DataFrame(
        {
            'alpha': alpha_extra,
            'Cl':  np.concatenate(
                (
                    g_cl_neg(alpha_extra_neg) * Cl_known(alpha_extra_neg) + (1 - g_cl_neg(alpha_extra_neg)) * Cl_high(alpha_extra_neg),
                    g_cl_pos(alpha_extra_pos) * Cl_known(alpha_extra_pos) + (1 - g_cl_pos(alpha_extra_pos)) * Cl_high(alpha_extra_pos)
                )
                ),
            'Cd':  np.concatenate(
                (
                    g_cd_neg(alpha_extra_neg) * Cd_known(alpha_extra_neg) + (1 - g_cd_neg(alpha_extra_neg)) * Cd_high(alpha_extra_neg),
                    g_cd_pos(alpha_extra_pos) * Cd_known(alpha_extra_pos) + (1 - g_cd_pos(alpha_extra_pos)) * Cd_high(alpha_extra_pos)
                )
                )
        }
    )    
    return  df_extra