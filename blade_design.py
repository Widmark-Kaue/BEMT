#%% Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import root
from matplotlib.colors import BASE_COLORS
from utils import *

def blade_design(
    airfoil_name:str, 
    tip_speed_ratio:float, 
    number_of_blades:int, 
    number_of_sections:int = 50, 
    tip_correction_model:str = 'Prandtl',
    plot:bool = True
    ) -> pd.DataFrame:
    
    
    #Load airfoil points
    x, y = np.loadtxt(airfoil_path.joinpath(f'{airfoil_name}.dat'), skiprows=1, unpack=True)

    ####  Load c_drg file and plot Cl/Cd x alpha
    c_lift_drag = process_file(airfoil_path.joinpath(f'{airfoil_name}_c_drg.txt'))
    cols = c_lift_drag.columns

    alpha_opt = []
    cl_opt  = []
    cd_opt = []
    re = []
    
    # Colors for plotting
    colors = list(BASE_COLORS.keys())
    
    fig, ax = plt.subplots(figsize=[10, 4.8])
    
    for i, col in enumerate(cols):
        re.append(float(col.split()[-1]))
        
        LD = c_lift_drag[col][0][:, 1]/c_lift_drag[col][0][:, 2]
        alpha = c_lift_drag[col][0][:,  0]
        
        opt_loc = np.where(LD == max(LD))[0][0]
        alpha_opt.append(alpha[opt_loc])
        cl_opt.append(c_lift_drag[col][0][opt_loc, 1])
        cd_opt.append(c_lift_drag[col][0][opt_loc, 2])
        
        ax.plot(alpha, LD, colors[i], label=f'Re = {re[-1]:.3e},'+ r' $\alpha_{opt} =$'+ f'{alpha_opt[-1]:.1f}°')
        ax.plot(alpha_opt[-1], max(LD), f'{colors[i]}o')

    ax.set_title(r'L/D x $\alpha$')
    ax.set_xlabel(r'$\alpha$ [deg]')
    ax.set_ylabel('Cl/Cd')

    ax.legend()
    ax.grid()
    
    # Arifoil plot
    ax_inset = fig.add_axes([0.65, 0.15, 0.25, 0.25])
    ax_inset.plot(x, y, 'k', label = airfoil_name)
    
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_xticklabels([])
    ax_inset.set_yticklabels([])
    ax_inset.axis('equal')
    ax_inset.legend()
    
    if plot:
        plt.show()
    else:
        plt.close()

    df_opt = pd.DataFrame({
        'Re': re,
        'alpha_opt': alpha_opt,
        'cl_opt': cl_opt,
        'cd_opt': cd_opt,
    })
    
    
    ### Define stations and local rotational speed ratio
    r_R = np.linspace(0.11, 1, number_of_sections)
    x = r_R*tip_speed_ratio
    
    ### Induction Factors
    # axial induction factor
    a_line_func = lambda a : (1 - 3*a)/(4*a - 1)
    a_root = lambda a: a*(1 - a) - x**2 * a_line_func(a)* (1 + a_line_func(a))
    sol = root(lambda a: a_root(a), 0.26 * np.ones(len(x)), method = 'lm')
    a = sol.x
    
    assert np.all(a < 1/3), 'No valid point finded to axial induction factor'    
            
    # tangential induction factor
    a_line = (1 - 3 *a)/(4*a - 1)
    
    #### Flow Angle and Pitch Angle
    line_max_re = df_opt.loc[df_opt['Re'].idxmax()]

    alpha_opt = line_max_re['alpha_opt']

    # Flow angle
    phi = np.rad2deg(np.arctan((1 - a)/(1 + a_line)/x))

    # Pitch angle
    theta_opt = phi - alpha_opt

    ### Chord Distribution
    B = number_of_blades          
    phi_rad = np.deg2rad(phi)

    F = tip_correction(phi_rad, tip_speed_ratio/x, number_of_blades, model = tip_correction_model)

    # Tangential force coefficient
    Ct = line_max_re['cl_opt']*np.sin(phi_rad) - line_max_re['cd_opt']*np.cos(phi_rad)

    # Solidity
    sigma = 4*x*a_line * np.sin(phi_rad)**2/(1 - a)/Ct

    # Chord Distribution
    c_R = 2*np.pi*sigma*x/(B*tip_speed_ratio)
    
    ### Save rotor parameters only for the sigma <=1 stations    
    df_sections = pd.DataFrame({
        'a':  a, 
        'a_line': a_line, 
        'x': x, 
        'r/R': r_R,  
        'phi': phi, 
        'theta': theta_opt,
        'Ct': Ct,
        'sigma': sigma,
        'c/R': c_R,
        'Tip Correction': F
        } )
        
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[10, 8])
    
    ax1.plot(x/tip_speed_ratio, theta_opt,'k', label = 'Optimun pitch angle')
    ax1.plot(x/tip_speed_ratio, phi, 'k--', label = 'Flow angle')

    ax1.set_title('Pitch and Flow Angle')
    
    # ax1.set_xlabel('r/R')
    ax1.set_ylabel(r'$\theta$, $\phi$ [deg]')

    ax1.grid()
    ax1.legend()
    
    ax2.plot(x/tip_speed_ratio, c_R, 'ks-',  label = 'Without Tip Correction')

    ax2.plot(x/tip_speed_ratio, c_R*F, 'ko--', label = 'With Tip Correction')

    ax2.set_title('Chord Distribution')
    ax2.set_xlabel(r'$r/R$')
    ax2.set_ylabel(r'$c/R$')

    ax2.legend()
    ax2.grid()
    if plot:
        plt.show()
    else:
        plt.close()
        
    return df_sections

#%% Example

if __name__ ==  "__main__":
    df1 = blade_design('s834', 7, 2, plot=False)
# %%
