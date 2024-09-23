#%% Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import BASE_COLORS
from pathlib import Path

#%% Colors
colors = list(BASE_COLORS.keys())

#%% Paths
image_path = Path('images')
image_path.mkdir(exist_ok=True)

airfoil_path = Path('airfoil')
#%% functions
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
              

def blade_design(
    airfoil_name:str, 
    tip_speed_ratio:float, 
    number_of_blades:int, 
    number_of_sections:int = 50, 
    section_distribution:str = 'uniform',
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
    
    #### Flow Angle and Pitch Angle
    line_max_re = df_opt.loc[df_opt['Re'].idxmax()]

    alpha_opt = line_max_re['alpha_opt']

    # axial induction factor
    x0 = 0.26
    xf = 0.3333
    points = np.linspace(0, 1, number_of_sections)
    
    if section_distribution.lower() == 'uniform':
        a = points *(xf - x0) + x0    
    elif section_distribution.lower() == 'sine':
        a = 0.5*(1 + np.cos(np.pi * points)) * (xf - x0) + x0
    elif section_distribution.lower() == 'exp':
        k = 5
        a = (1 - np.exp(-k * points))  * (xf - x0) + x0


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
        'r/R': x/tip_speed_ratio,  
        'phi': phi, 
        'theta': theta_opt
        } )

    # Plotting
    locs = df_sections['r/R'] <= 1 #valid points
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[10, 8])
    
    ax1.plot(x[locs]/tip_speed_ratio, theta_opt[locs],'k', label = 'Optimun pitch angle')
    ax1.plot(x[locs]/tip_speed_ratio, phi[locs], 'k--', label = 'Flow angle')

    ax1.set_title('Pitch and Flow Angle')
    # ax1.set_xlabel('r/R')
    ax1.set_ylabel(r'$\theta$, $\phi$ [deg]')

    ax1.grid()
    ax1.legend()
        
    ### Chord Distribution
    B = number_of_blades          
    phi_rad = np.deg2rad(phi[locs])

    # Tip Prandtl's correction function
    f = B/2/np.sin(phi_rad) * (tip_speed_ratio/x[locs] - 1)
    F = 2/np.pi * np.arccos(np.exp(-f))

    # Tangential force coefficient
    Ct = line_max_re['cl_opt']*np.sin(phi_rad) - line_max_re['cd_opt']*np.cos(phi_rad)

    # Solidity
    sigma = 4*x[locs]*a_line[locs] * np.sin(phi_rad)**2/(1 - a[locs])/Ct

    # Chord Distributio
    c_R = 2*np.pi*sigma*x[locs]/(B*tip_speed_ratio)
    
    ### Filter and Save data in dataframe
    df_sections = df_sections[df_sections['r/R'] <= 1]
    
    df_sections['sigma'] = sigma
    df_sections['c/R'] = c_R
    df_sections['Ct'] = Ct
    df_sections['Tip Correction'] = F
        
    ax2.plot(x[locs]/tip_speed_ratio, c_R, 'ks-',  label = 'Without Tip Correction')

    ax2.plot(x[locs]/tip_speed_ratio, c_R*F, 'ko--', label = 'With Tip Correction')

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
    df1 = blade_design('s834', 7, 2, number_of_sections=20,plot=False)
    df2 = blade_design('s834', 7, 2, number_of_sections=20, section_distribution='sine', plot=False)
    df3 = blade_design('s834', 7, 2, number_of_sections=20, section_distribution='exp', plot=False)

    # Comparison between each distribution type
    plt.plot(df2['r/R'],df2['c/R'], '^--',label = 'Sine')
    plt.plot(df3['r/R'],df3['c/R'], 'o--',label = 'Exp')
    plt.plot(df1['r/R'],df1['c/R'], 's--',label = 'Uniform')

    plt.grid()
    plt.legend()
    plt.show()
# %%
