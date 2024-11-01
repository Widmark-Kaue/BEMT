import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json
import pickle

from scipy.optimize import root
from scipy.integrate import simpson
from matplotlib.colors import BASE_COLORS
from src.utils import *

from dataclasses import dataclass, field

@dataclass
class Rotor:
    number_of_blades: int
    tip_speed_ratio: float
    number_of_sections: int
    number_of_sections_useful:int = field(default=None, repr=False, init=False)
    airfoil_name: str = field(default=None, repr=False)
    airfoil_coord: np.ndarray = field(default_factory = lambda: np.empty((1, 2)), repr=False, init=False)
    alpha_opt: float = field(default=None, repr=False)
    Cl_opt: float = field(default=None, repr=False)
    Cd_opt: float = field(default=None, repr=False)
    tip_correction_model:str = field(default=None)
    rotor_name:str = field(default=None, repr=False)
    CP_opt:float = field(init=False, default=None, repr=False)
    CT_opt:float = field(init=False, default=None, repr=False)
    sections:pd.DataFrame = field(init = False, default_factory = pd.DataFrame, repr = False)
    
    # A class attribute to count the number of instances
    _rotor_count: int = field(init=False, default=0, repr=False)

    def __post_init__(self):
        # Increment the class attribute
        type(self)._rotor_count += 1
        
        # if rotor name is not defined, define it
        if self.rotor_name is None:
            self.rotor_name = f'Rotor {self._rotor_count}'
    
    def load_airfoil_prop(self, plot:bool = True):
        assert self.airfoil_name is not None, 'Airfoil name is not defined'
        
        #Load airfoil points
        airfoil_coord = np.loadtxt(airfoil_path.joinpath(f'{self.airfoil_name}.dat'), skiprows=1)

        ####  Load c_drg file and plot Cl/Cd x alpha
        c_lift_drag = process_file(airfoil_path.joinpath(f'{self.airfoil_name}_c_drg.txt'))
        cols = c_lift_drag.columns

        alpha_opt = []
        cl_opt  = []
        cd_opt = []
        re = []
        
        
        if plot:
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
            
            if plot:
                ax.plot(alpha, LD, colors[i], label=f'Re = {re[-1]:.3e},'+ r' $\alpha_{opt} =$'+ f'{alpha_opt[-1]:.1f}°')
                ax.plot(alpha_opt[-1], max(LD), f'{colors[i]}o')


        if plot:
            ax.set_title(r'L/D x $\alpha$')
            ax.set_xlabel(r'$\alpha$ [deg]')
            ax.set_ylabel('Cl/Cd')

            ax.legend()
            ax.grid()
            
            # Arifoil plot
            ax_inset = fig.add_axes([0.65, 0.15, 0.25, 0.25])
            ax_inset.plot(airfoil_coord[:,0], airfoil_coord[:,1], 'k', label = self.airfoil_name)
            
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            ax_inset.set_xticklabels([])
            ax_inset.set_yticklabels([])
            ax_inset.axis('equal')
            ax_inset.legend()
        
            plt.show()
 
        df_opt = pd.DataFrame({
            'Re': re,
            'alpha_opt': alpha_opt,
            'cl_opt': cl_opt,
            'cd_opt': cd_opt,
        })
        
        line_max_re = df_opt.loc[df_opt['Re'].idxmax()]

        self.alpha_opt = line_max_re['alpha_opt']
        self.Cl_opt = line_max_re['cl_opt']
        self.Cd_opt = line_max_re['cd_opt']
        self.airfoil_coord = airfoil_coord
        
        
    def blade_design(self, r0_R:float = 0.1, solidity:str = 'Cn', plot:bool = True, filter_invalid_solidity:bool = True):
        ### Define stations and local rotational speed ratio
        r_R = np.linspace(r0_R, 1, self.number_of_sections)
        x = r_R*self.tip_speed_ratio
        
        ### Induction Factors
        # axial induction factor
        a_line_func = lambda a : (1 - 3*a)/(4*a - 1)
        a_root = lambda a: a*(1 - a) - x**2 * a_line_func(a)* (1 + a_line_func(a))
        sol = root(lambda a: a_root(a), 0.26 * np.ones(len(x)), method = 'lm')
        a = sol.x
        
        assert np.all(a < 1/3), 'No valid point finded to axial induction factor'    
                
        # rotational induction factor
        a_line = (1 - 3 *a)/(4*a - 1)
        
        #### Flow Angle and Pitch Angle
        # Flow angle
        phi = np.rad2deg(np.arctan((1 - a)/(1 + a_line)/x))

        # Pitch angle
        theta_opt = phi - self.alpha_opt

        ### Chord Distribution
        phi_rad = np.deg2rad(phi)

        # Tip Correction
        F = tip_hub_correction(phi_rad, self.tip_speed_ratio/x, self.number_of_blades, model = self.tip_correction_model)
        
        # Tangential and normal force coefficient
        Ct = self.Cl_opt*np.sin(phi_rad) - self.Cd_opt*np.cos(phi_rad)
        Cn = self.Cl_opt*np.cos(phi_rad) + self.Cd_opt*np.sin(phi_rad)
        
        # Solidity
        match solidity:
            case 'Cn':
                sigma = 4*a*np.sin(phi_rad)**2 / (1 - a)/Cn
            case 'Ct':
                sigma = 4*x*a_line * np.sin(phi_rad)**2/(1 - a)/Ct
            case 'Cl':             
                sigma = 4*x*a_line/np.sqrt((1 - a)**2 + x**2*(1+a_line)**2)/self.Cl_opt
        
        if np.any(sigma > 1) and filter_invalid_solidity: 
            valid_points = sigma < 1
            sigma = sigma[valid_points]
            r_R = r_R[valid_points]
            x = x[valid_points]
            a = a[valid_points]
            a_line = a_line[valid_points]
            phi = phi[valid_points]
            theta_opt = theta_opt[valid_points]
            F = F[valid_points]
            Cn = Cn[valid_points]
            Ct = Ct[valid_points]
            print(fr'Warning TSR = {self.tip_speed_ratio}: solidity > 1 - This Points will be ignored')
        
        self.number_of_sections_useful = len(sigma)
        
        # Chord Distribution
        c_R = 2*np.pi*sigma*x/(self.number_of_blades*self.tip_speed_ratio)
        
        # Save sections        
        self.sections = pd.DataFrame(
            {
                'a': a,
                'a_line': a_line,
                'x': x,
                'r_R': r_R,
                'phi': phi,
                'theta_opt': theta_opt,
                'Ct': Ct,
                'sigma': sigma,
                'c_R': c_R,
                'tip_correction': F
            }
        )
        
        self.CP_opt = 8/(self.tip_speed_ratio**2) * simpson(a_line*(1 - a)*x**3, x = x)
        self.CT_opt = simpson(4*a*(1-a)*F, x = x)
        
        # Plotting
        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[10, 8])
            
            ax1.plot(x/self.tip_speed_ratio, theta_opt,'k', label = 'Optimun pitch angle')
            ax1.plot(x/self.tip_speed_ratio, phi, 'k--', label = 'Flow angle')

            ax1.set_title('Pitch and Flow Angle')
            
            # ax1.set_xlabel('r/R')
            ax1.set_ylabel(r'$\theta$, $\phi$ [deg]')

            ax1.grid()
            ax1.legend()
            
            ax2.plot(r_R, c_R, 'ks-',  label = 'Without Tip Correction')

            ax2.plot(r_R, c_R*F, 'ko--', label = 'With Tip Correction')

            ax2.set_title('Chord Distribution')
            ax2.set_xlabel(r'$r/R$')
            ax2.set_ylabel(r'$c/R$')

            ax2.legend()
            ax2.grid()
            plt.show()
    
    def blade_geometry(self) -> tuple:
        r_R = self.sections['r_R'].to_numpy()
        c_R = self.sections['c_R'].to_numpy() * self.sections['tip_correction'].to_numpy()
        theta = np.deg2rad(self.sections['theta_opt'].to_numpy())
        airfoil_coord = self.airfoil_coord
        
        le = np.zeros((len(r_R), 2))
        te = np.zeros((len(r_R), 2))
        le[:,0] = r_R 
        te[:,0] = r_R 
        
        le[:,1] = c_R/2 
        te[:,1] = - c_R/2 
       
        return le, te
    
    @staticmethod
    def save(obj, name:str):
        if not name.endswith('.pkl'):
            name = name + '.obj'
        with open(rotor_path.joinpath(name), 'wb') as file:
            pickle.dump(obj, file)
    @staticmethod
    def load(name:str):
        path = rotor_path.joinpath(name)
        if path.exists():
            if name.endswith('.obj'):
                with open(path, 'rb') as file:
                    return pickle.load(file)
            elif name.endswith('.json'):
                with open(path, 'r') as file:
                    data = json.load(file)
                    tip_speed_ratio = data['R']*data['omega']/data['V0']
                    rotor = Rotor(
                        number_of_blades = data['number_of_blades'],
                        tip_speed_ratio=tip_speed_ratio,
                        number_of_sections=len(data['sections']['r']),
                        Cl_opt=data['Cl'],
                        Cd_opt=data['Cd'],
                        )
                    
                    
                    
                    rotor.number_of_sections_useful = len(data['sections']['r'])
                    sections = pd.DataFrame(data['sections'])
                    sections['r_R'] = sections['r']/data['R']
                    sections['x'] = sections['r_R']*rotor.tip_speed_ratio
                    sections['c_R'] = sections['c']/data['R']
                    sections['sigma'] = sections['c']*rotor.number_of_blades/(2*np.pi*sections['r'])
                    
                    sections['theta_opt'] = sections['theta_p'] + data['beta']
                    sections['phi'] = np.arctan( (1-sections['a'])/(1+sections['a_line'])/sections['x'])
                    
                    
                    Vrel = data['V0']*(1 - sections['a'])/np.sin(sections['phi'])
                    factor = 0.5*data['rho']*sections['c']*Vrel**2  
                    sections['Ct'] = sections['pT']/factor
                    sections['Cn'] = sections['pN']/factor

                    rotor.sections = sections
                    return rotor