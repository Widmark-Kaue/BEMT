import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import simpson
from scipy.optimize import root, brentq
from rotor import Rotor
from utils import *

from dataclasses import dataclass, field
from typing import Iterable, Union
from time import time
@dataclass
class BEMTResult:
    rotor: Rotor
    TSR: Iterable[float]
    tip_correction_model: str
    threeD_correction: bool
    def __post_init__(self):
        self.CP = np.zeros(len(self.TSR))
        self.CT = np.zeros(len(self.TSR))
        
        self.dCP = np.zeros((self.rotor.number_of_sections_useful, len(self.TSR)))
        self.dCT = np.zeros((self.rotor.number_of_sections_useful, len(self.TSR)))
        self.dCn = np.zeros((self.rotor.number_of_sections_useful, len(self.TSR)))
        self.dCt = np.zeros((self.rotor.number_of_sections_useful, len(self.TSR)))
        
        self.phi = np.zeros((self.rotor.number_of_sections_useful, len(self.TSR)))
        self.alpha = np.zeros((self.rotor.number_of_sections_useful, len(self.TSR)))
        
        self.a = np.zeros((self.rotor.number_of_sections_useful, len(self.TSR)))
        self.a_line = np.zeros((self.rotor.number_of_sections_useful, len(self.TSR)))
        self.x = np.zeros((self.rotor.number_of_sections_useful, len(self.TSR)))
        
        self.iterations = np.zeros(len(self.TSR))
        self.converged = np.zeros(len(self.TSR))
        self.error = list()

def bemt(rotor: Rotor,TSR:Union[float, Iterable], Cd_null:bool = False, tip_correction_model:str = 'Prandtl', threeD_correction: bool = False, iter:int = 100, tol:float = 1e-3) -> BEMTResult:
    
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
        coeff_extra = coefficients_extrapolation(Cl_mat,Cd_mat, alpha_shift=5)
        Cl_extra = interp1d(coeff_extra['alpha'],  coeff_extra['Cl'])
        Cd_extra = interp1d(coeff_extra['alpha'],  coeff_extra['Cd'])
    else:
        Cl_extra = lambda alpha: rotor.Cl_opt
        Cd_extra = lambda alpha: rotor.Cd_opt
        
    
    if Cd_null:
        Cd_extra = lambda alpha: 0
            
    # Step 0 - Prepare geometric parameters (Using intermediate points to avoid edge effects)
    r_R = rotor.sections['r_R'].to_numpy()[1:-1]
    c_R = rotor.sections['c_R'].to_numpy()[1:-1] * rotor.sections['tip_correction'].to_numpy()[1:-1]
    sigma = rotor.sections['sigma'].to_numpy()[1:-1] * rotor.sections['tip_correction'].to_numpy()[1:-1]
    theta = np.deg2rad(rotor.sections['theta_opt'].to_numpy()[1:-1])
    
    if not isinstance(TSR, Iterable):
        TSR = np.array([TSR])
        
    # Step 0.5 - Prepare BEMT parameters
    results = BEMTResult(rotor = rotor, TSR = TSR, tip_correction_model=tip_correction_model, threeD_correction=threeD_correction)
    
    for k, tsr in enumerate(TSR):  
        x = r_R*tsr                 

        # Step 1 - Initialize the BEMT parameters
        a =  rotor.sections['a'].to_numpy()[1:-1]
        a_line = rotor.sections['a_line'].to_numpy()[1:-1]
        a_new =  np.zeros(len(x))
        a_line_new = np.zeros(len(x))
        
        phi = np.zeros(len(x))
        alpha = np.zeros(len(x))
        Ct = np.zeros(len(x))
        Cn = np.zeros(len(x))

        error = list()
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
                
                sol = root(func, (1/3)*np.ones(len(a[locs_root])), method = 'lm')
                assert sol.success, "The root find process to axial induction factor failed. Consider increasing the number of iterations"
                # assert np.all(sol.x <= 0.5), "The axial induction factor is greater than 0.5. Consider increasing the number of iterations"
                a_new[locs_root] = sol.x
            
            # Update rotational induction factor
            a_line_new = 4*F*np.sin(phi) *  np.cos(phi)/(sigma * Ct) - 1
            a_line_new = 1/a_line_new
            
            #  Step 7 - Check convergence: Root Mean Square Error
            error_a = np.sqrt(np.sum((a_new - a)**2)/len(a))
            error_a_line = np.sqrt(np.sum((a_line_new - a_line)**2)/len(a_line))
            error.append([error_a, error_a_line]) 
        
            if  np.all(np.array(error[-1]) < tol):
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
        dC_P = 4 * np.pi * sigma**2 * (1- a)**3 * Ct/(c_R * rotor.number_of_blades * tsr * (1 + a_line) * np.tan(phi) * np.sin(phi)**2)
        dC_T = rotor.number_of_blades * Cn * (1 - a)**2 *c_R/(np.sin(phi)**2 * np.pi * tsr)
        dC_P = x * rotor.number_of_blades * Ct * (1 - a)**2 *c_R/(np.sin(phi)**2 * np.pi*tsr)
        
        # Include boundary conditions
        dC_T = np.concatenate(([0], dC_T, [0]))
        dC_P = np.concatenate(([0], dC_P, [0]))

        
        # Step 9 - Compute total thrust and power
        x_full = rotor.sections['r_R'].to_numpy()*tsr
        C_T = simpson(dC_T, x = x_full)
        C_P = simpson(dC_P, x = x_full)
        
        # Step 10 - Store results
        results.CP[k] = C_P
        results.CT[k] = C_T
        results.dCP[:,k] = dC_P
        results.dCT[:,k] = dC_T
        results.dCn[1:-1,k] = Cn
        results.dCt[1:-1,k] = Ct
        results.phi[1:-1,k] = phi
        results.alpha[1:-1,k] = alpha
        
        results.a[1:-1,k] = a
        results.a_line[1:-1,k] = a_line
        results.x[:,k] = x_full
        
        
        results.iterations[k] = i
        results.converged[k] = converged
        results.error.append(np.array(error))
            
    
    return results
if __name__ == '__main__':
    tsr = 7
    rotor = Rotor(number_of_blades=2, number_of_sections=100, tip_correction_model='',tip_speed_ratio=tsr, airfoil_name='s834')
    rotor.load_airfoil_prop(plot=False)
    rotor.blade_design(r_R0=0.11,plot=False)
    results = bemt(rotor=rotor, TSR=7, Cd_null=True, tip_correction_model='',iter=500, tol=1e-3)
    pass