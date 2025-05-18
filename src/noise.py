import numpy as np
import inspect

from scipy.special import jv
from typing import Union
from dataclasses import dataclass, field


@dataclass
class noise:
    microphones: np.ndarray
    sound_speed: float = field(default=340)
    density:float = field(default=1.22)
    
    @property
    def microphones_to_polar(self)-> np.ndarray:
        x = self.microphones[:,0]
        y = self.microphones[:,1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arccos(x/r)
        return np.array([r,theta]).T


@dataclass
class farfield(noise):
    
    
    
    """ Funções auxiliares """
    # def __check_function__(self, method, **kwargs)-> any:
    #     if not hasattr(self, method):
    #         raise ValueError(f"Method '{method}' not found in TonalNoiseModel.")
        
    #     method_fn = getattr(self, method)
    #     sig = inspect.signature(method_fn)
    #     param_dict = sig.parameters

    #     # Lista de argumentos esperados (exceto 'self')
    #     expected_args = [k for k in param_dict if k != 'self']
        
    #     # Checagem de argumentos inesperados
    #     for k in kwargs:
    #         if k not in expected_args:
    #             raise ValueError(f"Argumento inesperado '{k}' para o método '{method}'. Esperados: {expected_args}")

    #     # Checagem de argumentos obrigatórios (sem default)
    #     required_args = [
    #         k for k, v in param_dict.items()
    #         if k not in ['self', 'number_of_harmonics']
    #         and v.default is inspect.Parameter.empty
    #     ]
    #     missing = [k for k in required_args if k not in kwargs]
    #     if missing:
    #         raise ValueError(f"Argumentos obrigatórios ausentes para o método '{method}': {missing}")
    #     return method_fn
        
    def __psiVDL__(self, kx: float, hanson_aproximation:bool = True)-> tuple:
        """
        psiV, psiD, psiL
        """
        if hanson_aproximation:
            if kx == 0:
                return 2/3, 1 ,1
            else:
                V = 8/(kx**2) * ( 2/kx * np.sin(0.5*kx) - np.cos(0.5*kx) )
                DL = 2/kx * np.sin(0.5*kx)
                return V, DL, DL
    
    """ Methods """
    # def loading_noise(self, method:str = 'hansonReff', number_of_harmonics:int = 1, **kwargs):
    #     method_fn = self.__check_function__(method, **kwargs)
        
    #     # Chamada do método
    #     return method_fn(number_of_harmonics=number_of_harmonics, **kwargs)

    
    def hansonReff(
        self, 
        number_of_harmonics:int , 
        number_of_blades:int, 
        Mt:float, 
        rtip: float, 
        BD:float, 
        loading:np.array, 
        Mx:float=0,
        zeff = 0.8, 
        hanson_distribution_aproximation:bool = True, 
        include_imag_part:bool = False,
        rms:bool = False
        ) -> tuple:
        
        # Define short variable names
        B = number_of_blades
        T, Q = loading
        Mr = np.sqrt(Mx**2 + zeff**2 * Mt**2)
        
        # Microphone positions
        nmics = self.microphones.shape[0]
        yVec = self.microphones[:,1]
        thetaVec = self.microphones_to_polar[:, 1]
        
        # Initialize variables
        PLoad = np.zeros((nmics, number_of_harmonics))
        if hanson_distribution_aproximation:
            psiLFunc = lambda x: self.__psiVDL__(x)[2]
        else:   
            assert False, "Distribution not implemented"
            
        
        for imic in range(nmics):
            y = yVec[imic]
            theta = thetaVec[imic]
            # Cache
            sinthe = np.sin(theta)
            costhe = np.cos(theta)
            
            # Cache for Thrust and torque term
            ThrustTerm = costhe*T/(1 - Mx*costhe)
            TorqueTerm =Q/(zeff**2 * Mt*rtip)
            for m in range(1, number_of_harmonics+1):
                
                #Wave number and Source Transform
                kx = 2*m*B*BD*Mt/(Mr*(1 - Mx*costhe))
                psiL = psiLFunc(kx)
                
                # Bessel Term
                besselTerm = jv(m*B, m*B*zeff*Mt*sinthe/(1 - Mx*costhe))      
                
                # Calculate Pload
                PLoad[imic, m-1] = m*B*Mt*sinthe/(2*np.pi*y*rtip*(1 - Mx*costhe)) * (ThrustTerm - TorqueTerm)*psiL*besselTerm
        
        
        
        if include_imag_part:
            PLoad = PLoad*1j
            
        if rms:
            Prms = np.abs(PLoad) *np.sqrt(2)/2
            return Prms
        return PLoad
        
    def garrickWatkinsReff(
        self, 
        number_of_harmonics:int,
        number_of_blades:int, 
        reff: float,
        loading:np.array,
        Mtip:float, 
        Mx:float=0, 
        ):
         
        # Define short variable names
        B = number_of_blades
        m = np.arange(1, number_of_harmonics+1)
        k = m*B*Mtip/(reff)
        T, Q = loading
        
        # Microphone positions
        nmics = self.microphones.shape[0]
        x = self.microphones[:,0]
        y = self.microphones[:,1]
        
        beta = np.sqrt(1-Mx**2)
        s0 = np.sqrt(x**2 + beta**2 * y**2)
        sigma = (Mx*x + s0)/(beta**2)
        # Initialize variables
        PT = np.zeros((nmics, number_of_harmonics))
        PQ = np.zeros((nmics, number_of_harmonics))
        for imic in range(nmics):
            
            cte1 = k/(2*np.pi*s0[imic])
            cte2 = jv(m*B, k*y[imic]*reff/s0[imic])
            
            PT[imic] = cte1* ( T*(Mx + x[imic]/s0[imic])*(1/(beta**2)) ) * cte2
            
            PQ[imic] = cte1* (- Q*B*m/(k*reff**2)) * cte2
        
            
             
            # PLoad[imic] = k/(2*np.pi*s0[imic])
            
            # PLoad[imic] *= np.abs(T*(Mx + x[imic]/s0[imic])*(1/(beta**2)) - Q*B*m/(k*reff**2))
            
            # PLoad[imic] *= jv(m*B, k*y[imic]*reff/s0[imic]) 
        
        Prms = np.abs(PT+PQ)*np.sqrt(2)/2
        # PTrms = np.abs(PT)*np.sqrt(2)/2
        # PQrms = np.abs(PQ)*np.sqrt(2)/2
        # # Prms = PTrms + PQrms
        return Prms
    

           
        
    
    def hansonSteady(self, hanson_distribution_aproximation:bool = True)-> tuple:
       
        
        # Define short variable names
        number_of_mics = self.microphones.shape[0]
        c = self.sound_of_speed
        rho = self.density
        
        for imic in range(number_of_mics):
            r, theta = self.microphones[imic]
            theta = np.deg2rad(theta)
            for m in self.harmonics:
                pass
            
            
    

    
    


    
