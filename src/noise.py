import numpy as np

from scipy.special import jn, jv
from struct import Rotor
from dataclasses import dataclass, field

@dataclass
class TonalNoiseModel:
    rotor: Rotor
    microphones: np.ndarray             # matrix with number of rows = number of microphones and number of columns = 2
    attached: bool = field(default=True)
    number_of_harmonics: int
    
    
    def hanson(self, hansen_distribution_aproximation:bool = True)-> tuple:
        pass
    
    def hansenTransient(self)-> tuple:
        pass
    
    def microfones_cartesian(self)-> np.ndarray:
        x = self.microphones[:,0]*np.cos(self.microphones[:,1])
        y = self.microphones[:,0]*np.sin(self.microphones[:,1])            
        return np.array([x,y]).T


    def psiVDL(self, kx: float, hansen_aproximation:bool = True)-> tuple:
        """
        psiV, psiD, psiL
        """
        if hansen_aproximation:
            if kx == 0:
                return 2/3, 1 ,1
            else:
                V = 8/(kx**2) * ( 2/kx * np.sin(0.5*kx) - np.cos(0.5*kx) )
                DL = 2/kx * np.sin(0.5*kx)
                return V, DL, DL
