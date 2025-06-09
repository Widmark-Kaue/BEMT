import numpy as np
import inspect

from scipy.special import jv
from typing import Union, Iterable
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
        
    def __psiVDL__(self, kx_v: Iterable, hanson_aproximation:bool = True) -> np.ndarray:
        """
        psiV, psiD, psiL
        """
       
        if not isinstance(kx_v, Iterable):
            kx_v = [kx_v]
            
        psi = np.zeros((len(kx_v), 3)) # type: ignore
        
        for i, kx in enumerate(kx_v):
            if hanson_aproximation:
                if kx == 0:
                    psi[i] = np.array([2/3, 1 ,1])
                else:
                    V = 8/(kx**2) * ( 2/kx * np.sin(0.5*kx) - np.cos(0.5*kx) )
                    DL = 2/kx * np.sin(0.5*kx)
                    psi[i] = np.array([V, DL, DL])
        
        return psi

    
    def hansonReff(
        self, 
        number_of_harmonics:int , 
        number_of_blades:int, 
        Mt:float, 
        rtip: float, 
        BD:float, 
        loading:Iterable, 
        Mx:float=0,
        zeff = 0.8, 
        hanson_distribution_aproximation:bool = True, 
        phase:bool = False
        ) -> np.ndarray:
        
        # Define short variable names
        B = number_of_blades
        T, Q = loading
        Mr = np.sqrt(Mx**2 + zeff**2 * Mt**2)
        omega_c0 = Mt/rtip

        
        # Microphone positions
        nmics = self.microphones.shape[0]
        yVec = np.sqrt(self.microphones[:,1]**2 + self.microphones[:, 2]**2)
        theta1Vec = self.microphones_to_polar[:, 1]
        thetaVec = np.arccos( np.cos(theta1Vec) * np.sqrt(1 - Mx**2 * np.sin(theta1Vec)**2) + Mx * np.sin(theta1Vec)**2)  
        
        SrVec = yVec/np.sin(thetaVec)
         
        # Initialize variables
        PLoad = np.zeros((nmics, number_of_harmonics), dtype=complex)
        if hanson_distribution_aproximation:
            psiLFunc = lambda x: self.__psiVDL__(x)[0,2]
        else:   
            assert False, "Distribution not implemented"
        
        
        for imic in range(nmics):
            y = yVec[imic]
            theta = thetaVec[imic]
            Sr = SrVec[imic]
            
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
                
                if phase:
                    PLoad[imic, m -1] *= np.exp(1j*m*B*(omega_c0*Sr - np.pi/2))
        
        
      
            
        
        Prms = np.abs(PLoad) *np.sqrt(2)/2
        return Prms
        
    def garrickWatkinsReff(
        self, 
        number_of_harmonics:int,
        number_of_blades:int, 
        reff: float,
        loading:Iterable,
        Mrot:float, 
        Mx:float=0, 
        ):
         
        # Define short variable names
        B = number_of_blades
        m = np.arange(1, number_of_harmonics+1)
        k = m*B*Mrot/(reff)
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
        
        
        Prms = np.abs(PT+PQ)*np.sqrt(2)/2
        
        return Prms
    
    
    def hansonSteady(
        self,
        number_of_harmonics:int , 
        number_of_blades:int, 
        Mt:float, 
        rtip: float, 
        Mx :float,
        z:np.ndarray,
        b:np.ndarray,
        MCA:np.ndarray,
        loading:list[np.ndarray],
        hanson_distribution_aproximation:bool = True):
       

        
        # Define short variable names
        c = self.sound_speed
        rho = self.density
        D = 2*rtip       
        B = number_of_blades
        omega_c0 = Mt/rtip
        
        # Distribution as function of radius
        BD = b/D
        dTdr, dQdr = loading
        Mr = np.sqrt(Mx**2 + z**2 * Mt**2)
        
        message = 'Arrays with properties distribuied by the radius needed had the same size'
        # assert len(dTdr) == len(dQdr) == len(z) == len(b) == len(MCA), message
        
        # Microphone positions
        nmics = self.microphones.shape[0]
        theta1Vec = self.microphones_to_polar[:, 1]
        thetaVec = np.arccos( np.cos(theta1Vec) * np.sqrt(1 - Mx**2 * np.sin(theta1Vec)**2) + Mx * np.sin(theta1Vec)**2)  
        
        YVec = np.sqrt(self.microphones[:, 1]**2 + self.microphones[:, 2]**2) # Far-Field aproximation y >> z
        SrVec = YVec/np.sin(thetaVec)
        
        # Initialize variables
        PLoad = np.zeros((nmics, number_of_harmonics), dtype=complex)
        if hanson_distribution_aproximation:
            psiLFunc = lambda x: self.__psiVDL__(x)[:,2]
        else:   
            assert False, "Distribution not implemented"
            
        for imic in range(nmics):
            # Microphone Position
            Y = YVec[imic]
            theta = thetaVec[imic]
            Sr = SrVec[imic]
            
            # Cache
            sinthe = np.sin(theta)
            costhe = np.cos(theta)
            cte0 = 1 - Mx*costhe
            kx_no_m = 2*B*BD*Mt/(Mr*cte0)           # missing only multipy by m
            phis_no_m = 2*B*Mt/(Mr*cte0) * MCA/D    # missing only multipy by m
            
            for m in range(1, number_of_harmonics+1):
                #Wave number and Source Transform
                kx = m*kx_no_m
                psiL = psiLFunc(kx)
                
                # Phase lag due to sweep
                phis = m*phis_no_m
                
                # Bessel function
                JmB = jv(m*B,  m*B*z*Mt*sinthe/cte0)
                
                # dPdr
                # cte1 = m*B*Mt*sinthe *np.exp(1j*m*B*(omega_c0*Sr - np.pi/2))/(4*np.pi*Y*rtip*cte0)
                # cte1 = 1j*m*B*Mt*sinthe/(4*np.pi*Y*rtip*cte0)
                # cte1 = 1j*m*B*Mt*sinthe *np.exp(1j*m*B*(omega_c0*Sr - np.pi/2))/(4*np.pi*Y*rtip*cte0)
                cte1 = m*B*Mt*sinthe*np.exp(1j*m*B*(omega_c0*Sr - np.pi/2))/(2*np.sqrt(2)*np.pi*Y*rtip*cte0)
                
                dPdr = cte1*(costhe*dTdr/cte0 - dQdr/(z**2*Mt*rtip)) * psiL*JmB*np.exp(1j*phis)
                
                Preal = np.trapezoid(np.real(dPdr), z)
                Pimag = np.trapezoid(np.imag(dPdr), z)
                PLoad[imic, m -1] = Preal + 1j*Pimag
            
        
        Prms = np.abs(PLoad) * np.sqrt(2)/2
        return Prms
    
    def hansonSteady_liftDrag(
        self,
        number_of_harmonics:int , 
        number_of_blades:int, 
        Mt:float, 
        rtip: float, 
        Mx :float,
        z:np.ndarray,
        b:np.ndarray,
        MCA:np.ndarray,
        loading:list[np.ndarray]
    ):
        # Define short variable names
        c = self.sound_speed
        rho = self.density
        D = 2*rtip       
        B = number_of_blades
        omega_c0 = Mt/rtip
        
        # Distribution as function of radius
        BD = b/D
        dLdr, dDdr = loading
        Mr = np.sqrt(Mx**2 + z**2 * Mt**2)
        
        message = 'Arrays with properties distribuied by the radius needed had the same size'
        assert len(dLdr) == len(dDdr) == len(z) == len(b) == len(MCA), message
        
        # Microphone positions
        nmics = self.microphones.shape[0]
        theta1Vec = self.microphones_to_polar[:, 1]
        thetaVec = np.arccos( np.cos(theta1Vec) * np.sqrt(1 - Mx**2 * np.sin(theta1Vec)**2) + Mx * np.sin(theta1Vec)**2)  
        
        YVec = self.microphones[:, 1] # Far-Field aproximation y >> z
        SrVec = YVec/np.sin(thetaVec)
        # SrVec = np.sqrt()
        
        # Initialize variables
        PLoad = np.zeros((nmics, number_of_harmonics), dtype=complex)
        psiLFunc = lambda x: self.__psiVDL__(x)[:,2]
            
        for imic in range(nmics):
            # Microphone Position
            Y = YVec[imic]
            theta = thetaVec[imic]
            Sr = SrVec[imic]
            
            # Cache
            sinthe = np.sin(theta)
            costhe = np.cos(theta)
            cte0 = 1 - Mx*costhe
            kx_no_m = 2*B*BD*Mt/(Mr*cte0)           # missing only multipy by m
            ky_no_m =  2*B*BD*(Mx - Mr**2 *costhe)/(z*Mr *cte0) # missing only multipy by m
            phis_no_m = 2*B*Mt/(Mr*cte0) * MCA/D    # missing only multipy by m
            
            for m in range(1, number_of_harmonics+1):
                #Wave number and Source Transform
                kx = m*kx_no_m
                ky = m*ky_no_m
                psiLD = psiLFunc(kx)
                
                # Phase lag due to sweep
                phis = m*phis_no_m
                
                # Bessel function
                JmB = jv(m*B,  m*B*z*Mt*sinthe/cte0)
                
                # dPdr
                cte1 = B*sinthe *np.exp(1j*m*B*(omega_c0*Sr - np.pi/2))/(8*np.pi*(Y/D)*cte0)
                
                dPdr = cte1*np.exp(1j*phis)*JmB*(1j*kx*dDdr + 1j*ky*dLdr)*psiLD
                
                Preal = np.trapezoid(np.real(dPdr), z)
                Pimag = np.trapezoid(np.imag(dPdr), z)
                PLoad[imic, m -1] = Preal + 1j*Pimag
            
        
        Prms = np.abs(PLoad) * np.sqrt(2)/2
        return Prms
            
            
    

    
    


    
