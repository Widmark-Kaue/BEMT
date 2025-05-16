import numpy as np
from dataclasses import dataclass, field


@dataclass
class Rotor:
    name: str
    number_of_blades: int
    radius: float
    thickness: np.ndarray
    chord: np.ndarray
    twist: np.ndarray
    sweep: np.ndarray
    airfoils_name: list 
    omega: float
    V0:float
    rho:float
    sound_of_speed:float = field(default=340)
    airfoil_distribution: list = field(default_factory=list)
    thrust:np.ndarray = field(init=False, repr=False, default_factory=lambda: np.empty(1))
    torque:np.ndarray = field(init=False, repr=False, default_factory=lambda: np.empty(1))
    CL:np.ndarray = field(init=False, repr=False, default_factory=lambda: np.empty(1))
    CD:np.ndarray = field(init=False, repr=False, default_factory=lambda: np.empty(1))
    
    def __post_init__(self):
        if self.airfoils_distribution == []:
            # set equal distribution of airfoils 
            self.airfoils_distribution = [1/len(self.airfoils_name) for _ in range(len(self.airfoils_name))]
    
    def loading_simulation_data(self):
        """
        Carrega os dados de carregamento do rotor"
        """
        pass

    @property
    def mach_tip(self):
        return self.omega*self.radius/self.sound_of_speed
    @property
    def mach_flight(self):
        return self.V0/self.sound_of_speed
    
    @property
    def diameter(self):
        return 2*self.radius