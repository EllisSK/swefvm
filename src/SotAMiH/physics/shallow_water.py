import numpy as np

from.base import Physics

class ShallowWater1D(Physics):
    def __init__(self, dx):
        self.g = 9.81
        self.dx = dx
    
    def flux(self, Q_array, zb):
        eta = Q_array[:, 0]
        q = Q_array[:, 1]

        h = eta - zb
        u = np.divide(q, h, where=h > 0, out=np.zeros_like(q))
        
        F_array = np.empty_like(Q_array)
        F_array[:, 0] = q
        F_array[:, 1] = (q * u) + (0.5 * self.g * (np.power(eta, 2) - (2*eta*zb)))
        
        return F_array
    
    def source(self, Q_array, zb, mannings_n):
        eta = Q_array[:, 0]
        q = Q_array[:, 1]
        
        h = np.maximum(eta - zb, 0.0) 
        
        S_array = np.zeros_like(Q_array)
        
        dzb_dx = np.gradient(zb, self.dx)
        bed_slope = -self.g * eta * dzb_dx
        
        if mannings_n != 0:
            friction = self.g * (mannings_n**2) * q * np.abs(q) / (h**(7/3))
        else:
            friction = np.zeros_like(bed_slope)
        
        S_array[:, 1] = bed_slope - friction
        
        return S_array
          
    def max_wave_speed(self, Q_array, zb):
        eta = Q_array[:, 0]
        q = Q_array[:, 1]

        h = np.maximum(eta - zb, 0.0)
        u = np.divide(q, h, where=h > 0, out=np.zeros_like(q))
        a = np.sqrt(self.g * h)

        return np.max(np.abs(u) + a)
    
    def dynamic_timestep(self, Q_array, zb) -> float:
        max_speed = self.max_wave_speed(Q_array, zb)
        #Shifts result down 1 bit ensuring fp errors never take Cr over 1
        return np.nextafter(self.dx / max_speed, -np.inf)