import numpy as np

from .base import Physics

class ShallowWater(Physics):
    def __init__(self):
        self.g = 9.81

    def flux(self, Q_array, zb, normal_idx: int = 1):
        eta = Q_array[..., 0]
        q_n = Q_array[..., normal_idx]

        h = eta - zb
        u_n = np.divide(q_n, h, where=h > 0, out=np.zeros_like(q_n))

        F_array = np.empty_like(Q_array)
        F_array[..., 0] = q_n

        for i in range(1, Q_array.shape[-1]):
            if i == normal_idx:
                F_array[..., i] = (q_n * u_n) + (0.5 * self.g * (np.power(eta, 2) - (2 * eta * zb)))
            else:
                q_i = Q_array[..., i]
                u_i = np.divide(q_i, h, where=h > 0, out=np.zeros_like(q_i))
                F_array[..., i] = q_n * u_i

        return F_array

    def source(self, Q_array, mesh, mannings_n):
        eta = Q_array[..., 0]
        zb = mesh.zb

        h = np.maximum(eta - zb, 0.0)

        S_array = np.zeros_like(Q_array)

        q_mag_sq = np.zeros_like(eta)
        for i in range(1, Q_array.shape[-1]):
            q_mag_sq += np.power(Q_array[..., i], 2)
        q_mag = np.sqrt(q_mag_sq)

        for d in mesh.directions:
            dzb_dd = np.gradient(zb, mesh.spacing(d), axis=d)
            bed_slope = -self.g * eta * dzb_dd

            i = d + 1
            q_i = Q_array[..., i]

            if mannings_n != 0:
                friction = self.g * (mannings_n**2) * q_i * q_mag / (h**(7/3))
            else:
                friction = np.zeros_like(bed_slope)

            S_array[..., i] = bed_slope - friction

        return S_array

    def max_wave_speed(self, Q_array, zb):
        eta = Q_array[..., 0]

        h = np.maximum(eta - zb, 0.0)
        a = np.sqrt(self.g * h)

        u_mag_sq = np.zeros_like(eta)
        for i in range(1, Q_array.shape[-1]):
            q_i = Q_array[..., i]
            u_i = np.divide(q_i, h, where=h > 0, out=np.zeros_like(q_i))
            u_mag_sq += np.power(u_i, 2)
        u_mag = np.sqrt(u_mag_sq)

        return np.max(u_mag + a)

    def dynamic_timestep(self, Q_array, mesh) -> float:
        max_speed = self.max_wave_speed(Q_array, mesh.zb)
        inv_spacing_sum = sum(1.0 / mesh.spacing(d) for d in mesh.directions)
        return np.nextafter(1.0 / (max_speed * inv_spacing_sum), -np.inf)