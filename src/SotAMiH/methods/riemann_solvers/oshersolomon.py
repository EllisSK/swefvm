import numpy as np

from .base import RiemannSolver

class OsherSolomonSolver(RiemannSolver):
    def __init__(self) -> None:
        pass

    def solve(self, Q_L, Q_R, physics, zb):
            g = physics.g
            
            eta_L = Q_L[:, 0]
            q_L = Q_L[:, 1]
            eta_R = Q_R[:, 0]
            q_R = Q_R[:, 1]

            h_L = np.maximum(eta_L - zb, 0.0)
            h_R = np.maximum(eta_R - zb, 0.0)
            
            u_L = np.divide(q_L, h_L, out=np.zeros_like(q_L), where=h_L > 0.0)
            u_R = np.divide(q_R, h_R, out=np.zeros_like(q_R), where=h_R > 0.0)
            
            a_L = np.sqrt(g * h_L)
            a_R = np.sqrt(g * h_R)

            #Left sonic points
            u_S0 = (u_L + (2.0 * a_L)) / 3.0
            a_S0 = u_S0
            h_S0 = np.power(a_S0, 2) / g
            q_S0 = h_S0 * u_S0
            
            #Right sonic points
            u_S1 = (u_R - (2.0 * a_R)) / 3.0
            a_S1 = -u_S1
            h_S1 = np.power(a_S1, 2) / g
            q_S1 = h_S1 * u_S1
            
            #Intermediate values
            u_star = 0.5 * (u_L + u_R) + a_L - a_R
            a_13 = a_L + (0.5 * (u_L - u_star))
            h_13 = np.power(a_13, 2) / g
            q_13 = h_13 * u_star
            
            a_23 = a_R - (0.5 * (u_R - u_star))
            h_23 = np.power(a_23, 2) / g
            q_23 = h_23 * u_star
            
            #Combine eta and q vectors into intermediate state (Q) vectors
            Q_S0 = np.stack((h_S0 + zb, q_S0), axis=1)
            Q_S1 = np.stack((h_S1 + zb, q_S1), axis=1)
            Q_13 = np.stack((h_13 + zb, q_13), axis=1)
            Q_23 = np.stack((h_23 + zb, q_23), axis=1)
            
            #Calculate fluxes
            F_0 = physics.flux(Q_L, zb)
            F_1 = physics.flux(Q_R, zb)
            F_S0 = physics.flux(Q_S0, zb)
            F_S1 = physics.flux(Q_S1, zb)
            F_13 = physics.flux(Q_13, zb)
            F_23 = physics.flux(Q_23, zb)
            
            #Determine conditions for entire vectors to determine the flux at each point
            cond_c1 = u_L - a_L >= 0.0
            cond_c2 = u_L - a_L < 0.0
            
            cond_c3 = u_R + a_R >= 0.0
            cond_c4 = u_R + a_R < 0.0
            
            col1 = cond_c1 & cond_c3
            col2 = cond_c1 & cond_c4
            col3 = cond_c2 & cond_c3
            col4 = cond_c2 & cond_c4
            
            cond_r1 = (u_star >= 0.0) & (u_star - a_13 >= 0.0)
            cond_r2 = (u_star >= 0.0) & (u_star - a_13 < 0.0)
            cond_r3 = (u_star <= 0.0) & (u_star + a_23 >= 0.0)
            cond_r4 = (u_star <= 0.0) & (u_star + a_23 < 0.0)
            
            F_int = np.zeros_like(Q_L)
            
            #Apply the conditions for the 16 possibilities
            F_int[cond_r1 & col1] = F_0[cond_r1 & col1]
            F_int[cond_r1 & col2] = F_0[cond_r1 & col2] + F_1[cond_r1 & col2] - F_S1[cond_r1 & col2]
            F_int[cond_r1 & col3] = F_S0[cond_r1 & col3]
            F_int[cond_r1 & col4] = F_S0[cond_r1 & col4] - F_S1[cond_r1 & col4] + F_1[cond_r1 & col4]
            
            F_int[cond_r2 & col1] = F_0[cond_r2 & col1] - F_S0[cond_r2 & col1] + F_13[cond_r2 & col1]
            F_int[cond_r2 & col2] = F_0[cond_r2 & col2] - F_S0[cond_r2 & col2] + F_13[cond_r2 & col2] - F_S1[cond_r2 & col2] + F_1[cond_r2 & col2]
            F_int[cond_r2 & col3] = F_13[cond_r2 & col3]
            F_int[cond_r2 & col4] = F_13[cond_r2 & col4] - F_S1[cond_r2 & col4] + F_1[cond_r2 & col4]
            
            F_int[cond_r3 & col1] = F_0[cond_r3 & col1] - F_S0[cond_r3 & col1] + F_23[cond_r3 & col1]
            F_int[cond_r3 & col2] = F_0[cond_r3 & col2] - F_S0[cond_r3 & col2] + F_23[cond_r3 & col2] - F_S1[cond_r3 & col2] + F_1[cond_r3 & col2]
            F_int[cond_r3 & col3] = F_23[cond_r3 & col3]
            F_int[cond_r3 & col4] = F_23[cond_r3 & col4] - F_S1[cond_r3 & col4] + F_1[cond_r3 & col4]
            
            F_int[cond_r4 & col1] = F_0[cond_r4 & col1] - F_S0[cond_r4 & col1] + F_S1[cond_r4 & col1]
            F_int[cond_r4 & col2] = F_0[cond_r4 & col2] - F_S0[cond_r4 & col2] + F_1[cond_r4 & col2]
            F_int[cond_r4 & col3] = F_S1[cond_r4 & col3]
            F_int[cond_r4 & col4] = F_1[cond_r4 & col4]
            
            return F_int