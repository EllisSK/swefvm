import numpy as np

from .base import RiemannSolver

class HLLSolver(RiemannSolver):
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
        
        S_L = np.minimum(u_L - a_L, u_R - a_R)
        S_R = np.maximum(u_L + a_L, u_R + a_R)
        
        dry_L = h_L == 0.0
        dry_R = h_R == 0.0
        
        S_L[dry_L] = u_R[dry_L] - 2 * a_R[dry_L]
        S_R[dry_R] = u_L[dry_R] - 2 * a_L[dry_R]
        
        F_L = physics.flux(Q_L, zb)
        F_R = physics.flux(Q_R, zb)
        
        F_int = np.zeros_like(Q_L)
        
        cond_L = S_L >= 0
        cond_R = S_R <= 0
        cond_star = ~(cond_L | cond_R)
        
        F_int[cond_L] = F_L[cond_L]
        F_int[cond_R] = F_R[cond_R]
        
        SL_star = S_L[cond_star, np.newaxis]
        SR_star = S_R[cond_star, np.newaxis]
        
        F_int[cond_star] = (
            (SR_star * F_L[cond_star]) - (SL_star * F_R[cond_star]) + 
            (SL_star * SR_star * (Q_R[cond_star] - Q_L[cond_star]))
        ) / (SR_star - SL_star)
        
        return F_int