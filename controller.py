import numpy as np
from cvxpy import *
# import mosek
from scipy.linalg import solve_discrete_are
from scipy.linalg import solve_discrete_lyapunov
from scipy.linalg import block_diag
import time
import torch


SAVE_A1 = r"C:\Users\90721\Desktop\Desko_lzy\Desko_pytorch\save_model\A1.pt"
SAVE_B1 = r"C:\Users\90721\Desktop\Desko_lzy\Desko_pytorch\save_model\B1.pt"
SAVE_C1 = r"C:\Users\90721\Desktop\Desko_lzy\Desko_pytorch\save_model\C1.pt"


def dlqr(A, B, Q, R):
    '''
    dlqr solves the discrete time LQR controller
    Inputs:     A, B: system dynamics
    Outputs:    K: optimal feedback matrix
    '''
    P = solve_discrete_are(A, B, Q, R)
    K = - np.dot(np.linalg.inv(R + np.dot(B.T, np.dot(P, B))), np.dot(B.T, np.dot(P, A)))

    return K


class base_mpc(object):
    
    def __init__(self,model,args):
        self.control_horizon = args['control_horizon']
        self.pred_horizon = args['MPC_pred_horizon']
        self.a_dim = args['act_dim']
        self.LQT_gamma = 0.5  # for cost discount

        self.end_weight = args['end_weight']
        self.args = args
        self.model = model

        self._build_matrices(args)
    
    def _build_matrices(self, args):

        # 通过cvxpy构建线性问题 -> 包含参数设计
        self.latent_dim = args['latent_dim'] + args['state_dim']
        self.state_dim = args['state_dim']
        self.Q = self.args['Q']
        self.R = self.args['R'] 

        self.A_holder = Parameter((self.latent_dim, self.latent_dim,))
        self.B_holder = Parameter((self.latent_dim, self.args['act_dim'],))
        self.K_holder = Parameter((self.args['act_dim'], self.latent_dim,))
        self.P_holder = Parameter((self.latent_dim, self.latent_dim,))

        self.u_s_holder = Parameter((self.args['act_dim'],))
        self.ref = Parameter(self.state_dim)

    def _build_controller(self,env):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.shift_
        self.reference = (env.xs - self.shift) / self.scale

        self._shift_and_scale_bounds(self.args)
        self._set_LQR_controller()

        self._create_set_point_u_prob()
        self._create_prob(self.args)
    
    def _restore_koopman(self):
        
        self.A_1 = torch.load(SAVE_A1).numpy()
        self.B_1 = torch.load(SAVE_B1).numpy()
        self.C_1 = torch.load(SAVE_C1).numpy()






if __name__ == '__main__':
    pass