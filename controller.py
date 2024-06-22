import numpy as np
from cvxpy import *
import mosek
from scipy.linalg import solve_discrete_are
from scipy.linalg import solve_discrete_lyapunov
from scipy.linalg import block_diag
import time
import torch


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
    def __init__(self, model, args):
        self.control_horizon = args['control_horizon']
        self.pred_horizon = args['pred_horizon']
        self.a_dim = args['act_dim']
        self.LQT_gamma = 0.5  # for cost discount

        # self.end_weight = args['end_weight']
        self.args = args
        self.model = model

        self._build_matrices(args)

    def _build_matrices(self, args):

        # 通过cvxpy构建线性问题 -> 包含参数设计
        self.x_dim = self.model.A_1.shape[0]
        self.u_dim = self.model.B_1.shape[0]
        self.s_dim = self.model.C_1.shape[1]
        # self.Q = np.diag([1., .1, 10., 0.01])
        # self.R = np.diag([0.1])
        self.Q = args['Q']
        self.R = args['R']

        self.A_holder = Parameter((self.x_dim, self.x_dim,))
        self.B_holder = Parameter((self.x_dim, self.u_dim,))

        self.K_holder = Parameter((self.u_dim, self.x_dim,))
        self.P_holder = Parameter((self.x_dim, self.x_dim,))

        self.u_s_holder = Parameter((self.a_dim,))
        self.ref = Parameter(self.s_dim)

    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.shift_
        # self.reference = (self.args['reference'] - self.shift) / self.scale
        self.reference = self.args['reference']

        self.A = self.model.A_1.detach().numpy().T
        self.B = self.model.B_1.detach().numpy().T
        self.C = self.model.C_1.detach().numpy().T

        self._shift_and_scale_bounds(self.args)
        self._set_LQR_controller()

        self._create_set_point_u_prob()  #
        self._get_set_point_u(self.reference)
        self.create_prob(self.args)

    def _shift_and_scale_bounds(self, args):
        if np.sum(self.scale) > 0. and np.sum(self.scale_u) > 0.:
            if args['apply_state_constraints']:
                self.s_bound_high = (args['s_bound_high'] - self.shift) / self.scale
                self.s_bound_low = (args['s_bound_lowh'] - self.shift) / self.scale
            else:
                self.s_bound_low = None
                self.s_bound_high = None
            if args['apply_action_constraints']:
                self.a_bound_high = (args['a_bound_high'] - self.shift_u) / self.scale_u
                self.a_bound_low = (args['a_bound_low'] - self.shift_u) / self.scale_u
            else:
                self.a_bound_low = None
                self.a_bound_high = None

    def _set_LQR_controller(self):

        A_1 = np.sqrt(self.LQT_gamma) * block_diag(self.A, np.eye(self.s_dim))
        B_1 = np.sqrt(self.LQT_gamma) * np.vstack((self.B, np.zeros([self.s_dim, self.a_dim])))
        C_1 = np.hstack((self.C, -np.eye(self.s_dim)))

        self.CQC = CQC = np.dot(C_1.T, np.dot(self.Q, C_1))
        self.K = dlqr(A_1, B_1, CQC, self.R)
        self.CPC = self._get_trm_cost(A_1, B_1, CQC, self.R, self.K)

    def _create_set_point_u_prob(self):

        self.u_s_var = Variable((self.a_dim))
        phi_s = Variable((self.x_dim))
        # x_s = hstack([self.ref, phi_s])
        X_s = hstack([phi_s, self.ref])

        # constraint = [self.C @ x_s == self.C @ (self.A @ x_s + self.B @ self.K @ X_s + self.B @ self.u_s_var)]
        constraint = [self.C @ phi_s == self.C @ (self.A @ phi_s + self.B @ self.K @ X_s + self.B @ self.u_s_var)]
        constraint += [self.C @ phi_s == self.ref]
        objective = quad_form(self.u_s_var, np.eye(self.a_dim))

        self._set_point_prob = Problem(Minimize(objective), constraint)

    def _get_set_point_u(self, reference):

        # self.ref.value = reference
        self.ref.value = self.reference
        try:
            self._set_point_prob.solve(solver=MOSEK, warm_start=True,
                                       mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual})
        except cvxpy.error.SolverError:
            self._set_point_prob.solve(solver=MOSEK, warm_start=True,
                                       mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual},
                                       verbose=True)

        if self._set_point_prob.status == 'optimal':
            self.u_s_holder.value = self.u_s_var.value
        else:
            print('no suitable set point control input')
            self.u_s_holder.value = np.zeros([self.a_dim])

    def _get_trm_cost(self, A, B, Q, R, K):
        '''
        get_trm_cost returns the matrix P associated with the terminal cost
        Outputs:    P: the matrix associated with the terminal cost in the objective
        '''

        A_lyap = (A + np.dot(B, K)).T
        Q_lyap = Q + np.dot(K.T, np.dot(R, K))

        P = solve_discrete_lyapunov(A_lyap, Q_lyap)

        return P

    def create_prob(self, args):
        self.u = Variable((self.a_dim, self.control_horizon))
        self.x = Variable((self.x_dim, self.pred_horizon + 1))
        self.x_init = Parameter(self.x_dim)

        objective = 0.
        constraints = [self.x[:, 0] == self.x_init]
        for k in range(self.pred_horizon):
            #     X = vstack([reshape(self.x[:, k], (self.x_dim, 1)), reshape(self.ref, (self.s_dim,1))])[:,0]
            #     k_u = k if k <= self.control_horizon-1 else self.control_horizon-1
            #     objective += quad_form(X, self.CQC) + quad_form(self.u[:, k_u]-self.u_s_holder, self.R)
            #     # objective += quad_form(X, self.CQC) + quad_form(self.u[:, k_u]-self.u_s_holder, self.R)
            #     constraints += [self.x[:, k + 1] == self.A_holder @ self.x[:, k] + self.B_holder @self.K @ X + self.B_holder @ self.u[:, k_u]]
            #     if args['apply_state_constraints']:
            #         constraints += [self.s_bound_low <= self.x[:self.s_dim, k],
            #                         self.x[:self.s_dim, k] <= self.s_bound_high]
            #     if args['apply_action_constraints'] and k <self.control_horizon:
            #         constraints += [self.a_bound_low <= self.K @ X + self.u[:, k], self.K @ X + self.u[:, k] <= self.a_bound_high]
            # X = vstack([reshape(self.x[:, -1], (self.x_dim,1)), reshape(self.ref, (self.s_dim,1))])[:,0]
            # objective += quad_form(X, self.CPC)
            # objective += quad_form(self.x[:self.state_dim, k], self.Q) + quad_form(self.u[:, k], self.R)
            k_u = k if k <= self.control_horizon - 1 else self.control_horizon - 1
            constraints += [self.x[:, k + 1] == self.A_holder @ self.x[:, k] + self.B_holder @ self.u[:, k_u]]
            objective += quad_form(self.C @ self.x[:, k] - self.reference, self.Q) + quad_form(self.u[:, k_u], self.R)
            # if args['apply_state_constraints']:
            #     constraints += [self.s_bound_low <= self.x[:self.state_dim, k], self.x[:self.state_dim, k] <= self.s_bound_high]
            if args['apply_action_constraints']:
                constraints += [self.a_bound_low <= self.u[:, k_u], self.u[:, k_u] <= self.a_bound_high]
        # objective += 100 * quad_form(self.x[:self.state_dim, -1], self.Q)
        self.prob = Problem(Minimize(objective), constraints)

        print("done")

    def check_controllability(self):

        gamma = [self.B]
        A = self.A
        for d in range(self.x_dim - 1):
            gamma.append(np.matmul(A, gamma[d]))

        gamma = np.concatenate(np.array(gamma), axis=1)
        rank = np.linalg.matrix_rank(gamma)
        print('rank of controllability matrix is ' + str(rank) + '/' + str(self.x_dim))

    # def _restore_koopman(self):

    #     self.A_1 = torch.load(SAVE_A1).numpy()
    #     self.B_1 = torch.load(SAVE_B1).numpy()
    #     self.C_1 = torch.load(SAVE_C1).numpy()


class MPC(base_mpc):
    def __init__(self, model, args):
        super(MPC, self).__init__(model, args)

    def choose_action(self, x_0, reference, *args):
        if hasattr(self, 'last_state'):
            if len(self.last_state) < 1:
                u = np.random.uniform(self.a_bound_low, self.a_bound_high)
                self.last_state = x_0
                return u
            else:
                phi_0 = self.model.encode([[self.last_state], [x_0]])
                self.last_state = x_0
        else:
            phi_0 = self.model.encode([x_0])
        self.x_init.value = phi_0

        self._get_set_point_u(self.reference)  ####
        # self.A_holder.value = self.model.A_1.detach().numpy().T
        self.A_holder.value = self.model.A_1.detach().numpy().T
        self.B_holder.value = self.model.B_1.detach().numpy().T
        self.prob.solve(solver=MOSEK, warm_start=True,
                        mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual})
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            X = np.concatenate([phi_0, self.reference])  # self.reference[0,:]
            u = np.dot(self.K, X) + self.u[:, 0].value
            u = u * self.scale_u + self.shift_u
        else:
            print("Error: Cannot solve mpc..")
            X = np.concatenate([phi_0, self.reference])  # self.reference[0,:]
            u = np.dot(self.K, X)
            u = u * self.scale_u + self.shift_u

        # u = np.zeros_like(u)

        return u

    def reset(self):
        if hasattr(self, 'last_state'):
            self.last_state = []

    def restore(self):
        success = self.model.parameter_restore(self.args)
        self._build_controller()
        return success

    def simple_choose_action(self, phi_0, *args):

        self.x_init.value = phi_0
        self.prob.solve(solver=OSQP, warm_start=True)
        self.prob.solve()
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            u = self.u[:, 0].value
        else:
            print("Error: Cannot solve mpc..")
            u = None

        return u


# 不如直接重写 不然很麻烦
class Upper_MPC(object):
    def __init__(self, model, shift, args):
        # super(MPC, self).__init__(model, args)
        self.shift_dim = args['mix_x_u']
        self.control_horizon = args['control_horizon']
        self.pred_horizon = args['pred_horizon']
        self.a_dim = args['act_dim']
        self.LQT_gamma = 0.5  # for cost discount
        # self.end_weight = args['end_weight']
        self.args = args
        self.model = model
        self.shift = np.array(shift[0].cpu())
        self.scale = np.array(shift[1].cpu())
        self.shift_u = np.array(shift[2].cpu())
        self.scale_u = np.array(shift[3].cpu())
        self._build_matrices(args)
        self.init = False
        self.next = False
        self.stable_u = np.array([2.92851107e+09, 9.66706684e+08, 2.89820385e+09])

    def _create_set_point_u_prob(self):

        phi_s_value = self.model.encode([self.reference_])
        self.u_s_var = Variable((self.a_dim))
        # self.phi_s_ = Parameter((self.x_dim))
        # phi_s_ = Variable((self.x_dim))
        self.mix_u = Parameter((self.shift_dim))
        # x_s = hstack([self.ref, phi_s])
        U_s = hstack([self.mix_u, self.u_s_var])
        # constraint = [self.C @ phi_s_ == self.C @ (self.A @ phi_s_ + self.B @ U_s)]
        # constraint = [phi_s_value ==  self.A @phi_s_value+ self.B @ U_s]
        # constraint += [self.C @ self.phi_s_ == phi_s_value]

        objective = quad_form(phi_s_value - self.A @ phi_s_value + self.B @ U_s, np.eye(phi_s_value.shape[0]))
        objective += quad_form(self.u_s_var - self.stable_u[0], np.eye(self.u_s_var.shape[0]))
        self._set_point_prob = Problem(Minimize(objective))

    def _get_set_point_u(self):

        # self.ref.value = reference
        self.ref.value = self.reference
        # self.mix_u.value = self.model.init_u_light(self.reference.reshape([-1,1]),self.stable_u.reshape([-1,1]))
        # iter = 5
        # for i in range(iter):
        #     # print(self.mix_u.value)
        #     self._set_point_prob.solve(solver=MOSEK, warm_start=True,
        #                 mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual})
        #     print(self._set_point_prob.status)
        #     self.mix_u.value = self.model.init_u_light(self.reference,self.u_s_var.value)
        #     print(self.u_s_var.value)
        # self.u_s_holder.value = self.u_s_var.value.reshape([-1])

        # try:
        #     self._set_point_prob.solve(solver=MOSEK, warm_start=True,
        #                     mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual})
        # except cvxpy.error.SolverError:
        #     self._set_point_prob.solve(solver=MOSEK, warm_start=True,
        #                     mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}, verbose=True)

        # if self._set_point_prob.status == 'optimal':
        #     self.u_s_holder.value = self.u_s_var.value
        # else:
        #     print('no suitable set point control input')
        #     self.u_s_holder.value = np.zeros([self.a_dim])

    def _shift_and_scale_bounds(self, args):
        if np.sum(self.scale) > 0. and np.sum(self.scale_u) > 0.:
            if args['apply_state_constraints']:
                self.s_bound_high = (args['s_bound_high'] - self.shift) / self.scale
                self.s_bound_low = (args['s_bound_lowh'] - self.shift) / self.scale
            else:
                self.s_bound_low = None
                self.s_bound_high = None
            if args['apply_action_constraints']:
                self.a_bound_high = (args['a_bound_high'] - self.shift_u) / self.scale_u
                self.a_bound_low = (args['a_bound_low'] - self.shift_u) / self.scale_u
                self.stable_u = (self.stable_u - self.shift_u) / self.scale_u
                print("wwwwwwww------------------------------------------------------------------------")
            else:
                self.a_bound_low = None
                self.a_bound_high = None

    def _build_matrices(self, args):

        # 通过cvxpy构建线性问题 -> 包含参数设计
        self.x_dim = self.model.A_1.shape[0]
        self.u_dim = self.model.B_1.shape[0]
        self.s_dim = self.model.C_1.shape[1]
        # self.Q = np.diag([1., .1, 10., 0.01])
        # self.R = np.diag([0.1])
        self.Q = np.diag([1.5, 0.5, 0.5, 1.5, 0.5, 0.5, 1.5, 0.5, 0.5])
        self.R = np.diag(0.05 * np.ones([3]))
        # self.Q = args['Q']
        # self.R = args['R']

        self.A_holder = Parameter((self.x_dim, self.x_dim,))
        self.B_holder = Parameter((self.x_dim, self.u_dim,))

        self.K_holder = Parameter((self.u_dim, self.x_dim,))
        self.P_holder = Parameter((self.x_dim, self.x_dim,))

        self.u_s_holder = Parameter((self.a_dim))
        self.ref = Parameter(self.s_dim)

    def _build_controller(self):

        # [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.shift_
        # self.reference = (self.args['reference'] - self.shift) / self.scale
        self.reference = self.args['reference']
        self.reference_ = self.args['reference_']

        self.A = self.model.A_1.detach().numpy().T
        self.B = self.model.B_1.detach().numpy().T
        self.C = self.model.C_1.detach().numpy().T

        self._shift_and_scale_bounds(self.args)
        self._create_set_point_u_prob()
        self._get_set_point_u()  ####

        self.create_prob(self.args)

    def create_prob(self, args):
        self.u = Variable((self.a_dim, self.control_horizon))
        self.x = Variable((self.x_dim, self.pred_horizon + 1))

        # initial u
        self.u_mix = Parameter((self.shift_dim, self.pred_horizon + 1))
        self.x_init = Parameter(self.x_dim)
        # the same size

        objective = 0.
        constraints = [self.x[:, 0] == self.x_init]

        # constraints += [self.u[:, 0] == self.u_init]
        # 有无软约束 约束的优先级？

        for k in range(self.pred_horizon):
            k_u = k if k <= self.control_horizon - 1 else self.control_horizon - 1
            mix = hstack([self.u_mix[:, k], self.u[:, k_u]])
            constraints += [self.x[:, k + 1] == self.A_holder @ self.x[:, k] + self.B_holder @ mix]
            objective += quad_form(self.C @ self.x[:, k] - self.reference, self.Q) + quad_form(
                self.u[:, k_u] - self.stable_u, self.R)
            # + quad_form(self.u[:, k_u], self.R)
            # if args['apply_state_constraints']:
            #     constraints += [self.s_bound_low <= self.x[:self.state_dim, k], self.x[:self.state_dim, k] <= self.s_bound_high]
            if args['apply_action_constraints']:
                constraints += [self.a_bound_low <= self.u[:, k_u], self.u[:, k_u] <= self.a_bound_high]
        # objective += 100 * quad_form(self.x[:self.state_dim, -1], self.Q)
        self.prob = Problem(Minimize(objective), constraints)

    def vector_matrix(vector):
        return vector.reshape([vector.shape[0], -1])

    def choose_action(self, x_0, reference, *args):
        if hasattr(self, 'last_state'):
            if len(self.last_state) < 1:
                u = np.random.uniform(self.a_bound_low, self.a_bound_high)
                self.last_state = x_0
                return u
            else:
                phi_0 = self.model.encode([[self.last_state], [x_0]])
                self.last_state = x_0
        else:
            phi_0 = self.model.encode([x_0])

        # self.A_holder.value = self.model.A_1.detach().numpy().T
        self.A_holder.value = self.model.A_1.detach().numpy().T
        self.B_holder.value = self.model.B_1.detach().numpy().T

        iter = 3  # 循环次数
        self.next = True
        ##------------------------------------method 1----------------------------------------------##
        for i in range(iter):
            if not self.init:
                x_aug = np.repeat([x_0], self.pred_horizon + 1, axis=0).T
                u_aug = np.zeros([self.a_dim, self.pred_horizon + 1])
                self.init = True
                self.u_mix.value = self.model.init_u(x_aug, u_aug, True)

            else:
                if self.next:
                    x_fh = self.x.value[:, 1:]
                    ##-----------这里必须引入phi_0-----------##
                    # print(phi_0-x_fh[:,0])
                    x_fh[:, 0] = phi_0
                    ##--------------------------------------##
                    x_nh = self.x.value[:, -1]
                    u_fh = self.u.value[:, 1:]
                    u_nh = np.repeat(Upper_MPC.vector_matrix(self.u.value[:, -1]),
                                     self.pred_horizon - self.control_horizon + 2, 1)
                    u_aug = np.concatenate((u_fh, u_nh), axis=1)
                    x_aug = np.concatenate((x_fh, Upper_MPC.vector_matrix(x_nh)), axis=1)
                    self.next = False
                else:
                    x_aug = self.x.value
                    u_fh = self.u.value
                    u_nh = np.repeat(Upper_MPC.vector_matrix(self.u.value[:, -1]),
                                     self.pred_horizon - self.control_horizon + 1, 1)
                    u_aug = np.concatenate((u_fh, u_nh), axis=1)

                # print(x_aug.shape)
                # print(u_aug.shape)
                self.u_mix.value = self.model.init_u(self.C.dot(x_aug), u_aug, False)

            ##------------------------------------method 2----------------------------------------------##

            # self.init = False

            # for i in range(iter):
            #     if not self.init:
            #         x_aug = np.repeat([x_0],self.pred_horizon+1,axis=0).T
            #         u_aug = np.zeros([self.a_dim, self.pred_horizon+1])
            #         self.init = True
            #         self.u_mix.value  = self.model.init_u(x_aug,u_aug,True)

            #     else:
            #         # if self.next:
            #         #     x_fh  = self.x.value[:,1:]
            #         #     ##-----------这里必须引入phi_0-----------##
            #         #     # print(phi_0-x_fh[:,0])
            #         #     x_fh[:,0] = phi_0
            #         #     ##--------------------------------------##
            #         #     x_nh  = self.x.value[:,-1]
            #         #     u_fh  = self.u.value[:,1:]
            #         #     u_nh  = np.repeat(Upper_MPC.vector_matrix(self.u.value[:,-1]),self.pred_horizon-self.control_horizon+2,1)
            #         #     u_aug = np.concatenate((u_fh, u_nh), axis=1)
            #         #     x_aug = np.concatenate((x_fh, Upper_MPC.vector_matrix(x_nh)), axis=1)
            #         #     self.next = False
            #         # else:
            #         x_aug = self.x.value
            #         u_fh  = self.u.value
            #         u_nh  = np.repeat(Upper_MPC.vector_matrix(self.u.value[:,-1]),self.pred_horizon-self.control_horizon+1,1)
            #         u_aug = np.concatenate((u_fh, u_nh), axis=1)

            #         # print(x_aug.shape)
            #         # print(u_aug.shape)
            #         self.u_mix.value  = self.model.init_u(self.C.dot(x_aug),u_aug,False)

            ##这样初始化是否合理？
            self.x_init.value = phi_0
            self.prob.solve(solver=MOSEK, warm_start=True,
                            mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual})
            # print(self.u[:, 0].value)
            # print(i)
            # print(self.prob.status)

        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            # X = np.concatenate([phi_0, self.reference])      #self.reference[0,:]
            # u = np.dot(self.K, X) + self.u[:, 0].value
            u = self.u[:, 0].value
            u = u * self.scale_u + self.shift_u
        else:
            print("Error: Cannot solve mpc..")
            X = np.concatenate([phi_0, self.reference])  # self.reference[0,:]
            u = np.dot(self.K, X)
            u = u * self.scale_u + self.shift_u

        # u = np.zeros_like(u)

        return u

    def reset(self):
        if hasattr(self, 'last_state'):
            self.last_state = []

    def restore(self):
        success = self.model.parameter_restore(self.args)
        self._build_controller()
        return success

    def simple_choose_action(self, phi_0, *args):

        self.x_init.value = phi_0
        self.prob.solve(solver=OSQP, warm_start=True)
        self.prob.solve()
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            u = self.u[:, 0].value
        else:
            print("Error: Cannot solve mpc..")
            u = None

        return u


if __name__ == '__main__':
    pass