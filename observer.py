import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
from three_tanks import three_tank_system as dreamer
from replay_memory import ReplayMemory
import my_args



def main():
    args = my_args.args
    args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device is : ", args['device'])
    env = dreamer(args)
    env = env.unwrapped
    args['state_dim'] = env.observation_space.shape[0]
    args['act_dim'] = env.action_space.shape[0]
    args['act_expand'] = args['act_expand'] * env.action_space.shape[0]
    args['reference'] = env.xs
    estimate(env, args)


class base_MHE(object):
    def __init__(self, model, args):
        self.pred_horizon = args['pred_horizon']
        self.state_dim = args['state_dim']
        self.latent_dim = args['latent_dim']
        self.A = model.A_1.detach().numpy()
        self.B = model.B_1.detach().numpy()
        self.C = model.C_1.detach().numpy()
        self.net = model.net
        self._build_matrices()

    def _build_matrices(self):
        self.Q = np.diag(0.08 * np.ones([self.latent_dim+self.state_dim]))
        self.R = np.diag(0.07 * np.ones([self.state_dim]))
        # self.R = np.dot(np.dot(self.C.T, self.Q), self.C)
        self.P = np.diag(0.05 * np.ones([self.latent_dim+self.state_dim]))
        self.scale_1 = 1.
        self.scale_2 = 1.
        self.scale_3 = 1.

        # H = self.pred_horizon
        # U, S, Vt = np.linalg.svd(self.A)
        # max_singular_value = np.max(S)
        # self.scale_1 = max_singular_value ** (2 * H)
        # self.scale_2 = 1/H
        # self.scale_3 = 1.
        #
        # print("Q:",self.Q)
        # print("R:", self.R)
        # print("scale_1 :", self.scale_1)
        # print("scale_2 :", self.scale_2)


    def MHE(self, x, u, g_init, args):
        self.g_pred = cp.Variable((self.pred_horizon, self.latent_dim + self.state_dim))
        self.x_pred = cp.Variable((self.pred_horizon, self.state_dim))
        self.max_l = cp.Variable()
        self.w = cp.Variable((self.pred_horizon, self.latent_dim + self.state_dim))
        self.x_true = x
        self.g_init = g_init
        self.u = u

        objective = 0.
        constraints = [self.x_pred[0, :] == self.g_pred[0, :] @ self.C]
        objective += self.scale_1 * cp.quad_form(self.g_pred[0, :] - self.g_init, self.P)

        for i in range(self.pred_horizon-1):
            constraints += [self.g_pred[i + 1, :] == self.g_pred[i, :] @ self.A + self.u[i, :] @ self.B + self.w[i, :]]
            constraints += [self.x_pred[i, :] == self.g_pred[i, :] @ self.C]
            l = cp.quad_form(self.x_true[i, :] - self.g_pred[i, :] @ self.C, self.R) + cp.quad_form(self.w[i, :], self.Q)
            constraints += [l <= self.max_l]
            objective += self.scale_2 * l
        objective += self.scale_3 * self.max_l

        self.prob = cp.Problem(cp.Minimize(objective), constraints)

        try:
            self.prob.solve()

        except cp.error.SolverError as e:
            print("Solver error:", e)
            pass

        optimal_g = self.g_pred[0, :].value
        g_pre = self.g_pred[1, :].value
        x_est = self.x_pred[0, :].value

        return optimal_g, g_pre, x_est


def estimate(env,args):
    args['re_generate_test'] = False
    # plot list
    x_true_list = []
    x_est_list = []

    # introduce model
    model = restore_data(args)
    MHE_method = base_MHE(model, args)
    # true data  ---- re-generate
    if args['re_generate_test']:
        replay_memory = ReplayMemory(args, env, predict_evolution=True)
        x_test = torch.tensor(replay_memory.x_test[:, :520, :]).squeeze().float()  # [1, n, 9]
        u_test = torch.tensor(replay_memory.u_test[:, :520, :]).squeeze().float()
        x_test, u_test = shift_scale(x_test, u_test, replay_memory.shift_)
    else:
        x_test = torch.load(args['SAVE_TEST_X'])
        u_test = torch.load(args['SAVE_TEST_U'])
        shift = torch.load(args['SAVE_SHIFT'])
        x_test = torch.tensor(x_test[:, :520, :]).squeeze().float()
        u_test = torch.tensor(u_test[:, :520, :]).squeeze().float()
        x_test, u_test = shift_scale(x_test, u_test, shift)

    x0_buff = x_test[0, :]
    g0 = model.net(x0_buff)
    g0 = torch.cat([x0_buff, g0])
    g0 = g0.detach().numpy()

    x_test = x_test.detach().numpy()
    u_test = u_test.detach().numpy()

    # 数据的输入是不是按照每一步一预测算的？？
    for i in range(x_test.shape[0] - args['pred_horizon']):
        x = x_test[i:i + args['pred_horizon'], :]
        u = u_test[i:i + args['pred_horizon'] - 1, :]
        g_opt, g0, x_est = MHE_method.MHE(x, u, g0, args)
        x_true_list.append(x[0, :])
        x_est_list.append(x_est)
        print("step:{}; x_true:{},x_pred{}".format(i, x[0, :], x_est))

    error = mse_measure(x_true_list, x_est_list)
    print("prediction error is:", error)

    #####-------------plot----------------###
    if args['re_generate_test']:
        shift = replay_memory.shift_
    # else:
    #     shift = shift
    x_true_list = [x * shift[1].numpy() + shift[0].numpy() for x in x_true_list]
    x_est_list = [x * shift[1].numpy() + shift[0].numpy() for x in x_est_list]
    # x_true_list = x_true_list * shift[1] + shift[0]
    # x_est_list = x_est_list * shift[1] + shift[0]
    x_true_array = np.array(x_true_list)
    x_est_array = np.array(x_est_list)

    plt.close()
    f, axs = plt.subplots(args['state_dim'], sharex=True, figsize=(15, 15))
    for i in range(args['state_dim']):
        axs[i].plot(x_true_array[:, i], label='x_true', color='k')
        axs[i].plot(x_est_array[:, i], label='x_pred', color='r')
        # axs[i].legend()
    axs[-1].set_xlabel('Steps')
    plt.legend()
    plt.savefig('data/MHE' + '.png')
    plt.show()
    print("plot")


def shift_scale(x, u, shift_ = None):
    x_choice = (x - shift_[0])/shift_[1]
    u_choice = (u - shift_[2])/shift_[3]
    return x_choice, u_choice

def shift_rescale(x, u, shift_ = None):
    x_re = x * shift_[1].numpy() + shift_[0].numpy()
    u_re = u * shift_[3].numpy() + shift_[2].numpy()
    return x_re, u_re


def restore_data(args):
    args['if_mix'] = False
    if not args['if_mix']:
        from Desko import Koopman_Desko
        model = Koopman_Desko(args)
    else:
        from mix_u import Koopman_Desko
        model = Koopman_Desko(args)
    # restore variables
    model.parameter_restore(args)

    return model


def mse_measure(list_true, list_pred):
    if len(list_true) != len(list_pred):
        raise ValueError("Lists must have the same length.")

    mse_list = []
    for x, y in zip(list_true, list_pred):
        mse = np.mean((x - y) ** 2)  # 计算九维向量的均方误差
        mse_list.append(mse)
    mse = np.mean(mse_list)
    return mse



if __name__ == '__main__':
    main()
