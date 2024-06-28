import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
from three_tanks import three_tank_system as dreamer
# from replay_memory import ReplayMemory
import my_args
from train import create_directories

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

    directory_list = ["estimation_result/physics", "estimation_result/noise"]
    create_directories(directory_list)

    estimate(env,args)


class MHE:
    """Moving horizon estimator
    x_k+1 = A x_k + B u_k + w_k,
    y_k = Cx_k + v_k,

    w is process disturbance,
    v is measurement noise
    """
    def __init__(self, A, B, g0, horizon, std, shift,  model, args):
        self.horizon = horizon
        self.latent_dim = args['latent_dim']
        self.state_dim = args['state_dim']
        self.if_sigma = args['if_sigma']
        self.std = std
        self.model = model
        self.select = [2, 5, 8]

        self.A = A
        self.B = B
        self.C_full = np.row_stack([np.eye(self.state_dim),
                                    np.zeros([self.latent_dim, self.state_dim])])
        self.C = self.C_full[:, self.select]

        # constraints
        self.state_scale = 10*np.ones(self.state_dim)
        self.bw = 5*np.ones(self.latent_dim+self.state_dim)

        # history of measurements and applied controls
        self.y_history = []
        self.u_history = []

        # previous horizon-1 state estimate
        self.g0 = g0.copy()

        self._build_matrices()


    def _build_matrices(self):
        std_mean = np.mean(self.std)
        extended_std = np.full((self.state_dim + self.latent_dim,), std_mean)
        extended_std[:9] = self.std
        # set Q R to be I
        # extended_std = np.ones(self.latent_dim+self.state_dim)
        extended_std = np.diag(extended_std)
        self.Q = np.linalg.inv(extended_std)
        # self.Q = np.eye(self.state_dim + self.latent_dim)
        self.R = np.transpose(self.C) @ self.Q @ self.C
        # self.P = np.eye(self.latent_dim+self.state_dim)

    def Weight_matrix_calculate(self, model, x):
        SCALE_DIAG_MIN_MAX = (-20, 2)
        log_sigma = model.noisemlp(x).detach()
        log_sigma = torch.clamp(log_sigma, min=SCALE_DIAG_MIN_MAX[0], max=SCALE_DIAG_MIN_MAX[1])
        self.sigma = torch.exp(log_sigma)
        self.sigma = torch.pow(self.sigma, 2)
        self.sigma_inv = 1.0/self.sigma
        self.sigma_inv = scale_sigma(self.sigma_inv)
        sigma_diag = self.sigma_inv.numpy()
        self.Q = np.diag(sigma_diag)
        self.R = np.transpose(self.C) @ self.Q @ self.C

    def update(self, u):
        self.u_history.append(u)
        if len(self.u_history) > self.horizon:
            # remove the first element
            self.u_history.pop(0)

    def observe(self, y):
        """ Update history """
        self.y_history.append(y)
        if len(self.y_history) > self.horizon:
            self.y_history.pop(0)
        return len(self.y_history)

    # # initial g
    def __call__(self, y, args):

        """State estimation using new observation y"""
        # N = 1,2,3,...,H,H,...H
        N = self.observe(y)
        g = cp.Variable((N, self.latent_dim + self.state_dim))
        w = cp.Variable((N, self.latent_dim + self.state_dim))
        x = cp.Variable((N, self.state_dim))
        self.max_l = cp.Variable()
        self.y_array = np.array(self.y_history)
        l_error = 0.
        cons = []

        if N == 1:
            print("initial guess")
            return self.g0 @ self.C_full
        else:
            if self.if_sigma and N == self.horizon:
                self.Weight_matrix_calculate(self.model, torch.tensor(self.g0, dtype=torch.float))
            for i in range(N):
                if i < N-1:
                    cons.append(
                        g[i + 1, :] == g[i, :] @ self.A + self.u_history[i] @ self.B + w[i, :]
                        )
                cons.append(x[i, :] == g[i, :] @ self.C_full)
                # cons.append(x[i, :] <= self.state_scale)
                # cons.append(x[i, :] >= -self.state_scale)
                # l_error += cp.quad_form(w[i, :],self.Q)
                l = cp.quad_form(self.y_array[i, self.select] - x[i, self.select], self.R) + cp.quad_form(w[i, :], self.Q)
                # cons.append(w[i, :] <= self.bw)
                # cons.append(w[i, :] >= -self.bw)
                l_error += l
                cons.append(l <= self.max_l)
            obj = cp.quad_form(g[0, :] - self.g0, self.Q) + l_error + self.max_l
            # solve mhe problem
            prob = cp.Problem(cp.Minimize(obj), cons)
            prob.solve(cp.MOSEK)
                # update initial state estimate
                # self.g0 = g.value[1, :] if N == self.u_history else g.value[0, :]
            if N == self.horizon:
                self.g0 = g.value[0, :] @ self.A + self.u_history[0] @ self.B
            else:
                self.g0 = g.value[0, :]
            return x.value[-1, :]


def estimate(env, args, test_pi = False, test_sigma = True):
    address_pi = 'save_model/pi/'
    address_nopi = 'save_model/nopi/'
    address_noise = 'save_model/'

    """Generate MHE estimation"""
    H = args['mhe_horizon']
    args['re_generate_test'] = True
    # true data  ----  re-generate
    if args['re_generate_test']:
        regenerate = ReGenerate(args, env)
        x_test = torch.tensor(regenerate.x_test[:, :, :]).squeeze().float()  # [1, n, 9]
        u_test = torch.tensor(regenerate.u_test[:, :, :]).squeeze().float()
        x_test, u_test = shift_scale(x_test, u_test, regenerate.shift_)
        shift = regenerate.shift_

    # for inital guess calculation
    x0_buff = x_test[0, :].cpu()
    # x_std = torch.std(x_test, axis=0)

    x_test = x_test.cpu().detach().numpy()
    u_test = u_test.cpu().detach().numpy()

    # x_std
    x_std = np.var(x_test, axis=0)
    # x_std = x_std/np.max(x_std)
    print("std:", x_std)

    if test_pi:
        print("----------------with pi-----------------")
        args['if_sigma'] = False
        # introduce model
        args['MODEL_SAVE'] = address_pi +'model_v1.pt'
        # pre-trained noise generate model
        args['NOISE_SAVE'] = address_pi +'noise.pt'
        args['SAVE_A1'] = address_pi +'A1.pt'
        args['SAVE_B1'] =  address_pi +'B1.pt'
        args['SAVE_C1'] =  address_pi +'C1.pt'
        # args['if_sigma'] = True
        model_pi = restore_data(args)
        A_pi = model_pi.A_1.detach().numpy()
        B_pi = model_pi.B_1.detach().numpy()
        g0 = model_pi.net(x0_buff)
        g0 = torch.cat([x0_buff, g0]).detach().numpy()
        # g0 = g0.detach().numpy()
        g0_guess = 1.2 * g0
        """MHE initial with g0 guess"""
        estimator = MHE(A_pi, B_pi, g0_guess, H, x_std, shift, model_pi, args)
        x_est_pi = np.zeros((x_test.shape[0] - 1, args['state_dim']))

        for t in range(x_test.shape[0] - 1):
            y = x_test[t, :]
            x_est_pi[t, :] = estimator(y, args)   # In this step we can add a condition that t > MHE_horizon or just run from 0
            estimator.update(u_test[t, :])
            print("step:{}; x_true:{},x_pred{}".format(t, y, x_est_pi[t, :]))
        err_pi = mse_np(x_est_pi, x_test[:-1, :])

        print("----------------without pi-----------------")
        args['if_sigma'] = False
        args['MODEL_SAVE'] = address_nopi +'model_v1.pt'
        args['SAVE_A1'] = address_pi +'A1.pt'
        args['SAVE_B1'] = address_pi +'B1.pt'
        args['SAVE_C1'] = address_pi +'C1.pt'
        model_nopi = restore_data(args)
        A_nopi = model_nopi.A_1.detach().numpy()
        B_nopi = model_nopi.B_1.detach().numpy()

        """MHE initial with g0 guess"""
        estimator = MHE(A_nopi, B_nopi, g0_guess, H, x_std, shift, model_nopi, args)
        x_est_nopi = np.zeros((x_test.shape[0] - 1, args['state_dim']))

        for t in range(x_test.shape[0]-1):
            y = x_test[t, :]
            # print(u_test[t, :])
            x_est_nopi[t, :] = estimator(y, args)
            estimator.update(u_test[t, :])
            print("step:{}; x_true:{},x_pred{}".format(t, y, x_est_nopi[t, :]))

        err_nopi = mse_np(x_est_nopi, x_test[:-1, :])


        print("ERROR without pi:{}; with pi:{}".format(err_nopi, err_pi))

        x_test = x_test * shift[1].cpu().numpy() + shift[0].cpu().numpy()
        x_est_nopi = x_est_nopi * shift[1].cpu().numpy() + shift[0].cpu().numpy()
        x_est_pi = x_est_pi * shift[1].cpu().numpy() + shift[0].cpu().numpy()
        """save curve"""
        torch.save(x_test, "estimation_result/physics/x_test.pt")
        torch.save(x_est_pi, "estimation_result/physics/x_est_pi.pt")
        torch.save(x_est_nopi, "estimation_result/physics/x_est_nopi.pt")

        # -------------- plot -------------- #
        color1 = "#038355"
        color2 = "#ffc34e"
        font = {'family': 'Times New Roman', 'size': 12}
        titles = ['XA1', 'XB1', 'T1',
                  'XA2', 'XB2', 'T2',
                  'XA3', 'XB3', 'T3']
        plt.rc('font', **font)
        f, axs = plt.subplots(3, 3, sharex=True, figsize=(15, 9))
        for i in range(3):
            for j in range(3):
                axs[i, j].plot(x_test[:, i * 3 + j], label='Ground Truth', color='k', linewidth=2)
                axs[i, j].plot(x_est_pi[:, i * 3 + j], label='Physics-informed Koopman', color=color1, linewidth=2)
                axs[i, j].plot(x_est_nopi[:, i * 3 + j], '--', label='Koopman', color=color2, linewidth=2)
                axs[i, j].set_title(titles[i * 3 + j])
                axs[i, j].legend().set_visible(False)
        for ax in axs[-1, :]:
            ax.set_xlabel('Time Steps')
        plt.tight_layout()
        handles, labels = axs[0, 0].get_legend_handles_labels()
        # f.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        f.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0.02))
        plt.subplots_adjust(bottom=0.15)
        plt.savefig('estimation_result/physics/MHE_pi.pdf')
        plt.show()

    if test_sigma:
        print("----------------without sigma-----------------")
        args['if_sigma'] = False
        # introduce model
        args['MODEL_SAVE'] = address_pi + 'model_v1.pt'
        # pre-trained noise generate model
        args['NOISE_SAVE'] = address_noise + 'noise.pt'
        args['SAVE_A1'] = address_pi + 'A1.pt'
        args['SAVE_B1'] = address_pi + 'B1.pt'
        args['SAVE_C1'] = address_pi + 'C1.pt'
        model_pi = restore_data(args)
        A_pi = model_pi.A_1.detach().numpy()
        B_pi = model_pi.B_1.detach().numpy()
        g0_holder = model_pi.net(x0_buff)
        g0_holder = torch.cat([x0_buff, g0_holder]).detach().numpy()
        g0_guess = 1.2 * g0_holder
        """MHE initial with g0 guess"""
        estimator_1 = MHE(A_pi, B_pi, g0_guess, H, x_std, shift, model_pi, args)
        x_est_nos = np.zeros((x_test.shape[0] - 1, args['state_dim']))

        for t in range(x_test.shape[0] - 1):
            y = x_test[t, :]
            x_est_nos[t, :] = estimator_1(y, args)
            estimator_1.update(u_test[t, :])
            print("step:{}; x_true:{},x_pred{}".format(t, y, x_est_nos[t, :]))
        err_nos = mse_np(x_est_nos, x_test[:-1, :])

        print("----------------with sigma-----------------")
        args['if_sigma'] = True
        """MHE initial with g0 guess"""
        estimator_2 = MHE(A_pi, B_pi, g0_guess, H, x_std, shift, model_pi, args)
        x_est_s = np.zeros((x_test.shape[0] - 1, args['state_dim']))
        for t in range(x_test.shape[0] - 1):
            y = x_test[t, :]
            x_est_s[t, :] = estimator_2(y, args)
            estimator_2.update(u_test[t, :])
            print("step:{}; x_true:{},x_pred{}".format(t, y, x_est_s[t, :]))
        err_s = mse_np(x_est_s, x_test[:-1, :])
        print("ERROR fixed weights:{}; adaptive weights:{}".format(err_nos, err_s))

        x_test = x_test * shift[1].cpu().numpy() + shift[0].cpu().numpy()
        x_est_nos = x_est_nos * shift[1].cpu().numpy() + shift[0].cpu().numpy()
        x_est_s = x_est_s * shift[1].cpu().numpy() + shift[0].cpu().numpy()
        """save curve"""
        torch.save(x_test, "estimation_result/noise/x_test.pt")
        torch.save(x_est_nos, "estimation_result/noise/x_est_nos.pt")
        torch.save(x_est_s, "estimation_result/noise/x_est_s.pt")

        # -------------- plot -------------- #
        color1 = "#038355"
        color2 = "#ffc34e"
        font = {'family': 'Times New Roman', 'size': 12}
        titles = ['XA1', 'XB1', 'T1',
                  'XA2', 'XB2', 'T2',
                  'XA3', 'XB3', 'T3']
        plt.rc('font', **font)
        f, axs = plt.subplots(3, 3, sharex=True, figsize=(15, 9))
        for i in range(3):
            for j in range(3):
                axs[i, j].plot(x_test[:, i * 3 + j], label='Ground truth', color='k', linewidth=2)
                axs[i, j].plot(x_est_s[:, i * 3 + j], label='Self-adaptive weights', color=color1, linewidth=2)
                axs[i, j].plot(x_est_nos[:, i * 3 + j], '--', label='Fixed weights', color=color2, linewidth=2)
                axs[i, j].set_title(titles[i * 3 + j])
                axs[i, j].legend().set_visible(False)
        for ax in axs[-1, :]:
            ax.set_xlabel('Time Steps')
        plt.tight_layout()
        handles, labels = axs[0, 0].get_legend_handles_labels()
        f.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0.02))
        plt.subplots_adjust(bottom=0.15)
        plt.savefig('estimation_result/noise/MHE_noise_compare.pdf')
        plt.show()

def shift_scale(x, u, shift_ = None):
    x_choice = (x - shift_[0])/shift_[1]
    u_choice = (u - shift_[2])/shift_[3]
    return x_choice, u_choice


def shift_rescale(x, u, shift_ = None):
    x_re = x * shift_[1].numpy() + shift_[0].numpy()
    u_re = u * shift_[3].numpy() + shift_[2].numpy()
    return x_re, u_re

# def shift_minmax(x, u, shift_ = None):
#     x_choice = (x - shift_[1])/(shift_[0]-shift_[1])
#     u_choice = (u - shift_[3])/(shift_[2]-shift_[3])
#     return x_choice, u_choice
#
# def shift_reminmax(x, shift_ = None):
#     x_re = x * (shift_[0].numpy-shift_[1].numpy) + shift_[1].numpy
#     return x_re

def restore_data(args):
    args['if_mix'] = False
    from Desko import Koopman_Desko
    model = Koopman_Desko(args)
    # restore variables
    model.parameter_restore(args)
    return model


def mse_measure(list_true, list_pred):
    if len(list_true) != len(list_pred):
        raise ValueError("Lists must have the same length.")

    mse_list = []
    for x, y in zip(list_true, list_pred):
        mse = np.mean((x - y) ** 2)
        mse_list.append(mse)
    mse = np.mean(mse_list)
    return mse


def mse_np(x, y):
    assert x.shape == y.shape, "Arrays must have the same shape"
    squared_diff = (x - y) ** 2
    sum_squared_diff = np.sum(squared_diff)
    mse_value = sum_squared_diff / np.prod(x.shape)
    return mse_value


def scale_sigma(sigma):
    min_val = torch.min(sigma)
    max_val = torch.max(sigma)
    range = max_val - min_val
    target_min = 0.9
    target_max = 1.1
    scale = (target_max - target_min) / range
    normalized_sigma = (sigma - min_val) * scale + target_min
    return normalized_sigma


class ReGenerate():
    def __init__(self, args, env):
        self.batch_size = args['batch_size']
        self.seq_length = args['pred_horizon']
        self.env = env
        self.total_steps = 0
        self.length = args['mhe_test_length']
        print("generating data...")
        self._generate_data(args)

    def _generate_data(self, args):
        # Generate test scenario
        self.x_test = []
        self.x_test.append(self.env.reset())
        action = self.env.get_testaction(args['max_test_ep_steps']*self.length)
        self.u_test = action
        for t in range(1, args['max_test_ep_steps']*self.length):
            step_info = self.env.step(t, self.u_test)
            # self.u_test.append(step_info[1])
            self.x_test.append(np.squeeze(step_info[0]))
            if step_info[3]['data_collection_done']:
                break

        x = torch.stack(self.x_test).float()
        u = self.u_test[:-1, :].float()

        # Reshape and trim data sets
        self.x_test = x.reshape(-1, x.shape[0], args['state_dim']).to(args['device'])
        self.u_test = u.reshape(-1, x.shape[0]-1, args['act_dim']).to(args['device'])

        self.shift_x = torch.mean(self.x_test, axis=(0,1))
        self.scale_x = torch.std(self.x_test, axis=(0,1))
        self.shift_u = torch.mean(self.u_test, axis=(0,1))
        self.scale_u = torch.std(self.u_test, axis=(0,1))
        self.shift_ = [self.shift_x,self.scale_x,self.shift_u,self.scale_u]


if __name__ == "__main__":
    main()