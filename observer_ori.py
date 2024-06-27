import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
from three_tanks import three_tank_system as dreamer
# from replay_memory import ReplayMemory
import my_args

import progressbar



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
    def __init__(self, model, std, shift, args):
        self.pred_horizon = args['mhe_horizon']
        self.state_dim = args['state_dim']
        self.latent_dim = args['latent_dim']
        self.A = model.A_1.detach().numpy()
        self.B = model.B_1.detach().numpy()
        # self.C = model.C_1.detach().numpy()
        self.C_full = np.row_stack([np.eye(self.state_dim),
                                    np.zeros([self.latent_dim, self.state_dim])])
        self.select = [2, 5, 8]
        self.C = self.C_full[:, self.select]
        self.net = model.net
        self.std = std
        self.upper = torch.tensor([0.1, 0.4, 600, 0.1, 0.4, 600, 0.1, 0.3, 600])
        self.lower = torch.tensor([0, 0.1, 0, 0, 0.1, 0, 0, 0.1, 0])
        # self.delta_bound = [0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02]
        # self.deltalow_bound = [-0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02, -0.02]
        # self.delta_u_bound = [1e4,1e4,1e4]
        self.s_bound_low = np.array((self.lower - shift[0].cpu()) / shift[1].cpu())
        self.s_bound_high = np.array((self.upper - shift[0].cpu()) / shift[1].cpu())
        print(self.s_bound_high)
        # self.select = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # self.select = [0, 1, 3, 4, 6, 7]
        if args['if_sigma']:
            self.noisemlp = model.noisemlp
        self._build_matrices(args)

    def _build_matrices(self, args):
        print(self.std)
        std_mean = torch.mean(self.std)
        extended_std = torch.full((args['state_dim']+args['latent_dim'],), std_mean)
        extended_std[:9] = self.std
        extended_std = np.diag(extended_std)
        self.Q = np.linalg.inv(extended_std)
        print(self.Q)
        self.R = np.transpose(self.C) @ self.Q @ self.C
        # self.P = self.Q
        # self.Q = np.diag(0.1 * np.ones([self.latent_dim+self.state_dim]))
        # self.R = np.diag(0.1 * np.ones([self.state_dim]))
        self.P = np.diag(0.05 * np.ones([self.latent_dim+self.state_dim]))
        self.scale_1 = 1.
        self.scale_2 = 1.
        self.scale_3 = 1.

    def Weight_matrix_calculate(self, model, x):
        SCALE_DIAG_MIN_MAX = (-20, 2)
        log_sigma = model.noisemlp(x).detach()
        log_sigma = torch.clamp(log_sigma, min=SCALE_DIAG_MIN_MAX[0], max=SCALE_DIAG_MIN_MAX[1])
        self.sigma = torch.exp(log_sigma)
        self.sigma = self.sigma / torch.max(self.sigma)
        sigma_diag = self.sigma.numpy()
        sigma_matrix = np.diag(sigma_diag)
        self.Q = np.linalg.inv(sigma_matrix)
        # self.Q = np.linalg.inv(self.sigma)
        self.R = np.transpose(self.C) @ self.Q @ self.C


    # max l
    def MHE(self, x, u, g_init, model, args):
        self.g_pred = cp.Variable((self.pred_horizon, self.latent_dim + self.state_dim))
        self.x_pred = cp.Variable((self.pred_horizon, self.state_dim))
        self.max_l = cp.Variable()
        self.w = cp.Variable((self.pred_horizon, self.latent_dim + self.state_dim))
        self.x_true = x
        self.g_init = g_init
        self.u = u
        objective = 0.
        constraints =[]
        objective += self.scale_1 * cp.quad_form(self.g_pred[0, :] - self.g_init, self.P)
        for i in range(self.pred_horizon-1):
            # xk = torch.tensor(x[i, :])
            # g = model.net(xk)
            # g = torch.cat([xk, g])
            if args['if_sigma']:
                g = self.g_init
                self.Weight_matrix_calculate(model, torch.tensor(g, dtype=torch.float))
                # print(self.R)
            constraints += [self.g_pred[i + 1, :] == self.g_pred[i, :] @ self.A + self.u[i, :] @ self.B + self.w[i, :]]
            constraints += [self.x_pred[i, :] == self.g_pred[i, :] @ self.C_full]
            constraints += [self.s_bound_low <= self.x_pred[i, :],
                            self.x_pred[i, :] <= self.s_bound_high]
            l = cp.quad_form(self.x_true[i, self.select] - self.g_pred[i, :] @ self.C, self.R) + cp.quad_form(self.w[i, :], self.Q)
            constraints += [l <= self.max_l]
            objective += self.scale_2 * l
        objective += self.scale_3 * self.max_l

        self.prob = cp.Problem(cp.Minimize(objective), constraints)

        try:
            self.prob.solve(solver=cp.MOSEK)

        except cp.error.SolverError as e:
            print("Solver error:", e)
            pass

        optimal_g = self.g_pred[0, :].value
        g_pre = self.g_pred[1, :].value
        x_est = self.x_pred[0, :].value

        return optimal_g, g_pre, x_est

    # base MHE
    def MHE(self, x, u, g_init, model, args):
        self.g_pred = cp.Variable((self.pred_horizon, self.latent_dim + self.state_dim))
        self.x_pred = cp.Variable((self.pred_horizon, self.state_dim))
        self.x_true = x
        self.g_init = g_init
        self.u = u

        objective = 0.
        constraints =[]
        objective += self.scale_1 * cp.quad_form(self.g_pred[0, :] - self.g_init, 0.05*self.Q)


        for i in range(self.pred_horizon-1):
            # xk = torch.tensor(x[i, :])
            # g = model.net(xk)
            # g = torch.cat([xk, g])
            if args['if_sigma']:
                g = self.g_init
                self.Weight_matrix_calculate(model, torch.tensor(g, dtype=torch.float))
            # if np.all(self.u[i, :] - self.u[i-1, :] <= self.delta_u_bound):
            #     constraints += [self.deltalow_bound <= (self.x_pred[i + 1, :] - self.x_pred[i, :]),
            #         (self.x_pred[i + 1, :] - self.x_pred[i, :]) <= self.delta_bound]
            constraints += [self.g_pred[i + 1, :] == self.g_pred[i, :] @ self.A + self.u[i, :] @ self.B]
            constraints += [self.x_pred[i, :] == self.g_pred[i, :] @ self.C_full]
            constraints += [self.s_bound_low <= self.x_pred[i, :],
                            self.x_pred[i, :] <= self.s_bound_high]
            l = cp.quad_form(self.x_true[i, self.select] - self.g_pred[i, :] @ self.C, self.R)
            objective += self.scale_2 * l

        self.prob = cp.Problem(cp.Minimize(objective), constraints)

        try:
            self.prob.solve(solver=cp.MOSEK)

        except cp.error.SolverError as e:
            print("Solver error:", e)
            pass

        optimal_g = self.g_pred[0, :].value
        g_pre = self.g_pred[1, :].value
        x_est = self.x_pred[0, :].value

        return optimal_g, g_pre, x_est


def estimate(env,args):
    args['re_generate_test'] = True
    # plot list
    x_true_list = []
    x_est_list = []
    x_est_list2 = []

    # introduce model
    # args['MODEL_SAVE'] = "save_model/pi large noise_0.059/model_v1.pt"
    # args['NOISE_SAVE'] = "save_model/pi large noise_0.059/noise.pt"
    # args['SAVE_A1'] = "save_model/pi large noise_0.059/A1.pt"
    # args['SAVE_B1'] = "save_model/pi large noise_0.059/B1.pt"
    # args['SAVE_C1'] = "save_model/pi large noise_0.059/C1.pt"
    model = restore_data(args)
    # true data  ----  re-generate
    if args['re_generate_test']:
        regenerate = ReGenerate(args, env)
        x_test = torch.tensor(regenerate.x_test[:, 50:, :]).squeeze().float()  # [1, n, 9]
        u_test = torch.tensor(regenerate.u_test[:, 50:, :]).squeeze().float()
        x_test, u_test = shift_scale(x_test, u_test, regenerate.shift_)
        shift = regenerate.shift_
    else:
        x_test = torch.load(args['SAVE_TEST_X'])
        u_test = torch.load(args['SAVE_TEST_U'])
        shift = torch.load(args['SAVE_SHIFT'])
        x_test = torch.tensor(x_test[:, :, :]).squeeze().float()
        u_test = torch.tensor(u_test[:, :, :]).squeeze().float()
        x_test, u_test = shift_scale(x_test, u_test, shift)

    # std of scaled data
    # x_std = torch.std(x_test, axis=0)
    x_std = torch.std(x_test, axis=0)
    x_std = x_std / torch.max(x_std)
    MHE_method = base_MHE(model, x_std, shift, args)

    x0_buff = x_test[0, :].cpu()
    g0 = model.net(x0_buff)
    g0 = torch.cat([x0_buff, g0])
    g0 = g0.detach().numpy()

    x_test = x_test.cpu().detach().numpy()
    u_test = u_test.cpu().detach().numpy()

    # ----------------with pi------------------#

    #------------------sigma-------------------#
    args['if_sigma'] = True
    for i in range(x_test.shape[0] - args['mhe_horizon']):
        x = x_test[i:i + args['mhe_horizon'], :]
        u = u_test[i:i + args['mhe_horizon'] - 1, :]
        # if (i % 100 == 0):
        #     xi_buff = torch.tensor(x_test[i, :])
        #     g_i = model.net(xi_buff)
        #     g_i = torch.cat([xi_buff, g_i])
        #     g_i = g_i.detach().numpy()
        #     g_opt, g0, x_est = MHE_method.MHE(x, u, g_i, model, args)
        # else:
        # set initial value again: very good result.
        # model not accurate enough
        g_opt, g0, x_est = MHE_method.MHE(x, u, g0, model, args)
        x_true_list.append(x[0, :])
        x_est_list.append(x_est)
        print("step:{}; x_true:{},x_pred{}".format(i, x[0, :], x_est))

    error = mse_measure(x_true_list, x_est_list)


    # # ------------------without pi-------------------#
    # # introduce model
    # args['MODEL_SAVE'] = "save_model/model_v1.pt"
    # args['NOISE_SAVE'] = "save_model/noise.pt"
    # args['SAVE_A1'] = "save_model/A1.pt"
    # args['SAVE_B1'] = "save_model/B1.pt"
    # args['SAVE_C1'] = "save_model/C1.pt"
    # model = restore_data(args)
    # # true data  ---- re-generate
    # # std of scaled data
    # # x_std = torch.std(x_test, axis=0)
    # MHE_method = base_MHE(model, x_std, shift, args)
    # # ------------------sigma-------------------#
    # args['if_sigma'] = True
    # for i in range(x_test.shape[0] - args['mhe_horizon']):
    #     x = x_test[i:i + args['mhe_horizon'], :]
    #     u = u_test[i:i + args['mhe_horizon'] - 1, :]
    #     # if (i % 200 == 0):
    #     #     xi_buff = torch.tensor(x_test[i, :])
    #     #     g_i = model.net(xi_buff)
    #     #     g_i = torch.cat([xi_buff, g_i])
    #     #     g_i = g_i.detach().numpy()
    #     #     g_opt, g0, x_est = MHE_method.MHE(x, u, g_i, model, args)
    #     # else:
    #     g_opt, g0, x_est = MHE_method.MHE(x, u, g0, model, args)
    #     # x_true_list.append(x[0, :])
    #     x_est_list2.append(x_est)
    #     print("step:{}; x_true:{},x_pred{}".format(i, x[0, :], x_est))

    # error2 = mse_measure(x_true_list, x_est_list2)
    #
    # #------------------no sigma-------------------#
    # args['if_sigma'] = False
    # for i in range(x_test.shape[0] - args['mhe_horizon']):
    #     x = x_test[i:i + args['mhe_horizon'], :]
    #     u = u_test[i:i + args['mhe_horizon'] - 1, :]
    #     g_opt, g0, x_est = MHE_method.MHE(x, u, g0, model, args)
    #     x_est_list2.append(x_est)
    #     print("step:{}; x_true:{},x_pred{}".format(i, x[0, :], x_est))
    #
    # error2 = mse_measure(x_true_list, x_est_list2)


    print("prediction error is:", error)
    # print("prediction error no pi is:", error2)
    print("save result....")

    torch.save(x_true_list, "result/x_true.pt")
    torch.save(x_est_list, "result/x_withpi.pt")
    torch.save(x_est_list2, "result/x_withoutpi.pt")


    error_list1 = [np.mean(abs(true_val - est_val)) for true_val, est_val in zip(x_true_list, x_est_list)]
    error_list2 = [np.mean(abs(true_val - est_val)) for true_val, est_val in zip(x_true_list, x_est_list2)]
    err_array = np.array(error_list1)
    err_array2 = np.array(error_list2)

    #####-------------plot_state----------------###
    if args['re_generate_test']:
        shift = regenerate.shift_
    # else:
    #     shift = shift
    x_true_list = [x * shift[1].cpu().numpy() + shift[0].cpu().numpy() for x in x_true_list]
    x_est_list = [x * shift[1].cpu().numpy() + shift[0].cpu().numpy() for x in x_est_list]
    x_est_list2 = [x * shift[1].cpu().numpy() + shift[0].cpu().numpy() for x in x_est_list2]
    # x_true_list = x_true_list * shift[1] + shift[0]
    # x_est_list = x_est_list * shift[1] + shift[0]
    x_true_array = np.array(x_true_list)
    x_est_array = np.array(x_est_list)
    x_est_array2 = np.array(x_est_list2)

    plt.figure(1)
    color1 = "#038355"  # 孔雀绿
    color2 = "#ffc34e"  # 向日黄
    font = {'family': 'Times New Roman', 'size': 12}
    plt.rc('font', **font)
    f, axs = plt.subplots(3, 3, sharex=True, figsize=(15, 9))
    for i in range(3):  # 行索引
        for j in range(3):  # 列索引
            # 绘制每个子图
            axs[i, j].plot(x_true_array[:, i * 3 + j], label='x_true', color='k', linewidth=2)
            axs[i, j].plot(x_est_array[:, i * 3 + j], '--', label='x_with pi', color=color1, linewidth=2)
            # axs[i, j].plot(x_est_array2[:, i * 3 + j], '-.', label='x_without pi', color=color2, linewidth=2)
            axs[i, j].set_title(f'State {i * 3 + j + 1}')  # 设置子图标题
            axs[i, j].legend().set_visible(False)
    for ax in axs[-1, :]:
        ax.set_xlabel('Steps')
    # 调整子图布局
    plt.tight_layout()
    handles, labels = axs[0, 0].get_legend_handles_labels()
    f.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    plt.savefig('data/MHE.png')
    plt.show()

    #####-------------plot_error----------------###
    plt.figure(2)
    plt.rc('font', **font)
    plt.plot(err_array, '--', label='x_with pi', color=color1, linewidth=2)
    plt.plot(err_array2, '-.', label='x_without pi', color=color2, linewidth=2)
    plt.title('Error Comparison')
    plt.legend()
    # legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)
    # legend.get_frame().set_facecolor('lightgrey')  # 设置图例背景颜色
    # legend.get_frame().set_edgecolor('black')  # 设置图例边框颜色
    # legend.get_frame().set_linewidth(1.5)  # 设置图例边框宽度
    # # f.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    plt.savefig('data/MHEerror.png')
    plt.show()




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


class ReGenerate():
    def __init__(self, args, env):
        self.batch_size = args['batch_size']
        self.seq_length = args['pred_horizon']
        self.env = env
        self.total_steps = 0
        self.length = 6

        # print('validation fraction: ', args['val_frac'])


        if args['import_saved_data'] or args['continue_data_collection']:
            self._restore_data('./data/' + args['env_name'])
            # self._process_data(args)

            # print('creating splits...')
            # self._create_split(args)
            # self._determine_shift_and_scale(args)
        else:
            print("generating data...")
            self._generate_data(args)

    def _generate_data(self, args):
        # Generate test scenario
        self.x_test = []
        self.x_test.append(self.env.reset())
        action = self.env.get_testaction(args['max_ep_steps']*self.length)
        self.u_test = action
        for t in range(1, args['max_ep_steps']*self.length):
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

        print("x:", self.x_test.shape)
        print("shift:", self.shift_)



if __name__ == '__main__':
    main()
