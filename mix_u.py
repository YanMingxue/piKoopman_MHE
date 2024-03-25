import torch
import numpy as np
import math
import copy

import torch.nn as nn
import torch.optim as optim
import torch.distributions as torchd
from CSTR import CSTR_system
from three_tanks import three_tank_system
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt

# ===========for matplotlib==============#
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

SCALE_DIAG_MIN_MAX = (-20, 2)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class Koopman_Desko(object):
    """
    feature:
    -Encoder-decoder
    -LSTM
    -Deterministic

    """

    def __init__(
            self,
            args,
            **kwargs
    ):
        self.shift = None
        self.scale = None
        self.shift_u = None
        self.scale_u = None

        self.d = torch.nn.Parameter(torch.ones(1), requires_grad=True)  # data loss weight parameter
        self.p1 = torch.nn.Parameter(torch.ones(1), requires_grad=True)  # physics loss weight parameter
        # self.p2 = torch.nn.Parameter(torch.ones(1), requires_grad=True)  # physics loss weight parameter
        self.p3 = torch.nn.Parameter(torch.ones(1), requires_grad=True)  # physics loss weight parameter

        self.loss = 0
        self.d_loss = 0  # data loss
        self.p1_loss = 0
        self.p2_loss = 0
        self.p3_loss = 0  # physics loss

        self.test_loss = 0

        if args['ABCD'] == 1:
            if args['extend_state']:
                self._create_koopman_matrix_a1_extend(args)
            else:
                self._create_koopman_matrix_a1(args)
        elif args['ABCD'] == 2:
            if args['extend_state']:
                self._create_koopman_matrix_a2_extend(args)
            else:
                self._create_koopman_matrix_a2(args)

        self.net = MLP(args)
        self.net.apply(weights_init)

        self.mix = Mix_x_u(args)
        self.mix.apply(weights_init)

        self.net_para = {}
        self.mix_para = {}

        self.loss_buff = 100000

        # self.optimizer = optim.Adam([{'params': self.net.parameters(),'lr':args['lr']},{'params': [self.A_1,self.B_1,self.C_1,],'lr':args['lr']}])

        self.optimizer1 = optim.Adam(
            [{'params': self.net.parameters(), 'lr': args['lr1'], 'weight_decay': args['weight_decay']}])
        self.optimizer1_sch = torch.optim.lr_scheduler.StepLR(self.optimizer1, step_size=args['optimize_step'],
                                                              gamma=args['gamma'])

        self.optimizer3 = optim.Adam(
            [{'params': self.mix.parameters(), 'lr': args['lr1'], 'weight_decay': args['weight_decay']}])
        self.optimizer3_sch = torch.optim.lr_scheduler.StepLR(self.optimizer3, step_size=args['optimize_step'],
                                                              gamma=args['gamma'])

        if args['ABCD'] != 2:
            self.optimizer2 = optim.Adam(
                [{'params': [self.A_1, self.B_1, self.C_1, ], 'lr': args['lr2'], 'weight_decay': args['weight_decay']}])
            self.optimizer2_sch = torch.optim.lr_scheduler.StepLR(self.optimizer2, step_size=args['optimize_step'],
                                                                  gamma=args['gamma'])

        # loss weight optimizer
        self.optimizer4 = optim.Adam(
            [{'params': self.d, 'lr': args['lr1'], 'weight_decay': args['weight_decay']},
             {'params': self.p1, 'lr': args['lr1'], 'weight_decay': args['weight_decay']},
             # {'params': self.p2, 'lr': args['lr1'], 'weight_decay': args['weight_decay']},
             {'params': self.p3, 'lr': args['lr1'], 'weight_decay': args['weight_decay']}])
        self.optimizer4_sch = torch.optim.lr_scheduler.StepLR(self.optimizer4, step_size=args['optimize_step'],
                                                              gamma=args['gamma'])

        self.MODEL_SAVE = args['MODEL_SAVE']
        self.SAVE_A1 = args['SAVE_A1']
        self.SAVE_B1 = args['SAVE_B1']
        self.SAVE_C1 = args['SAVE_C1']

        self.OPTI1 = args['SAVE_OPTI1']
        self.OPTI2 = args['SAVE_OPTI2']

    def _create_koopman_matrix_a1(self, args):
        """
        In this approach
        A,B,C,D are regarded as the same as 
        the parameters in neural networks
        """
        scale = torch.tensor(0.01)

        self.A_1 = torch.randn(args['latent_dim'], args['latent_dim'])
        self.B_1 = torch.randn(args['act_dim'], args['latent_dim'])
        self.C_1 = torch.randn(args['latent_dim'], args['state_dim'])

        self.A_1 = scale * self.A_1
        self.A_1.requires_grad_(True)

        self.B_1 = scale * self.B_1
        self.B_1.requires_grad_(True)

        self.C_1 = scale * self.C_1
        self.C_1.requires_grad_(True)

    # def _lsqr_A_B_C(self):

    def _create_koopman_matrix_a1_extend(self, args):
        """
        In this approach
        A,B,C,D are regarded as the same as 
        the parameters in neural networks
        """
        scale = torch.tensor(0.01)
        args['act_expand'] = args['act_dim'] + args['mix_x_u']

        self.A_1 = torch.randn(args['latent_dim'] + args['state_dim'], args['latent_dim'] + args['state_dim'])
        # self.B_1 = torch.randn(args['act_dim'], args['latent_dim']+args['state_dim'])
        self.B_1 = torch.randn(args['act_expand'], args['latent_dim'] + args['state_dim'])
        self.C_1 = torch.randn(args['latent_dim'] + args['state_dim'], args['state_dim'])

        self.A_1 = scale * self.A_1 + torch.eye(args['latent_dim'] + args['state_dim'],
                                                args['latent_dim'] + args['state_dim'])
        self.A_1.requires_grad_(True)

        self.B_1 = scale * self.B_1
        self.B_1.requires_grad_(True)

        self.C_1 = scale * self.C_1
        self.C_1[:args['state_dim'], :] = torch.eye(args['state_dim'], args['state_dim']) \
                                          + torch.randn(args['state_dim'], args['state_dim']) * scale
        self.C_1.requires_grad_(True)

    def _create_koopman_matrix_a2(self, args):
        """
        In this approach
        A,B,C,D will be solved by traditional 
        system analysis method

        TODO:合理的ABC参数初始化
        """
        self.A_1 = torch.randn(args['latent_dim'], args['latent_dim'])
        self.B_1 = torch.randn(args['act_dim'], args['latent_dim'])
        self.C_1 = torch.randn(args['latent_dim'], args['state_dim'])

    def _create_koopman_matrix_a2_extend(self, args):
        """
        In this approach
        A,B,C,D are regarded as the same as 
        the parameters in neural networks
        """

        self.A_1 = torch.randn(args['latent_dim'] + args['state_dim'], args['latent_dim'] + args['state_dim'])
        self.B_1 = torch.randn(args['act_dim'], args['latent_dim'] + args['state_dim'])
        self.C_1 = torch.randn(args['latent_dim'] + args['state_dim'], args['state_dim'])

    def _create_encoder(self, args):
        self.encoder = encoder()
        # bijector-> for scale
        # 目前先不用概率分布 先写一个确定的
        # bijector = tfp.bijectors.Affine(shift=self.mean, scale_diag=self.sigma)

    def _create_optimizer(self, args):

        pass

    def learn(self, e, x_train, x_val, shift, args):
        self.train_data = DataLoader(dataset=x_train, batch_size=args['batch_size'], shuffle=True, drop_last=True)
        self.loss = 0
        self.d_loss = 0
        self.p1_loss = 0
        self.p2_loss = 0
        self.p3_loss = 0
        count = 0

        for x_, u_ in self.train_data:
            self.pred_forward(x_, u_, shift, args)
            count += 1

        self.optimizer1.zero_grad()
        if args['ABCD'] != 2:
            self.optimizer2.zero_grad()
            self.optimizer3.zero_grad()

        self.loss.backward()

        self.optimizer1.step()
        self.optimizer1_sch.step()
        self.optimizer3.step()
        self.optimizer3_sch.step()

        if args['ABCD'] != 2:
            self.optimizer2.step()
            # self.optimizer2.step()
            self.optimizer2_sch.step()
        self.optimizer4.step()
        self.optimizer4_sch.step()

        loss_buff = self.loss / count

        if loss_buff < self.loss_buff:
            # Koopman_Desko.parameter_store(self,args)
            # self.net_para = self.net.state_dict()
            # self.A_1_restore = self.A_1
            # self.B_1_restore = self.B_1
            # self.C_1_restore = self.C_1
            self.net_para = copy.deepcopy(self.net.state_dict())
            self.mix_para = copy.deepcopy(self.mix.state_dict())
            self.A_1_restore = copy.deepcopy(self.A_1)
            self.B_1_restore = copy.deepcopy(self.B_1)
            self.C_1_restore = copy.deepcopy(self.C_1)
            self.loss_buff = loss_buff

        # validation_test
        self.loss = 0
        self.d_loss = 0
        self.p1_loss = 0
        self.p2_loss = 0
        self.p3_loss = 0
        count = 0
        self.val_data = DataLoader(dataset=x_val, batch_size=args['batch_size'], shuffle=True, drop_last=True)
        for x_, u_ in self.val_data:
            self.pred_forward(x_, u_, shift, args)
            count += 1

        print('epoch {}: loss_traning data {} loss_val data {} minimal loss {} learning_rate {}'.format(e, loss_buff,
                                                                                                        self.loss / count,
                                                                                                        self.loss_buff,
                                                                                                        self.optimizer1_sch.get_last_lr()))
        # print('epoch {}: loss_traning data {} minimal loss {} learning_rate {}'.format(e, loss_buff, self.loss_buff, self.optimizer1_sch.get_last_lr()))


    def test_(self, test, args):
        self.test_data = DataLoader(dataset=test, batch_size=10, shuffle=True)
        for x_, u_ in self.test_data:
            self.pred_forward_test(x_, u_, args)

    def pred_forward(self, x, u, shift, args):

        pred_horizon = args['pred_horizon']
        ##
        # x = self.net(x)
        x0_buff = x[:, 0, :]
        x0 = self.net(x0_buff)
        x_pred_all = self.net(x)[:, 1:, :]

        loss = nn.MSELoss()

        if args['extend_state']:
            x0 = torch.cat([x0_buff, x0], 1)

        input_mix = torch.cat([u, x[:, :-1, :]], 2)  # x多一位，预测的XK+1

        u_mix = torch.cat([self.mix(input_mix), u], 2)  ##注意顺序是不是对的->时序

        if args['ABCD'] == 2:
            x1_buff = x[:, 1, :]
            x1 = self.net(x1_buff)
            if args['extend_state']:
                x1 = torch.cat([x1_buff, x1], 1)
                u0 = u_mix[:, 0, :]
                x_all = torch.cat([x0, u0], 1)
                K = torch.linalg.lstsq(x_all, x1).solution
                self.A_1 = K[:args['state_dim'] + args['latent_dim'], :]
                self.B_1 = K[-args['act_dim']:, :]
                print("try")

        x_pred_matrix = torch.zeros_like(x[:, 1:, :])
        x_pred_matrix_all = torch.zeros([x.shape[0], x.shape[1] - 1, args['latent_dim']])

        for i in range(pred_horizon - 1):
            # 这里的u还是用的收集到的u感觉有点奇怪
            # ----------测试转的效果---------#

            x0 = torch.matmul(x0, self.A_1) + torch.matmul(u_mix[:, i, :], self.B_1)
            if args['extend_state']:
                # x_pred = x0[:,-args['state_dim']:]
                x_pred = torch.matmul(x0, self.C_1)
            else:
                x_pred = torch.matmul(x0, self.C_1)
            # x_pred = torch.matmul(x0,self.C_1)
            x_pred_matrix_all[:, i, :] = x0[:, -args['latent_dim']:]
            x_pred_matrix[:, i, :] = x_pred

        ###########################
        # -----------loss function-------------#
        # X_k+1_prediction = X_k+1
        self.d_loss += loss(x_pred_matrix, x[:, 1:, :]) * 10
        # Z_k+1=AZ_k+Bu_k
        self.d_loss += loss(x_pred_all, x_pred_matrix_all)
        # -----------boundary loss-----------##
        self.d_loss += loss(x_pred_matrix[:, -1, :], x[:, -1, :]) * 10
        self.d_loss += loss(x_pred_all[:, -1, :], x_pred_matrix_all[:, -1, :])

        ##########################
        # --------------physics informed----------------- #
        # system = CSTR_system()
        system = three_tank_system()
        x_pred_matrix_re = x_pred_matrix.detach() * shift[1] + shift[0]
        u_re = u * shift[3] + shift[2]
        # ############################
        '''
        loss1:  X_last = X_1 + sum(ΔXi)
        action = input[step-1, :]
        '''
        dx = torch.zeros([args['batch_size'], args['state_dim']])
        pred_dx = torch.zeros([args['batch_size'], args['state_dim']])
        for j in range(x_pred_matrix.size(0)):
            dx_batch = torch.zeros([1, args['state_dim']])
            for i in range(x_pred_matrix.size(1) - 1):
                dx_t = system.derivative(x_pred_matrix_re[j, i, :].detach().numpy(), u_re[j, i + 1, :].detach().numpy())
                dx_batch += np.array(dx_t, dtype=np.float32) * system.h
            dx[j, :] = (dx_batch - shift[0]) / shift[1]
            pred_dx = x_pred_matrix[j, -1, :] - x_pred_matrix[j, 0, :]
        self.p1_loss = loss(pred_dx, dx)

        ##########################
        '''
        loss1_1:  X_last = f(f(f(X1,u1)))...
        '''
        dx = torch.zeros([args['batch_size'], args['state_dim']])
        pred_dx = torch.zeros([args['batch_size'], args['state_dim']])
        for j in range(x_pred_matrix.size(0)):
            # x1 -> x2
            dx_batch = system.derivative(x_pred_matrix_re[j, 0, :].detach().numpy(),
                                         u_re[j, 1, :].detach().numpy()) * system.h
            for i in range(x_pred_matrix.size(1) - 2):
                dx_batch_t = system.derivative(x_pred_matrix_re[j, 0, :].detach().numpy() + dx_batch,
                                               u_re[j, i + 1, :].detach().numpy())
                dx_batch += np.array(dx_batch_t, dtype=np.float32) * system.h
            dx_batch = torch.tensor(np.array(dx_batch, dtype=np.float32))
            dx[j, :] = (dx_batch - shift[0]) / shift[1]
            pred_dx = (x_pred_matrix[j, -1, :] - x_pred_matrix[j, 0, :])
        self.p1_loss += loss(pred_dx, dx)
        #
        # ##############################
        # '''
        # loss2: X_k+1 - X_k = ΔX_k
        # '''
        # dxk = torch.zeros([args['batch_size'], args['pred_horizon'] - 2, args['state_dim']])
        # pred_dxk = torch.zeros([args['batch_size'], args['pred_horizon'] - 2, args['state_dim']])
        # for j in range(x_pred_matrix.size(0)):
        #     for i in range(x_pred_matrix.size(1) - 1):
        #         dxk_s = system.derivative(x_pred_matrix_re[j, i, :].detach().numpy(),
        #                                   u_re[j, i + 1, :].detach().numpy())
        #         dxk_s = np.array(dxk_s, dtype=np.float32)
        #         dxk_s = torch.tensor(dxk_s * system.h).float()
        #         dxk[j, i, :] = (dxk_s - shift[0]) / shift[1]
        #         pred_dxk[j, i, :] = x_pred_matrix[j, i + 1, :] - x_pred_matrix[j, i, :]
        # self.p1_loss += loss(pred_dxk, dxk)
        # #
        # ##############################
        # '''
        # loss3: g(xk+1) = g(f(xk,uk))
        # '''
        # # shift: [self.shift_x,self.scale_x,self.shift_u,self.scale_u]
        # gxk = torch.zeros([args['batch_size'], args['pred_horizon'] - 2, args['latent_dim']])
        # pred_gxk = torch.zeros([args['batch_size'], args['pred_horizon'] - 2, args['latent_dim']])
        # x_pred_matrix_re = x_pred_matrix.detach() * shift[1] + shift[0]
        # u_re = u * shift[3] + shift[2]
        #
        # for j in range(x_pred_matrix_re.size(0)):
        #     for i in range(x_pred_matrix_re.size(1) - 1):
        #         dxk_s = system.derivative(x_pred_matrix_re[j, i, :].detach().numpy(),
        #                                   u_re[j, i + 1, :].detach().numpy())
        #         dxk_s = np.array(dxk_s, dtype=np.float32) * system.h
        #         dxk_s = torch.tensor(dxk_s)
        #         xk_s = (dxk_s + x_pred_matrix_re[j, i, :]).float()
        #         xkn_s = (xk_s - shift[0]) / shift[1]
        #         gxk_s = self.net(xkn_s)  # g(f(xk,uk))
        #         gxk[j, i, :] = gxk_s
        #         pred_gxk[j, i, :] = x_pred_matrix_all[j, i + 1, 0]  # g(xk+1)
        # self.p3_loss += loss(pred_gxk, gxk)
        #
        # # # self adaptive loss
        # self.loss = (1/pow(self.d, 2))*self.d_loss + (1/pow(self.p1, 2))*self.p1_loss\
        #             + (1/pow(self.p3, 2))*self.p3_loss + torch.log(1+1*pow(self.d, 2))\
        #             + torch.log(1+pow(self.p1, 2)) + torch.log(1+pow(self.p3, 2))

        # self.loss=(1/pow(self.d, 2))*self.d_loss + (1/pow(self.p1, 2))*self.p1_loss\
        #           + torch.log(1+10*pow(self.d, 2))\
        #           + torch.log(1+pow(self.p1, 2))
        self.loss = self.d_loss

        self.displace1 = x_pred[0, :]
        self.displace2 = x[0, i + 1, :]

    def pred_forward_test(self, x, u, shift, test, args, e=0):
        # print(x.shape) #[1,max_ep_steps*30,9]
        self.test_loss = 0
        x_pred_list = []
        x_sum_list = []
        x_real_list = []
        x_time_list = []
        plt.close()
        f, axs = plt.subplots(args['state_dim'], sharex=True, figsize=(15, 15))
        time_all = np.arange(x.shape[1])
        print("done")

        if test:
            for i in range(0, args['max_ep_steps']*args['test_steps'] - args['pred_horizon'] + 1, args['pred_horizon'] - 1):
            # for i in range(int(args['max_ep_steps']/(args['pred_horizon']-1))):
                x_pred = x[:, i:i + args['pred_horizon']]
                u_pred = u[:, i:i + args['pred_horizon']]
                x_pred_list_buff, x_real_list_buff, x_sum_list_buff = \
                    Koopman_Desko.pred_forward_test_buff(self, x_pred, u_pred, args)
                # print(len(x_pred_list_buff))  #[15,2]
                x_pred_list.append(np.array(x_pred_list_buff))
                x_real_list.append(np.array(x_real_list_buff))
                x_sum_list.append(np.array(x_sum_list_buff))
                x_time_list.append(np.arange(i + 1, i + args['pred_horizon']))

            print("test_loss{}".format(self.test_loss))
            ## scale back
            x = x * shift[1] + shift[0]
            x_pred_list = x_pred_list * shift[1] + shift[0]
            ##-------------------plot-----------------------##
            for i in range(args['state_dim']):
                axs[i].plot(time_all, x[:, :, i].T, 'k')
                for j in range(len(x_time_list)):
                    axs[i].plot(x_time_list[j], x_pred_list[j][:, :, i], 'r')

            plt.xlabel('Time Step')
            plt.savefig('data/predictions_' + str(e) + '.png')
            print("plot")

            return x_pred_list, x_real_list, x_sum_list

        ##----------------------------------------------##
        else:
            return Koopman_Desko.pred_forward_test_buff(self, x, u, args)

    def pred_forward_test_buff(self, x, u, args):
        pred_horizon = args['pred_horizon']
        # print(x.shape)  # [1,16,9]
        # 测试模式
        self.net.eval()

        x0_buff = x[:, 0, :]
        x0 = self.net(x0_buff)

        if args['extend_state']:
            x0 = torch.cat([x0_buff, x0], 1)

        input_mix = torch.cat([u, x[:, :, :]], 2)
        u = torch.cat([self.mix(input_mix), u], 2)  ##注意顺序是不是对的->时序
        # if args['act_expand'] > args['act_dim']:
        # if args['act_expand']  == 6:
        #     u = torch.cat([torch.square(u),u],2)
        # if args['act_expand']  == 9:
        #     u = torch.cat([torch.pow(u,3),torch.square(u),u],2)

        if args['ABCD'] == 2:
            x1_buff = x[:, 1, :]
            x1 = self.net(x1_buff)
            if args['extend_state']:
                x1 = torch.cat([x1_buff, x1], 1)
                u0 = u[:, 0, :]
                x_all = torch.cat([x0, u0], 1)
                K = torch.linalg.lstsq(x_all, x1)
                print("try")

        # x_next = []
        x_pred_list = []
        x_sum_list = []
        x_real_list = []

        loss_test = 0
        loss = nn.MSELoss()

        x_pred_all = self.net(x[:, :-1, :])

        x_pred_matrix = torch.zeros_like(x[:, 1:, :])
        for i in range(pred_horizon - 1):
            x0 = torch.matmul(x0, self.A_1) + torch.matmul(u[:, i, :], self.B_1)

            if args['extend_state']:
                # print("s")
                x_pred = x0[:, -args['state_dim']:]
                x_pred = torch.matmul(x0, self.C_1)
            else:
                x_pred = torch.matmul(x0, self.C_1)

            # x_pred = torch.matmul(x0,self.C_1)
            x_pred_matrix[:, i, :] = x_pred
            # print(x_pred)
            x_pred_list.append(x_pred.detach().numpy())
            x_real_list.append(x[:, i + 1, :].detach().numpy())
            # print(x[:,i+1,:])

        self.test_loss += loss(x_pred_matrix, x[:, 1:, :])/args['test_steps']
        # 回归训练模式
        self.net.train()
        # print(loss_test)

        return x_pred_list, x_real_list, x_sum_list
        # print("x_pred{},x_real{}".format(x_pred[7],x[7,i+1,:]))

    def parameter_store(self, args):

        # save nn
        torch.save(self.net_para, self.MODEL_SAVE)
        # save A1 B1 C1
        torch.save(self.A_1_restore, self.SAVE_A1)
        torch.save(self.B_1_restore, self.SAVE_B1)
        torch.save(self.C_1_restore, self.SAVE_C1)

        torch.save(self.optimizer1.state_dict(), self.OPTI1)
        if args['ABCD'] != 2:
            torch.save(self.optimizer2.state_dict(), self.OPTI2)

        print("store!!!")

    def parameter_restore(self, args):

        self.A_1 = torch.load(self.SAVE_A1)
        self.B_1 = torch.load(self.SAVE_B1)
        self.C_1 = torch.load(self.SAVE_C1)

        self.net = MLP(args)
        self.net.load_state_dict(torch.load(self.MODEL_SAVE))
        self.net.eval()

        self.optimizer1.load_state_dict(torch.load(self.OPTI1))
        if args['ABCD'] != 2:
            self.optimizer2.load_state_dict(torch.load(self.OPTI2))

        print("restore!")

    def set_shift_and_scale(self, replay_memory):

        self.shift = replay_memory.shift_[0]
        self.scale = replay_memory.shift_[1]
        self.shift_u = replay_memory.shift_[2]
        self.scale_u = replay_memory.shift_[3]

    def _create_optimizer(self, args):
        pass


class encoder(nn.Module):
    """
    encoder -> through LSTM

    input_dim -> 

    """

    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        outputs, (hidden, cell) = self.rnn(input)

        # outputs = [input, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        return outputs, hidden, cell


class MLP(nn.Module):

    def __init__(self, args):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(args['state_dim'], 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(100, 80),
            nn.Linear(128, args['latent_dim']),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(80, args['latent_dim']),
            # nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class Mix_x_u(nn.Module):

    def __init__(self, args):
        super(Mix_x_u, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(args['state_dim'] + args['act_dim'], 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(100, 80),
            nn.Linear(128, args['mix_x_u']),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(80, args['latent_dim']),
            # nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    pass
