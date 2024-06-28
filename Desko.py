import torch
import numpy as np
import math
import copy
import torch.distributions as dist
import torch.distributions.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.distributions as torchd
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
import torch.nn.functional as F

from three_tanks import three_tank_system

#===========for matplotlib==============#
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'





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

        self.d = torch.nn.Parameter(torch.ones(1).to(args['device']), requires_grad=True) # data loss weight parameter
        self.p2 = torch.nn.Parameter(torch.ones(1).to(args['device']),
                                     requires_grad=True)  # physics loss weight parameter
        self.p3 = torch.nn.Parameter(torch.ones(1).to(args['device']),
                                     requires_grad=True)# physics loss weight parameter

        # self.loss = 0
        # self.d_loss = 0  # data loss
        # self.p2_loss = 0
        # self.p3_loss = 0  # physics loss

        if args['ABCD'] == 1:
            if args['extend_state']:
                # 用这个
                self._create_koopman_matrix_a1_extend(args)
            else:
                self._create_koopman_matrix_a1(args)
        elif args['ABCD'] == 2:
            if args['extend_state']:
                self._create_koopman_matrix_a2_extend(args)
            else:
                self._create_koopman_matrix_a2(args)

        self.net = MLP(args).to(args['device'])
        self.net.apply(weights_init)

        self.noisemlp = Noise_MLP(args).to(args['device'])

        self.net_para = {}
        self.noise_para = {}

        self.loss_buff = 100000


        # self.optimizer = optim.Adam([{'params': self.net.parameters(),'lr':args['lr']},{'params': [self.A_1,self.B_1,self.C_1,],'lr':args['lr']}])

        self.optimizer1 = optim.Adam([{'params': self.net.parameters(), 'lr': args['lr1'], 'weight_decay':args['weight_decay']},
                                      {'params': self.noisemlp.parameters(), 'lr': args['lr1'], 'weight_decay': args['weight_decay']}])
        self.optimizer1_sch = torch.optim.lr_scheduler.StepLR(self.optimizer1, step_size=args['optimize_step'], gamma=args['gamma'])

        if args['ABCD'] != 2:
            self.optimizer2 = optim.Adam([{'params': [self.A_1,self.B_1,self.C_1,],'lr':args['lr2'],'weight_decay':args['weight_decay']}])
            self.optimizer2_sch = torch.optim.lr_scheduler.StepLR(self.optimizer2, step_size=args['optimize_step'], gamma=args['gamma'])

        self.optimizer3 = optim.Adam(
            [{'params': self.d, 'lr': args['lr3'], 'weight_decay': args['weight_decay']},
             {'params': self.p2, 'lr': args['lr3'], 'weight_decay': args['weight_decay']},
             {'params': self.p3, 'lr': args['lr3'], 'weight_decay': args['weight_decay']}])
        self.optimizer3_sch = torch.optim.lr_scheduler.StepLR(self.optimizer3, step_size=args['optimize_step'],
                                                              gamma=args['gamma'])

        self.MODEL_SAVE = args['MODEL_SAVE']
        self.NOISE_SAVE = args['NOISE_SAVE']
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
        
        self.A_1 = torch.randn(args['latent_dim'], args['latent_dim']).to(args['device'])
        self.B_1 = torch.randn(args['act_dim'], args['latent_dim']).to(args['device'])
        self.C_1 = torch.randn(args['latent_dim'], args['state_dim']).to(args['device'])

        self.A_1 = scale*self.A_1
        self.A_1.requires_grad_(True)

        self.B_1 = scale*self.B_1
        self.B_1.requires_grad_(True)

        self.C_1 = scale*self.C_1
        self.C_1.requires_grad_(True)
    


    def _create_koopman_matrix_a1_extend(self, args):
        # use this one
        """
        In this approach
        A,B,C,D are regarded as the same as 
        the parameters in neural networks
        """
        scale = torch.tensor(0.01)
        
        self.A_1 = torch.randn(args['latent_dim']+args['state_dim'], args['latent_dim']+args['state_dim']).to(args['device'])
        self.B_1 = torch.randn(args['act_expand'], args['latent_dim']+args['state_dim']).to(args['device'])
        self.C_1 = torch.zeros(args['latent_dim']+args['state_dim'], args['state_dim']).to(args['device'])
        # self.C_2 = torch.randn(args['latent_dim']+args['state_dim'], args['output_dim']).to(args['device'])

        self.A_1 = scale * self.A_1+torch.eye(args['latent_dim']+args['state_dim'], args['latent_dim']+args['state_dim']).to(args['device'])
        self.A_1.requires_grad_(True)

        self.B_1 = scale*self.B_1
        self.B_1.requires_grad_(True)

        # C does not to be learned. Select as [I,0]
        # self.C_1 = scale*self.C_1
        # self.C_1[:args['state_dim'],:]= (1.0*torch.eye(args['state_dim'],args['state_dim'])\
        #                                  +torch.randn(args['state_dim'],args['state_dim'])*scale).to(args['device'])
        # self.C_1.requires_grad_(True)

        self.C_1[:args['state_dim'],:] = torch.eye(args['state_dim'],args['state_dim'])



    def _create_koopman_matrix_a2(self, args):
        """
        In this approach
        A,B,C,D will be solved by traditional 
        system analysis method

        TODO: Initialization of A B and C
        """ 
        self.A_1 = torch.randn(args['latent_dim'], args['latent_dim']).to(args['device'])
        self.B_1 = torch.randn(args['act_dim'], args['latent_dim']).to(args['device'])
        self.C_1 = torch.randn(args['latent_dim'], args['state_dim']).to(args['device'])

    def _create_koopman_matrix_a2_extend(self, args):
        """
        In this approach
        A,B,C,D are regarded as the same as 
        the parameters in neural networks
        """
        
        self.A_1 = torch.randn(args['latent_dim']+args['state_dim'], args['latent_dim']+args['state_dim']).to(args['device'])
        self.B_1 = torch.randn(args['act_dim'], args['latent_dim']+args['state_dim']).to(args['device'])
        self.C_1 = torch.randn(args['latent_dim']+args['state_dim'], args['state_dim']).to(args['device'])

    def _create_optimizer(self, args):
        pass

    def learn(self, e, x_train,x_val,shift,args):
        self.train_data = DataLoader(dataset=x_train, batch_size=args['batch_size'], shuffle=True, drop_last=True)
        for x_, u_ in self.train_data:
            self.loss = 0
            self.d_loss = 0
            self.p2_loss = 0
            self.p3_loss = 0
            x_ = x_.to(args['device'])
            u_ = u_.to(args['device'])
            self.pred_forward(x_, u_, shift, args)

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
                self.optimizer2_sch.step()

        if args['if_pi']:
            loss_buff = self.d_loss  # take the loss of the last batch as L_d
        else:
            loss_buff = self.loss

        if loss_buff < self.loss_buff:
            self.net_para = copy.deepcopy(self.net.state_dict())
            self.noise_para = copy.deepcopy(self.noisemlp.state_dict())
            self.A_1_restore = copy.deepcopy(self.A_1)
            self.B_1_restore = copy.deepcopy(self.B_1)
            self.C_1_restore = copy.deepcopy(self.C_1)
            self.loss_buff = loss_buff


        # validation_test
        self.train_data = DataLoader(dataset=x_val, batch_size=args['batch_size'], shuffle=True, drop_last=True)
        for x_, u_ in self.train_data:
            self.loss = 0
            self.d_loss = 0
            self.p1_loss = 0
            self.p2_loss = 0
            self.p3_loss = 0
            x_ = x_.to(args['device'])
            u_ = u_.to(args['device'])
            self.pred_forward(x_, u_, shift, args)

        if args['if_pi']:
            print(
                'epoch {}: loss_traning data {} loss_val data {} minimal loss {} learning_rate {}'.format(e, loss_buff,
                                                                                                          self.d_loss,
                                                                                                          self.loss_buff,
                                                                                                          self.optimizer1_sch.get_last_lr()))
            return loss_buff, self.d_loss
        else:
            print(
                'epoch {}: loss_traning data {} loss_val data {} minimal loss {} learning_rate {}'.format(e, loss_buff,
                                                                                                          self.loss,
                                                                                                          self.loss_buff,
                                                                                                          self.optimizer1_sch.get_last_lr()))
            return loss_buff, self.loss
    def test_(self, test, args):
        self.test_data = DataLoader(dataset=test, batch_size=10, shuffle=True)
        for x_, u_ in self.test_data:
            x_ = x_.to(args['device'])
            u_ = u_.to(args['device'])
            self.pred_forward_test(x_, u_, args)


    def pred_forward(self,x,u,shift,args):
        pred_horizon = args['pred_horizon']
        x0_buff = x[:,0,:]
        x0 = self.net(x0_buff)
        x_pred_all = self.net(x)[:,1:,:]

        loss = nn.MSELoss()

        if args['extend_state']:
            x0 = torch.cat([x0_buff,x0],1)

        if args['ABCD'] == 2:
            x1_buff = x[:,1,:]
            x1 = self.net(x1_buff)
            if args['extend_state']:
                x1 = torch.cat([x1_buff,x1],1)
                u0 = u[:,0,:]
                x_all = torch.cat([x0,u0],1)
                K = torch.linalg.lstsq(x_all,x1).solution
                self.A_1 = K[:args['state_dim']+args['latent_dim'],:]
                self.B_1 = K[-args['act_dim']:,:]


        x_pred_matrix = torch.zeros_like(x[:,1:,:])
        x_pred_matrix_all = torch.zeros([x.shape[0],x.shape[1]-1,args['latent_dim']]).to(args['device'])
        x_pred_matrix_n = torch.zeros_like(x[:, 1:, :])
        x_pred_matrix_all_n = torch.zeros([x.shape[0], x.shape[1] - 1, args['latent_dim']]).to(args['device'])

        self.w_mean = torch.zeros(args['batch_size'], args['state_dim'] + args['latent_dim']).to(args['device'])
        base_distribution = dist.MultivariateNormal(torch.zeros(args['latent_dim'] + args['state_dim']),
                                                    torch.eye(args['latent_dim'] + args['state_dim']))
        self.epsilon = base_distribution.sample((args['batch_size'],)).to(args['device'])

        SCALE_DIAG_MIN_MAX = (-20, 2)
        if args['if_sigma']: # Trained with the noise characteristic network
            x0_n = x0
            # assume sigma remain the same within one forward prediction window
            log_sigma = self.noisemlp(x0_n)
            log_sigma = torch.clamp(log_sigma, min=SCALE_DIAG_MIN_MAX[0], max=SCALE_DIAG_MIN_MAX[1])
            self.sigma = torch.exp(log_sigma)
            bijector = transforms.AffineTransform(loc=self.w_mean, scale=self.sigma)
            self.w = bijector(self.epsilon)
            for i in range(pred_horizon-1):
                # self.sigma = torch.exp(log_sigma).unsqueeze(1)
                # bijector = transforms.AffineTransform(loc=self.w_mean, scale=self.sigma)
                # self.w = bijector(self.epsilon[:, :, i, :])
                # # self.w = self.w_mean + self.e_sigma * self.epsilon[:, :, i]
                # self.w = self.w.squeeze()
                x0_n = torch.matmul(x0_n,self.A_1)+torch.matmul(u[:,i,:],self.B_1) + self.w
                x_pred_n = torch.matmul(x0_n, self.C_1)
                x_pred_matrix_all_n[:,i,:] = x0_n[:, -args['latent_dim']:]
                x_pred_matrix_n[:,i,:] = x_pred_n

        for i in range(pred_horizon-1):
            x0 = torch.matmul(x0,self.A_1)+torch.matmul(u[:,i,:],self.B_1)
            x_pred = torch.matmul(x0,self.C_1)
            x_pred_matrix_all[:,i,:] = x0[:, -args['latent_dim']:]
            x_pred_matrix[:,i,:] = x_pred


        self.select = [2, 5, 8]
        d_loss, p2_loss, p3_loss = 0., 0., 0.
        if args['if_sigma']:
            self.d_loss += loss(x_pred_matrix_n[:, :, :], x[:, 1:, :]) * 10
            self.d_loss += loss(x_pred_all[:, :, :], x_pred_matrix_all_n[:, :, :])
            # -----------terminal constraints-----------##
            self.d_loss += loss(x_pred_matrix_n[:, -1, :], x[:, -1, :]) * 10
            self.d_loss += loss(x_pred_all[:, -1, :], x_pred_matrix_all_n[:, -1, :])
        else:
            self.d_loss += loss(x_pred_matrix[:, :, :], x[:, 1:, :]) * 10
            self.d_loss += loss(x_pred_all[:, :, :], x_pred_matrix_all[:, :, :])
            # -----------terminal constraints-----------##
            self.d_loss += loss(x_pred_matrix[:, -1, :], x[:, -1, :]) * 10
            self.d_loss += loss(x_pred_all[:, -1, :], x_pred_matrix_all[:, -1, :])

        if args['if_pi']:
            ##########################
            # --------------physics informed----------------- #
            system = physics(args)
            x_pred_matrix_re = x_pred_matrix_n * shift[1] + shift[0]
            u_re = u * shift[3] + shift[2]

            # ############################
            '''
            loss2 : X_k+1 - X_k = ΔX_k
            '''
            dxk_s = system.derivative(x_pred_matrix_re[:, :-1, :], u_re[:, 1:, :]) * system.h
            dxk = (dxk_s - shift[0]) / shift[1]
            pred_dxk = x_pred_matrix_n[:, 1:, :] - x_pred_matrix_n[:, :-1, :]
            # self.p2_loss += 0.5 * loss(dxk[:, :, :], pred_dxk[:, :, :])
            self.p2_loss += 0.1 * loss(dxk[:, :, self.select], pred_dxk[:, :, self.select])

            ############################
            '''
            loss3 partial: g(xk+1) = g(f(xk,uk))
            '''
            xk = torch.zeros([args['batch_size'], args['pred_horizon'] - 2, args['state_dim']]).to(args['device'])
            # dxk_s = system.derivative(x_pred_matrix_re[:, :-1, :], u_re[:, 1:, :]) * system.h
            xk[:, :, :] = x_pred_matrix_re[:, 1:, :]
            xk[:, :, self.select] = dxk_s[:, :, self.select] + x_pred_matrix_re[:, :-1, self.select]
            xk = (xk - shift[0]) / shift[1]
            gxk = self.net(xk)  # g(f(xk,uk))
            pred_gxk = x_pred_matrix_all[:, 1:, :]  # g(xk+1)
            self.p3_loss += 100 * loss(pred_gxk, gxk)

            # self adaptive loss
            self.loss = (1/pow(self.d, 2))*self.d_loss \
                        + (1/pow(self.p2, 2))*self.p2_loss\
                        + (1/pow(self.p3, 2))*self.p3_loss\
                        + 200 * (torch.log(1 + pow(self.d, 2)) + torch.log(1 + pow(self.p2, 2)) + torch.log(1 + pow(self.p3, 2)))
        else:
            self.loss = self.d_loss

        # self.displace1 = x_pred[7,:]
        # self.displace2 = x[7, i+1, :]


    def pred_forward_test(self,x,u,shift,test,args,e=0,test_save=True):
        x = x.to(args['device'])
        u = u.to(args['device'])
        self.test_loss = 0
        count = 0
        x_pred_list = []
        x_sum_list = []
        x_real_list = []
        x_time_list = []
        plt.close()
        f, axs = plt.subplots(args['state_dim'], sharex=True, figsize=(15, 15))
        time_all = torch.arange(x.shape[1])

        if test:
            # for i in range(0,args['max_ep_steps']-args['pred_horizon'],10):
            for i in range(0, args['max_ep_steps'] * args['test_steps'] - args['pred_horizon'] + 1,
                           args['pred_horizon'] - 1):
                x_pred = x[:,i:i+args['pred_horizon']]
                u_pred = u[:,i:i+args['pred_horizon']]
                x_pred_list_buff,x_real_list_buff,x_sum_list_buff,loss_test = \
                    Koopman_Desko.pred_forward_test_buff(self,x_pred,u_pred,args)
                x_pred_list.append(torch.tensor(x_pred_list_buff))
                x_real_list.append(torch.tensor(x_real_list_buff))
                x_sum_list.append(torch.tensor(x_sum_list_buff))
                x_time_list.append(torch.arange(i+1,i+args['pred_horizon']))
                self.test_loss += loss_test
                count += 1
            self.test_loss = self.test_loss/count

            print("test_loss{}".format(self.test_loss))
            ## scale back
            if (e % 50 == 0) and args['plot_test']:
                x = x * shift[1] + shift[0]
                x_pred_list = torch.stack(x_pred_list).to(args['device']) * shift[1] + shift[0]
                x=x.squeeze().cpu()
                x_pred_list=x_pred_list.cpu()

                x_pred_list = np.array(x_pred_list)
                x = np.array(x)
                x_time_list = np.array(x_time_list)

             ##------------------- plot -----------------##
                color1 = "#038355"
                font = {'family': 'Times New Roman', 'size': 12}
                titles = ['XA1', 'XB1', 'T1',
                          'XA2', 'XB2', 'T2',
                          'XA3', 'XB3', 'T3']
                plt.rc('font', **font)
                f, axs = plt.subplots(3, 3, sharex=True, figsize=(15, 9))
                legend_created = False
                for i in range(3):
                    for j in range(3):
                        axs[i, j].plot(x[:, i * 3 + j], label='Ground Truth', color='k', linewidth=2)
                        for k in range(len(x_time_list)):
                            axs[i, j].plot(x_time_list[k], x_pred_list[k][:, :, i * 3 + j], label='Koopman Model Prediction', color=color1, linewidth=2)
                            if not legend_created:
                                handles, labels = axs[i, j].get_legend_handles_labels()
                                legend_created = True
                        axs[i, j].set_title(titles[i * 3 + j])
                        axs[i, j].legend().set_visible(False)

                for ax in axs[-1, :]:
                    ax.set_xlabel('Time Steps')
                plt.tight_layout()
                f.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0.02))
                plt.subplots_adjust(bottom=0.15)
                if test_save:
                    if args['if_pi']:
                        result_type = 'pi'
                    else:
                        result_type = 'nopi'
                    plt.savefig('open_loop_result/'+str(result_type)+'/predictions_new' + str(e) + '.pdf')
                    print("plot")
                    print("save test list")
                    torch.save(x_pred_list, 'open_loop_result/'+str(result_type)+'/x_pred_list.pt')
                    torch.save(x, 'open_loop_result/'+str(result_type)+'/x.pt')
                    torch.save(x_time_list, 'open_loop_result/'+str(result_type)+'/x_time_list.pt')

            return x_pred_list,x_real_list,x_sum_list,self.test_loss


        ##----------------------------------------------##
        else:
            return Koopman_Desko.pred_forward_test_buff(self,x,u,args)
    
    def pred_forward_test_buff(self,x,u,args):
        pred_horizon = args['pred_horizon']

        self.net.eval()

        x0_buff = x[:,0,:]        
        x0 = self.net(x0_buff)


        if args['extend_state']:
            x0 = torch.cat([x0_buff,x0],1)

        if args['ABCD'] == 2:
            x1_buff = x[:,1,:]
            x1 = self.net(x1_buff)
            if args['extend_state']:
                x1 = torch.cat([x1_buff,x1],1)
                u0 = u[:,0,:]
                x_all = torch.cat([x0,u0],1)
                K = torch.linalg.lstsq(x_all,x1)
                print("try")

        x_pred_list = []
        x_sum_list = []
        x_real_list = []

        loss_test = 0
        loss = nn.MSELoss()

        x_pred_all = self.net(x[:,:-1,:])

        x_pred_matrix = torch.zeros_like(x[:,1:,:])
        for i in range(pred_horizon-1):
            x0 = torch.matmul(x0,self.A_1)+torch.matmul(u[:,i,:],self.B_1)
            x_pred = torch.matmul(x0, self.C_1)
            x_pred_matrix[:,i,:] = x_pred
            x_pred_list.append(x_pred.cpu().detach().numpy())
            x_real_list.append(x[:,i+1,:].cpu().detach().numpy())

        loss_test += loss(x_pred_matrix, x[:, 1:, :])

        self.net.train()

        return x_pred_list,x_real_list,x_sum_list,loss_test



    def parameter_store(self,args):
        #save nn
        torch.save(self.net_para,self.MODEL_SAVE)
        #save noise sigma
        torch.save(self.noise_para, self.NOISE_SAVE)
        #save A1 B1 C1
        torch.save(self.A_1_restore,self.SAVE_A1)
        torch.save(self.B_1_restore,self.SAVE_B1)
        torch.save(self.C_1_restore,self.SAVE_C1)

        torch.save(self.optimizer1.state_dict(),self.OPTI1)
        if args['ABCD'] != 2:
            torch.save(self.optimizer2.state_dict(),self.OPTI2)

        print("store!!!")

        
        
    def parameter_restore(self,args):

        # self.A_1 = torch.load(self.SAVE_A1, map_location=torch.device(args['device']))
        self.A_1 = torch.load(self.SAVE_A1, map_location='cpu')
        self.B_1 = torch.load(self.SAVE_B1, map_location='cpu')
        self.C_1 = torch.load(self.SAVE_C1, map_location='cpu')

        self.net = MLP(args)
        self.net.load_state_dict(torch.load(self.MODEL_SAVE, map_location='cpu'))
        self.net.eval()

        if args['if_sigma']:
            self.noisemlp = Noise_MLP(args)
            self.noisemlp.load_state_dict(torch.load(self.NOISE_SAVE, map_location='cpu'))
            self.noisemlp.eval()
        print("restore!")


    def set_shift_and_scale(self, replay_memory):

        self.shift = replay_memory.shift_[0]
        self.scale = replay_memory.shift_[1]
        self.shift_u = replay_memory.shift_[2]
        self.scale_u = replay_memory.shift_[3]

    def _create_optimizer(self, args):
        pass


class physics():
    def __init__(self, args):
        self.args = args
        self.h = torch.tensor(0.001).to(self.args['device'])

        self.s2hr = torch.tensor(3600).to(self.args['device'])
        self.MW = torch.tensor(250e-3).to(self.args['device'])
        self.sum_c = torch.tensor(2E3).to(self.args['device'])
        self.T10 = torch.tensor(300).to(self.args['device'])
        self.T20 = torch.tensor(300).to(self.args['device'])
        self.F10 = torch.tensor(5.04).to(self.args['device'])
        self.F20 = torch.tensor(5.04).to(self.args['device'])
        self.Fr = torch.tensor(50.4).to(self.args['device'])
        self.Fp = torch.tensor(0.504).to(self.args['device'])
        self.V1 = torch.tensor(1).to(self.args['device'])
        self.V2 = torch.tensor(0.5).to(self.args['device'])
        self.V3 = torch.tensor(1).to(self.args['device'])
        self.E1 = torch.tensor(5e4).to(self.args['device'])
        self.E2 = torch.tensor(6e4).to(self.args['device'])
        self.k1 = torch.tensor(2.77e3).to(self.args['device']) * self.s2hr
        self.k2 = torch.tensor(2.6e3).to(self.args['device']) * self.s2hr
        self.dH1 = -torch.tensor(6e4).to(self.args['device']) / self.MW
        self.dH2 = -torch.tensor(7e4).to(self.args['device']) / self.MW
        self.aA = torch.tensor(3.5).to(self.args['device'])
        self.aB = torch.tensor(1).to(self.args['device'])
        self.aC = torch.tensor(0.5).to(self.args['device'])
        self.Cp = torch.tensor(4.2e3).to(self.args['device'])
        self.R = torch.tensor(8.314).to(self.args['device'])
        self.rho = torch.tensor(1000).to(self.args['device'])
        self.xA10 = torch.tensor(1).to(self.args['device'])
        self.xB10 = torch.tensor(0).to(self.args['device'])
        self.xA20 = torch.tensor(1).to(self.args['device'])
        self.xB20 = torch.tensor(0).to(self.args['device'])
        self.Hvap1 = -torch.tensor(35.3E3).to(self.args['device']) * self.sum_c
        self.Hvap2 = -torch.tensor(15.7E3).to(self.args['device']) * self.sum_c
        self.Hvap3 = -torch.tensor(40.68E3).to(self.args['device']) * self.sum_c

        self.noise_error_std = torch.tensor([0.001, 0.001, 0.01, 0.001, 0.001, 0.01, 0.001, 0.001, 0.01]).to(
            self.args['device'])
        self.noise_error_clip = torch.tensor([0.01, 0.01, 0.1, 0.01, 0.01, 0.1, 0.01, 0.01, 0.1]).to(
            self.args['device'])
        # self.kw = torch.tensor([0.01, 0.01, 0.1, 0.01, 0.01, 0.1, 0.01, 0.01, 0.1]).to(self.args['device'])
        # self.kw = torch.tensor([0.01, 0.01, 0.1, 0.01, 0.01, 0.1, 0.01, 0.01, 0.1]).to(self.args['device'])+\
        #           torch.clamp(torch.normal(mean=0, std=self.noise_error_std), -self.noise_error_clip, self.noise_error_clip) # noise deviation
        self.kw = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(self.args['device'])+ \
                  torch.clamp(torch.normal(mean=0, std=self.noise_error_std), -self.noise_error_clip,
                              self.noise_error_clip)  # noise deviation
        self.bw = torch.tensor([0.1, 0.1, 1, 0.1, 0.1, 1, 0.1, 0.1, 1]).to(self.args['device'])  # noise bound
    def random_noise(self,x):
        noise = torch.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                process_noise = torch.normal(mean=0, std=self.kw)
                process_noise = torch.clamp(process_noise, -self.bw, self.bw)
                noise[i, j, :] = process_noise
        return noise
    def derivative(self, x, us):
        xA1 = x[:,:,0]
        xB1 = x[:,:,1]
        T1 = x[:,:,2]

        xA2 = x[:,:,3]
        xB2 = x[:,:,4]
        T2 = x[:,:,5]

        xA3 = x[:,:,6]
        xB3 = x[:,:,7]
        T3 = x[:,:,8]

        Q1 = us[:,:,0]
        Q2 = us[:,:,1]
        Q3 = us[:,:,2]

        xC3 = 1 - xA3 - xB3
        x3a = self.aA * xA3 + self.aB * xB3 + self.aC * xC3

        xAr = self.aA * xA3 / x3a
        xBr = self.aB * xB3 / x3a
        xCr = self.aC * xC3 / x3a

        F1 = self.F10 + self.Fr
        F2 = F1 + self.F20
        # F3 = F2 - self.Fr - self.Fp

        f1 = self.F10 * (self.xA10 - xA1) / self.V1 + self.Fr * (xAr - xA1) / self.V1 - self.k1 * torch.exp(
            -self.E1 / (self.R * T1)) * xA1
        f2 = self.F10 * (self.xB10 - xB1) / self.V1 + self.Fr * (xBr - xB1) / self.V1 + self.k1 * torch.exp(
            -self.E1 / (self.R * T1)) * xA1 - self.k2 * torch.exp(
            -self.E2 / (self.R * T1)) * xB1
        f3 = self.F10 * (self.T10 - T1) / self.V1 + self.Fr * (T3 - T1) / self.V1 - self.dH1 * self.k1 * torch.exp(
            -self.E1 / (self.R * T1)) * xA1 / self.Cp - self.dH2 * self.k2 * torch.exp(
            -self.E2 / (self.R * T1)) * xB1 / self.Cp + Q1 / (self.rho * self.Cp * self.V1)

        f4 = F1 * (xA1 - xA2) / self.V2 + self.F20 * (self.xA20 - xA2) / self.V2 - self.k1 * torch.exp(
            -self.E1 / (self.R * T2)) * xA2
        f5 = F1 * (xB1 - xB2) / self.V2 + self.F20 * (self.xB20 - xB2) / self.V2 + self.k1 * torch.exp(
            -self.E1 / (self.R * T2)) * xA2 - self.k2 * torch.exp(
            -self.E2 / (self.R * T2)) * xB2
        f6 = F1 * (T1 - T2) / self.V2 + self.F20 * (self.T20 - T2) / self.V2 - self.dH1 * self.k1 * torch.exp(
            -self.E1 / (self.R * T2)) * xA2 / self.Cp - self.dH2 * self.k2 * torch.exp(
            -self.E2 / (self.R * T2)) * xB2 / self.Cp + Q2 / (self.rho * self.Cp * self.V2)

        f7 = F2 * (xA2 - xA3) / self.V3 - (self.Fr + self.Fp) * (xAr - xA3) / self.V3
        f8 = F2 * (xB2 - xB3) / self.V3 - (self.Fr + self.Fp) * (xBr - xB3) / self.V3
        f9 = F2 * (T2 - T3) / self.V3 + Q3 / (self.rho * self.Cp * self.V3) + (self.Fr + self.Fp) * (
                    xAr * self.Hvap1 + xBr * self.Hvap2 + xCr * self.Hvap3) / (
                     self.rho * self.Cp * self.V3)
        F = torch.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9])
        F = F.permute(1, 2, 0)
        return F

# class encoder(nn.Module):
#     def __init__(self, input_dim, hid_dim, n_layers, dropout):
#         super().__init__()
#         self.hid_dim = hid_dim
#         self.n_layers = n_layers
#
#         self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, input):
#         outputs, (hidden, cell) = self.rnn(input)
#         return outputs ,hidden, cell

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
    def forward(self,x):
        return self.model(x)


class Noise_MLP(nn.Module):
    def __init__(self, args):
        super(Noise_MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(args['state_dim']+args['latent_dim'], 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.Linear(100, 80),
            nn.Linear(64, args['latent_dim']+args['state_dim']),
        )
    def forward(self,x):
        return self.model(x)

if __name__ == '__main__':
    pass    
