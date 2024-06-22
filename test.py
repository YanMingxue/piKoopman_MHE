import torch
import torch.nn as nn
import torch.distributions as dist
import my_args
import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
from three_tanks import three_tank_system as dreamer
from replay_memory import ReplayMemory
import my_args
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


args = my_args.args

x_true_list = torch.load("result/x_true.pt")
x_hat_list = torch.load("result/x_hat.pt")
x_hat_nopi_list = torch.load("result/x_sigma_hat_nopi.pt")


print("plot....")
shift = torch.load("save_model/shift.pt")
# else:
#     shift = shift
x_true_list = [x * shift[1].cpu().numpy() + shift[0].cpu().numpy() for x in x_true_list]
x_est_list = [x * shift[1].cpu().numpy() + shift[0].cpu().numpy() for x in x_hat_list]
x_est_nopi_list = [x * shift[1].cpu().numpy() + shift[0].cpu().numpy() for x in x_hat_nopi_list]
x_true_array = np.array(x_true_list)
x_est_array = np.array(x_est_list)
x_est_nopi_array = np.array(x_est_nopi_list)

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
        axs[i, j].plot(x_est_nopi_array[:, i * 3 + j], '-.', label='x_without pi', color=color2, linewidth=2)
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

#
# base_distribution = dist.MultivariateNormal(torch.zeros(args['latent_dim']+args['state_dim']), torch.eye(args['latent_dim']+args['state_dim']))
# epsilon = base_distribution.sample((args['batch_size'], 1, args['pred_horizon']))
# print("epsilon shape:", epsilon.shape)

#
# # gpu test
# import torch
# print(torch.__version__)
# print(torch.version.cuda)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
#
# class AffineBijector(nn.Module):
#     def __init__(self, d_mean, sigma):
#         super(AffineBijector, self).__init__()
#         self.d_mean = nn.Parameter(torch.tensor(d_mean, dtype=torch.float32))
#         self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32))
#     def forward(self, epsilon):
#         d = self.d_mean + self.sigma * epsilon
#         return d
#
# t = 1
# bijector = AffineBijector(torch.zeros(args['state_dim']+args['latent_dim']), torch.ones(args['state_dim']+args['latent_dim']))
# d = bijector(epsilon[:, :, t])
# print(d.shape)
# # 重塑为形状 [-1, self.latent_dim]
# d = d.squeeze()
# print(d.shape)
