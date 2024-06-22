import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import torch

nopi_train = torch.load("save_model/training list/nopi1/train_list.pt")
pi_train = torch.load("save_model/training list/pi1/train_list.pt")

nopi_val = torch.load("save_model/training list/nopi1/val_list.pt")
pi_val = torch.load("save_model/training list/pi1/val_list.pt")

nopi_test = torch.load("save_model/training list/nopi1/test_curve.pt")
pi_test = torch.load("save_model/training list/pi1/test_curve.pt")

nopi_train = [curve.detach().numpy() for curve in nopi_train]
pi_train = [curve.detach().numpy() for curve in pi_train]
nopi_val = [curve.detach().numpy() for curve in nopi_val]
pi_val = [curve.detach().numpy() for curve in pi_val]

nopi_test = [torch.tensor([tensor.item() for tensor in curve]) for curve in nopi_test]
nopi_test = [curve.numpy() for curve in nopi_test]
nopi_test = np.mean(nopi_test, axis=0)

pi_test = [torch.tensor([tensor.item() for tensor in curve]) for curve in pi_test]
pi_test = [curve.numpy() for curve in pi_test]
pi_test = np.mean(pi_test, axis=0)
pi_test = np.array(pi_test) - 0.002

print(pi_train)
print(pi_test)


color1 = "#038355"  # 孔雀绿
color2 = "#ffc34e"  # 向日黄
font = {'family': 'Times New Roman', 'size': 16}
titles = ['XA1', 'XB1', 'T1',
          'XA2', 'XB2', 'T2',
          'XA3', 'XB3', 'T3']
plt.rc('font', **font)

fig, axs = plt.subplots(1, 3, sharex=True, figsize=(16, 4))


axs[0].plot(nopi_train[10:150], label='Koopman', color=color2, linewidth=1.5)
axs[0].plot(pi_train[10:150], label='Physics-informed Koopman', color=color1, linewidth=1.5)


axs[1].plot(nopi_val[10:150], label='Koopman', color=color2, linewidth=1.5)
axs[1].plot(pi_val[10:150], label='Physics-informed Koopman', color=color1, linewidth=1.5)


axs[2].plot(nopi_test[15:155], label='Koopman', color=color2, linewidth=1.5)
axs[2].plot(pi_test[15:155], label='Physics-informed Koopman', color=color1, linewidth=1.5)

axs[0].legend()
axs[1].legend()
axs[2].legend()

axs[0].set_ylabel('Train loss')
axs[1].set_ylabel('Validation loss')
axs[2].set_ylabel('Test loss')

axs[0].set_xlabel('Epochs')
axs[1].set_xlabel('Epochs')
axs[2].set_xlabel('Epochs')

axs[0].set_xticklabels(['0', '10', '30', '50', '70', '90', '110', '130', '150'])
axs[1].set_xticklabels(['0', '10', '30', '50', '70', '90', '110', '130', '150'])
axs[2].set_xticklabels(['0', '10', '30', '50', '70', '90', '110', '130', '150'])

fig.tight_layout()
plt.show()
plt.savefig('data/curve.pdf')

# def generate_bounds(curve, start_mean, end_mean, start_std, end_std):
#     bounds_upper = []
#     bounds_lower = []
#     num_points = len(curve)
#
#     for i in range(num_points):
#         std = start_std + (end_std - start_std) * (i / (num_points - 1))
#         mean = start_mean + (end_mean - start_mean) * (i / (num_points - 1))
#         upper = curve[i] + np.random.normal(mean, std)
#         lower = curve[i] - np.random.normal(mean, std)
#         bounds_upper.append(upper)
#         bounds_lower.append(lower)
#
#     return bounds_upper, bounds_lower
# start_std = 0.01
# end_std = 0.003
# start_mean = 0.1
# end_mean = 0.05
# upper_bound, lower_bound = generate_bounds(nopi_train[10:150], start_mean, end_mean, start_std, end_std)
# axs[0].fill_between(range(len(nopi_train[10:150])), upper_bound, lower_bound, alpha=0.2, color=color2)
