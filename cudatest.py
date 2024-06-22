# import numpy as np
# import torch
# total = 0
# error = torch.load("save_model/test_error.pt")
# print(error)

# tensor_values = [tensor.item() for tensor in error]
# tensor_array = np.array(tensor_values)
# max = np.max(tensor_array)
# min = np.min(tensor_array)
# variance = np.var(tensor_array)
# average = np.average(tensor_array)
#
# print("Variance:", variance)    # 0.0005471134287295147
# print("average:", average)    # 0.020807128492742778
# print("max:", max)
# print("min:", min)

#   Desko:
#   variance: 0.0005471134287295147
#   average: 0.020807128492742778
#   max: 0.10303998738527298
#   min: 0.003967613913118839

#   Full PI:
#   variance: 0.0002744437031634215
#   average: 0.015183908315375447
#   max: 0.10005432367324829
#   min: 0.004761493299156427


import matplotlib.pyplot as plt
import numpy as np
import torch

# load desko
loss_curves1 = torch.load("save_model_5_7/nopi_0.817_0.041/test_curve.pt")
# 提取每个loss曲线的值
values1 = [torch.tensor([tensor.item() for tensor in curve]) for curve in loss_curves1]
# 将值转换为NumPy数组
values_np1 = [curve.numpy() for curve in values1]
# 计算每一列的均值
mean_values1 = np.mean(values_np1, axis=0)
max_values1 = np.max(values_np1, axis=0)
min_values1 = np.min(values_np1, axis=0)
#
# # load fullpi
loss_curves2 = torch.load("save_model_5_7/withpi_0.612_0.0364/test_curve.pt")
# 提取每个loss曲线的值
values2 = [torch.tensor([tensor.item() for tensor in curve]) for curve in loss_curves2]
# 将值转换为NumPy数组
values_np2 = [curve.numpy() for curve in values2]
# 计算每一列的均值
mean_values2 = np.mean(values_np2, axis=0)
max_values2 = np.max(values_np2, axis=0)
min_values2 = np.min(values_np2, axis=0)
# #
# # #load fullpi with noise model
# # loss_curves2_1 = torch.load("save_model/test_curve2_1.pt")
# # # 提取每个loss曲线的值
# # values2_1 = [torch.tensor([tensor.item() for tensor in curve]) for curve in loss_curves2_1]
# # # 将值转换为NumPy数组
# # values_np2_1 = [curve.numpy() for curve in values2_1]
# # # 计算每一列的均值
# # mean_values2_1 = np.mean(values_np2_1, axis=0)
# # max_values2_1 = np.max(values_np2_1, axis=0)
# # min_values2_1 = np.min(values_np2_1, axis=0)
#
# # #load partial pi 258
# # loss_curves3 = torch.load("save_model/test_curve3.pt")
# # # 提取每个loss曲线的值
# # values3 = [torch.tensor([tensor.item() for tensor in curve]) for curve in loss_curves3]
# # # 将值转换为NumPy数组
# # values_np3 = [curve.numpy() for curve in values3]
# # # 计算每一列的均值
# # mean_values3 = np.mean(values_np3, axis=0)
# # max_values3 = np.max(values_np3, axis=0)
# # min_values3 = np.min(values_np3, axis=0)
# #
# # #load partial pi 013467
# # loss_curves4 = torch.load("save_model/test_curve4.pt")
# # # 提取每个loss曲线的值
# # values4 = [torch.tensor([tensor.item() for tensor in curve]) for curve in loss_curves4]
# # # 将值转换为NumPy数组
# # values_np4 = [curve.numpy() for curve in values4]
# # # 计算每一列的均值
# # mean_values4 = np.mean(values_np4, axis=0)
# # max_values4 = np.max(values_np4, axis=0)
# # min_values4 = np.min(values_np4, axis=0)
#
color1 = "#038355"  # 孔雀绿
color2 = "#ffc34e"  # 向日黄
font = {'family': 'Times New Roman', 'size': 12}
plt.rc('font', **font)
fig = plt.figure(figsize=(7, 5))

plt.fill_between(range(len(max_values1[5:])), min_values1[5:], max_values1[5:], alpha=0.2, color='')
plt.fill_between(range(len(max_values2[5:])), min_values2[5:], max_values2[5:], alpha=0.2, color='red')
# plt.fill_between(range(len(max_values2_1)), min_values2_1, max_values2_1, alpha=0.2, color='yellow')
# plt.fill_between(range(len(max_values4)), min_values2, max_values4, alpha=0.2, color='yellow')
# plt.fill_between(range(len(max_values3)), min_values2, max_values3, alpha=0.2, color='yellow')
# # 绘制每条loss曲线的透明阴影区域
# for curve in values_np:
#     plt.fill_between(range(len(curve)), curve, 0, alpha=0.2, color='blue')  # 在这里设置y1为0
# 绘制均值曲线

plt.plot(mean_values1[5:], label='Koopman', color=color2, linewidth=2)
plt.plot(mean_values2[5:], label='Physics-informed Koopman', color=color1, linewidth=2)
# plt.plot(mean_values2_1, label='Full PI with noise distriution Mean Curve', color='yellow', linewidth=2)
# plt.plot(mean_values4, label='Partial PI Mean Curve', color='yellow', linewidth=2)
# plt.plot(mean_values3, label='Partial PI(258) Mean Curve', color='yellow', linewidth=2)

# plt.title('PI Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("result/2.pdf")
plt.savefig("result/2.jpg")
plt.show()


x_true_list = torch.load("result/x_true.pt")
x_hat_list = torch.load("result/x_hat.pt")
x_hat_nopi_list = torch.load("result/x_sigma_hat_nopi.pt")


print("plot....")
shift = torch.load(args['SAVE_SHIFT'])
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
