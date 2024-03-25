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
loss_curves1 = torch.load("save_model/test_curve_pi.pt")
# 提取每个loss曲线的值
values1 = [torch.tensor([tensor.item() for tensor in curve]) for curve in loss_curves1]
# 将值转换为NumPy数组
values_np1 = [curve.numpy() for curve in values1]
# 计算每一列的均值
mean_values1 = np.mean(values_np1, axis=0)
max_values1 = np.max(values_np1, axis=0)
min_values1 = np.min(values_np1, axis=0)

# load fullpi
loss_curves2 = torch.load("save_model/test_curve.pt")
# 提取每个loss曲线的值
values2 = [torch.tensor([tensor.item() for tensor in curve]) for curve in loss_curves2]
# 将值转换为NumPy数组
values_np2 = [curve.numpy() for curve in values2]
# 计算每一列的均值
mean_values2 = np.mean(values_np2, axis=0)
max_values2 = np.max(values_np2, axis=0)
min_values2 = np.min(values_np2, axis=0)
#
# #load fullpi with noise model
# loss_curves2_1 = torch.load("save_model/test_curve2_1.pt")
# # 提取每个loss曲线的值
# values2_1 = [torch.tensor([tensor.item() for tensor in curve]) for curve in loss_curves2_1]
# # 将值转换为NumPy数组
# values_np2_1 = [curve.numpy() for curve in values2_1]
# # 计算每一列的均值
# mean_values2_1 = np.mean(values_np2_1, axis=0)
# max_values2_1 = np.max(values_np2_1, axis=0)
# min_values2_1 = np.min(values_np2_1, axis=0)

# #load partial pi 258
# loss_curves3 = torch.load("save_model/test_curve3.pt")
# # 提取每个loss曲线的值
# values3 = [torch.tensor([tensor.item() for tensor in curve]) for curve in loss_curves3]
# # 将值转换为NumPy数组
# values_np3 = [curve.numpy() for curve in values3]
# # 计算每一列的均值
# mean_values3 = np.mean(values_np3, axis=0)
# max_values3 = np.max(values_np3, axis=0)
# min_values3 = np.min(values_np3, axis=0)
#
# #load partial pi 013467
# loss_curves4 = torch.load("save_model/test_curve4.pt")
# # 提取每个loss曲线的值
# values4 = [torch.tensor([tensor.item() for tensor in curve]) for curve in loss_curves4]
# # 将值转换为NumPy数组
# values_np4 = [curve.numpy() for curve in values4]
# # 计算每一列的均值
# mean_values4 = np.mean(values_np4, axis=0)
# max_values4 = np.max(values_np4, axis=0)
# min_values4 = np.min(values_np4, axis=0)

plt.clf()
fig = plt.figure(figsize=(8, 6))

plt.fill_between(range(len(max_values1[5:])), min_values1[5:], max_values1[5:], alpha=0.2, color='blue')
plt.fill_between(range(len(max_values2[5:])), min_values2[5:], max_values2[5:], alpha=0.2, color='red')
# plt.fill_between(range(len(max_values2_1)), min_values2_1, max_values2_1, alpha=0.2, color='yellow')
# plt.fill_between(range(len(max_values4)), min_values2, max_values4, alpha=0.2, color='yellow')
# plt.fill_between(range(len(max_values3)), min_values2, max_values3, alpha=0.2, color='yellow')
# # 绘制每条loss曲线的透明阴影区域
# for curve in values_np:
#     plt.fill_between(range(len(curve)), curve, 0, alpha=0.2, color='blue')  # 在这里设置y1为0
# 绘制均值曲线

plt.plot(mean_values1[5:], label='PI Mean Curve', color='blue', linewidth=2)
plt.plot(mean_values2[5:], label='Desko Mean Curve', color='red', linewidth=2)
# plt.plot(mean_values2_1, label='Full PI with noise distriution Mean Curve', color='yellow', linewidth=2)
# plt.plot(mean_values4, label='Partial PI Mean Curve', color='yellow', linewidth=2)
# plt.plot(mean_values3, label='Partial PI(258) Mean Curve', color='yellow', linewidth=2)
plt.title('dataset size: 1000')
# plt.title('PI Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("result/2.pdf")
plt.savefig("result/2.jpg")
plt.show()
