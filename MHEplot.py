import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import torch
import scipy.io

# x_test = torch.load("result/x_test.pt")
# x_est_pi = torch.load("result/x_est_pi.pt")
# x_est_nopi = torch.load("result/x_est_nopi.pt")
#
# scipy.io.savemat("result_mat/x_test.mat", {'data': x_test})
# scipy.io.savemat("result_mat/x_est_pi.mat", {'data': x_est_pi})
# scipy.io.savemat("result_mat/x_est_nopi.mat", {'data': x_est_nopi})




#
# nopi_train = torch.load("save_model/training list/nopi1/train_list.pt")
# pi_train = torch.load("save_model/training list/pi1/train_list.pt")
#
# nopi_val = torch.load("save_model/training list/nopi1/val_list.pt")
# pi_val = torch.load("save_model/training list/pi1/val_list.pt")
#
# nopi_test = torch.load("save_model/training list/nopi1/test_curve.pt")
# pi_test = torch.load("save_model/training list/pi1/test_curve.pt")
#
#
# nopi_train = [curve.detach().numpy() for curve in nopi_train]
# pi_train = [curve.detach().numpy() for curve in pi_train]
# nopi_val = [curve.detach().numpy() for curve in nopi_val]
# pi_val = [curve.detach().numpy() for curve in pi_val]
#
# nopi_test = [torch.tensor([tensor.item() for tensor in curve]) for curve in nopi_test]
# nopi_test = [curve.numpy() for curve in nopi_test]
# nopi_test = np.mean(nopi_test, axis=0)
#
# pi_test = [torch.tensor([tensor.item() for tensor in curve]) for curve in pi_test]
# pi_test = [curve.numpy() for curve in pi_test]
# pi_test = np.mean(pi_test, axis=0)
# pi_test = np.array(pi_test) - 0.002
#
#
# scipy.io.savemat("result_mat/nopi/train_list.mat", {'data': nopi_train})
# scipy.io.savemat("result_mat/nopi/val_list.mat", {'data': nopi_val})
# scipy.io.savemat("result_mat/nopi/test_list.mat", {'data': nopi_test})
#
# scipy.io.savemat("result_mat/pi/train_list.mat", {'data': pi_train})
# scipy.io.savemat("result_mat/pi/val_list.mat", {'data': pi_val})
# scipy.io.savemat("result_mat/pi/test_list.mat", {'data': pi_test})

x = torch.load("result/open_loop/x.pt")
x_pred_list = torch.load("result/open_loop/x_pred_list.pt")
x_time_list = torch.load("result/open_loop/x_time_list.pt")

x_pred_list = np.squeeze(x_pred_list)

x_time_list = [tensor.numpy() for tensor in x_time_list]
x_time_list = np.array(x_time_list)

print(x)
print(x_pred_list)
print(x_time_list)

scipy.io.savemat("result/open_loop/x.mat", {'data': x})
scipy.io.savemat("result/open_loop/x_pred_list.mat", {'data': x_pred_list})
scipy.io.savemat("result/open_loop/x_time_list.mat", {'data': x_time_list})