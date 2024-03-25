import torch
from three_tanks import three_tank_system as dreamer
from torch.utils.data import DataLoader
import my_args
import numpy as np
from three_tanks import three_tank_system

T = 20.  # Time horizon
N = 20  # number of control intervals


def main():
    args = my_args.args
    args['act_expand'] = 1
    args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device is : ", args['device'])
    env = dreamer(args)
    args['state_dim'] = env.observation_space.shape[0]
    args['act_dim'] = env.action_space.shape[0]
    args['act_expand'] = args['act_expand'] * env.action_space.shape[0]
    args['reference'] = env.xs
    restore_trajectory(env, args)


def restore_trajectory(env, args):
    if not args['if_mix']:
        from Desko import Koopman_Desko
    else:
        from mix_u import Koopman_Desko
    model = Koopman_Desko(args)
    model.parameter_restore(args)

    x_train = torch.load(args['SAVE_TRAIN'])
    # x_val = torch.load(args['SAVE_VAL'])
    dataset_test = torch.load(args['SAVE_TEST'])
    shift = torch.load(args['SAVE_SHIFT'])


    train_data = DataLoader(dataset=x_train, batch_size=args['batch_size'], shuffle=True, drop_last=True)
    test_data = DataLoader(dataset=dataset_test, batch_size=1, shuffle=True, drop_last=True)

    for x_, u_ in train_data:
        x_ = x_.float().to(args['device'])
        u_ = u_.float().to(args['device'])
        comput_uncertainty(x_, u_, model, args)
        break



def comput_uncertainty(x, u, model, args):
    model.net.eval()
    g = model.net(x)[:, :, :]
    # [batch_size, horizon, state_dim+latent_dim]
    g_all = torch.cat([x, g], 2)
    # x_pred = torch.matmul(g_all, model.C_1[-args['latent_dim']:, :])
    x_pred = torch.matmul(g_all, model.C_1)
    error = x-x_pred

    g_all = g_all.reshape(-1, g_all.size(2))
    error = error.reshape(-1, error.size(2))

    dc = np.linalg.lstsq(g_all.detach().numpy(),error.detach().numpy(),rcond=None)[0]
    dc = torch.tensor(dc)
    # print(dc)

    sys = three_tank_system(args)



if __name__ == '__main__':
    main()