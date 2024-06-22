from replay_memory import ReplayMemory
# from variant import *

from three_tanks import three_tank_system as dreamer
from torch.utils.data import Dataset, DataLoader, random_split
import my_args
import torch

# from controller import MPC as build_func
from robustness_eval import *


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
    restore_data(env, args)


def restore_data(env, args):
    args['if_mix'] = True
    if not args['if_mix']:
        from Desko import Koopman_Desko
        from controller import MPC as build_func
        model = Koopman_Desko(args)

    else:
        from Desko import Koopman_Desko
        from controller import Upper_MPC as build_func
        model = Koopman_Desko(args)


    # restore variables
    shift = torch.load(args['SAVE_SHIFT'])
    model.parameter_restore(args)

    args['state_dim'] = env.observation_space.shape[0]
    args['act_dim'] = env.action_space.shape[0]
    args['s_bound_low'] = env.observation_space.low
    args['s_bound_high'] = env.observation_space.high
    args['a_bound_low'] = env.action_space.low
    args['a_bound_high'] = env.action_space.high

    # args['reference'] = env.xs

    # model.shift_ = np.loadtxt(args['shift_args'])

    # model.shift_ = [model.shift, model.scale, model.shift_u, model.scale_u]

    # args['reference'] = (env.xs - model.shift) / model.scale
    args['reference'] = (env.xs - shift[0]) / shift[1]
    args['reference_'] = env.xs

    controller = build_func(model, shift, args)
    # controller._build_controller()
    # controller.check_controllability()

    dynamic(controller, env, args, args)
    # _, paths = evaluation(args, env, controller)

    # for e in range(args['max_ep_steps']):
    #     print("?")


if __name__ == '__main__':
    main()


