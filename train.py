import numpy as np
import torch
from replay_memory import ReplayMemory
# from variant import *
from three_tanks import three_tank_system as dreamer
# from CSTR import CSTR_system as dreamer
from torch.utils.data import Dataset, DataLoader, random_split
import my_args
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')


def main():
    test_mse = []
    test_curve_total = []
    for i in range(1):
        print("Process number: {}".format(i))
        args = my_args.args
        args['act_expand'] = 1
        args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device is : ", args['device'])
        # args = VARIANT
        env = dreamer(args)
        print("Load environment...")
        env = env.unwrapped
        args['state_dim'] = env.observation_space.shape[0]
        args['act_dim'] = env.action_space.shape[0]
        args['act_expand'] = args['act_expand'] * env.action_space.shape[0]
        test_loss, test_curve = train(args, env)

        test_mse.append(test_loss)
        test_curve_total.append(test_curve)
        torch.save(test_mse, "save_model/test_error.pt")
        torch.save(test_curve_total, "save_model/test_curve.pt")

        print("test_error:", test_mse)
        clear()
    print(test_curve_total)
    print("done!")


def train(args, env):
    test_curve = []
    test_mse = 1e9
    if not args['if_mix']:
        from Desko import Koopman_Desko
    else:
        from mix_u import Koopman_Desko
    model = Koopman_Desko(args)
    # model.parameter_restore(args)
    if args['reload_data'] == False:
        replay_memory = ReplayMemory(args, env, predict_evolution=True)
    # model.set_shift_and_scale(replay_memory)

    #############################00000000000#########################
    if args['reload_data'] == True:
        print('Reload dataset...')
        x_train = torch.load(args['SAVE_TRAIN'])
        x_val = torch.load(args['SAVE_VAL'])
        dataset_test = torch.load(args['SAVE_TEST'])
        shift = torch.load(args['SAVE_SHIFT'])
    else:
        x_train = replay_memory.dataset_train
        shift = replay_memory.shift_
        torch.save(shift, args['SAVE_SHIFT'])
        print("save shift!")
        x_val = replay_memory.val_subset

    ##-----------continue train------------##
    args['restore'] = False
    if args['restore'] == True:
        model.parameter_restore(args)

    if args['reload_data'] == True:
        test_data = DataLoader(dataset=dataset_test, batch_size=1, shuffle=True, drop_last=True)
    else:
        test_data = DataLoader(dataset=replay_memory.dataset_test, batch_size=1, shuffle=True, drop_last=True)

    for e in range(args['num_epochs']):
        model.learn(e, x_train, x_val, shift, args)
        if (e % 10 == 0):
            model.parameter_store(args)
            if args['if_pi']:
                print("d={},p2={},p3={}".format(model.d, model.p2, model.p3))
        if (e % 1 == 0):
            for x, u in test_data:
                _, _, _, test_loss = model.pred_forward_test(x.float(), u.float(), shift, True, args, e)
        test_curve.append(test_loss)
        if test_loss < test_mse:
            test_mse = test_loss

    # plt.clf()
    # values = [tensor.item() for tensor in test_curve]
    # fig = plt.figure(figsize=(8, 4))
    # plt.plot(values)
    # plt.title('Test Error Curve')
    # plt.ylabel('MSE')
    # plt.show()

    return test_mse, test_curve

def clear():
    for key, value in globals().items():
        if key in ["test_mse","test_curve_total"] or callable(value) or value.__class__.__name__ == "module":
            continue
        globals()[key] = None

if __name__ == '__main__':
    main()