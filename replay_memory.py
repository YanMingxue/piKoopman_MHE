import math
import numpy as np
import random
import progressbar
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch


class MyDataSet(Dataset):

    def __init__(self, test, x = None, y = None):
        self.x = x
        self.u = y
        self.test = test



    def __len__(self):
        return len(self.x_choice)

    def __getitem__(self, index):
        return self.x_choice[index, :], self.y_choice[index]
    
    def determine_shift_and_scale(self, args):
        self.shift_x = torch.mean(self.x, axis=(0, 1)).to(args['device'])
        self.scale_x = torch.std(self.x, axis=(0, 1)).to(args['device'])
        self.shift_u = torch.mean(self.u, axis=(0, 1)).to(args['device'])
        self.scale_u = torch.std(self.u, axis=(0, 1)).to(args['device'])

        return [self.shift_x,self.scale_x,self.shift_u,self.scale_u]
    
    def shift_scale(self, shift_ = None):

        if self.test:
            self.x_choice = (self.x - shift_[0])/shift_[1]
            self.y_choice = (self.u - shift_[2])/shift_[3]
        else:
            self.x_choice = (self.x - self.shift_x)/self.scale_x
            self.y_choice = (self.u - self.shift_u)/self.scale_u
    
    



# Class to load and preprocess data
class ReplayMemory():
    def __init__(self, args, env, predict_evolution = False, LSTM = False):
        """Constructs object to hold and update training/validation data.
        Args:
            args: Various arguments and specifications
            shift: Shift of state values for normalization
            scale: Scaling of state values for normalization
            shift_u: Shift of action values for normalization
            scale_u: Scaling of action values for normalization
            env: Simulation environment
            net: Neural network dynamics model
            sess: TensorFlow session
            predict_evolution: Whether to predict how system will evolve in time
        """
        self.batch_size = args['batch_size']
        self.seq_length = args['pred_horizon']
        # self.shift_x = shift
        # self.scale_x = scale
        # self.shift_u = shift_u
        # self.scale_u = scale_u
        self.env = env
        self.total_steps = 0
        self.LSTM = LSTM

        # print('validation fraction: ', args['val_frac'])


        if args['import_saved_data'] or args['continue_data_collection']:
            self._restore_data('./data/' + args['env_name'])
            # self._process_data(args)

            # print('creating splits...')
            # self._create_split(args)
            # self._determine_shift_and_scale(args)
        else:
            print("generating data...")
            self._generate_data(args)

    def _generate_data(self, args):
        """Load data from environment
        Args:
            args: Various arguments and specifications
        """

        # Initialize array to hold states and actions
        x = []
        u = []

        # Define progress bar
        bar = progressbar.ProgressBar(maxval=args['total_data_size']).start()
        length_list = []
        done_list = []
        """
            Loop through episodes. In the offline dataset, we collect data trajectories with
            different initial state values and for each start point we run open-loop to get
            a trajectory. 
            In this case, we can cover as wider range of the value of state space as possible.
        """
        while True:
            x_trial = torch.zeros((args['max_ep_steps'], args['state_dim']), dtype=torch.float32)
            x_trial[0] = self.env.reset()
            self.action = self.env.get_action(args['max_ep_steps'] - 1)
            for t in range(1, args['max_ep_steps']):
                step_info = self.env.step(t, self.action)
                # print(u_trial.shape)
                # print(step_info[1].shape)
                # print(np.squeeze(step_info[0]).shape)
                # u_trial[t - 1] = step_info[1]
                x_trial[t] = torch.squeeze(step_info[0])

                if step_info[3]['data_collection_done']:
                    break

            done_list.append(step_info[3]['data_collection_done'])
            length_list.append(t)
            j = 0
            while j + self.seq_length < len(x_trial):
                x.append(x_trial[j:j + self.seq_length])
                u.append(self.action[j:j + self.seq_length - 1])
                j += 1

            if len(x) >= args['total_data_size']:
                break
            bar.update(len(x))
        bar.finish()


        # Generate test scenario
        self.x_test = []
        self.x_test.append(self.env.reset())
        action = self.env.get_action(args['max_ep_steps']*args['test_steps'])
        self.u_test = action.to(args['device'])
        for t in range(1, args['max_ep_steps']*args['test_steps']):
            step_info = self.env.step(t, self.u_test)
            self.x_test.append(np.squeeze(step_info[0]))
            if step_info[3]['data_collection_done']:
                break

        x = torch.stack(x).to(args['device'])
        u = torch.stack(u).to(args['device']).float()
        self.x = x.reshape(-1, self.seq_length, args['state_dim']).to(args['device'])
        self.u = u.reshape(-1, self.seq_length-1, args['act_dim']).to(args['device'])
        len_x = int(np.floor(len(self.x)/args['batch_size'])*args['batch_size'])
        self.x = self.x[:len_x]
        self.u = self.u[:len_x]

        self.dataset_train = MyDataSet(test = False, x = self.x, y = self.u)
        self.shift_ = self.dataset_train.determine_shift_and_scale(args)
        self.dataset_train.shift_scale()

        x = torch.stack(self.x_test).to(args['device'])
        u = self.u_test[:-1, :]

        # Reshape and trim data sets
        self.x_test = x.reshape(-1, x.shape[0], args['state_dim']).to(args['device'])
        self.u_test = u.reshape(-1, x.shape[0]-1, args['act_dim']).to(args['device'])

        self.dataset_test = MyDataSet(test = True, x = self.x_test, y = self.u_test)
        self.dataset_test.shift_scale(self.shift_)

        len_train = len(self.dataset_train)
        len_val = int(np.round(len_train*args['val_frac']))
        len_train -= len_val
        self.train_subset, self.val_subset = random_split(self.dataset_train,[len_train, len_val],generator=torch.Generator().manual_seed(1))


        torch.save(self.dataset_test,args['SAVE_TEST'])
        torch.save(self.x_test,args['SAVE_TEST_X'])
        torch.save(self.u_test, args['SAVE_TEST_U'])
        torch.save(self.train_subset,args['SAVE_TRAIN'])
        torch.save(self.val_subset,args['SAVE_VAL'])

        print("save_test_train_dataset！")

    def _store_test(self):
        pass

    def update_data(self, x_new, u_new, val_frac):
        """Update training/validation data
        TODO:
        Args:
            x_new: New state values
            u_new: New control inputs
            val_frac: Fraction of new data to include in validation set
        """
        pass

    def save_data(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(path + '/x.pt', self.x)
        torch.save(path + '/u.pt', self.u)
        torch.save(path + '/x_test.pt', self.x_test)
        torch.save(path + '/u_test.pt', self.u_test)
        torch.save(path + '/x_val.pt', self.x_val)
        torch.save(path + '/u_val.pt', self.u_val)

    def _restore_data(self, path):
        self.x = torch.load(path + '/x.pt')
        self.u = torch.load(path + '/u.pt')
        self.x_val = torch.load(path + '/x_val.pt')
        self.u_val = torch.load(path + '/u_val.pt')
        self.x_test = torch.load(path + '/x_test.pt')
        self.u_test = torch.load(path + '/u_test.pt')

