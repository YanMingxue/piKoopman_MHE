import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


class CSTR_system(gym.Env):

    def __init__(self, ref=np.array([0.18, 0.67, 480.3, 0.19, 0.65, 472.8, 0.06, 0.67, 474.9])):
        self.t = 0
        self.action_sample_period = 20  # 怎么改
        self.sampling_period = 0.01
        self.h = 0.001
        self.sampling_steps = int(self.sampling_period / self.h)
        self.delay = 5
        # self.delay = 0

        # 参数
        self.F = 4.998
        self.V = 1
        self.R = 8.314
        self.T0 = 300
        self.CA0 = 4
        self.DH1 = -5.0 * pow(10, 4)
        self.DH2 = -5.2 * pow(10, 4)
        self.DH3 = -5.4 * pow(10, 4)
        self.k10 = 3 * pow(10, 6)
        self.k20 = 3 * pow(10, 5)
        self.k30 = 3 * pow(10, 5)
        self.E1 = 5 * pow(10, 4)
        self.E2 = 7.53 * pow(10, 4)
        self.E3 = 7.53 * pow(10, 4)
        self.rho = 1000
        self.cp = 0.231
        self.w = 0.0

        self.x_init = [370, 3.41]

        self.kw = np.array([10, 0.1])  # noise deviation
        self.bw = np.array([20, 0.2])  # noise bound
        # action noise
        self.an = np.array(10)
        self.bn = np.array(1e2)

        self.xs = np.array([3.885709629082788e+02, 3.590804100931568])
        # self.us = 1.12 * np.array([2.9e9, 1.0e9, 2.9e9])

        high = np.array([1, 1])

        self.us = np.array(1e4)
        self.action_low = 0.4 * self.us
        self.action_high = 1.2 * self.us

        # 修改至 steady-state set-point
        self.reference = self.xs

        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        self.flag = True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, step, impulse=0):

        action = self.get_action(self.flag)
        # action = np.clip(action, self.action_low, self.action_high)
        x0 = self.state
        for i in range(self.sampling_steps):
            d_x = self.derivative(x0, action)
            # print(u)
            process_noise = np.random.normal(np.zeros_like(self.kw), self.kw)
            process_noise = np.clip(process_noise, -self.bw, self.bw)
            x0 = x0 + d_x * self.h + process_noise * self.h

        action = np.array(action)
        # print(action_list)
        self.state = x0
        self.t += 1

        cost = np.linalg.norm(self.state - self.reference)
        done = False
        data_collection_done = False

        if step % 10 == 0:
            self.flag = not self.flag

        return x0, action, cost, done, dict(reference=self.reference, data_collection_done=data_collection_done)

    def step_test(self, test_flag=False):

        action = self.get_action(test_flag)
        # action = np.clip(action, self.action_low, self.action_high)
        x0 = self.state
        for i in range(self.sampling_steps):
            d_x = self.derivative(x0, action)
            # print(u)
            process_noise = np.random.normal(np.zeros_like(self.kw), self.kw)
            process_noise = np.clip(process_noise, -self.bw, self.bw)
            x0 = x0 + d_x * self.h + process_noise * self.h

        action = np.array(action)
        # print(action_list)
        self.state = x0
        self.t += 1

        cost = np.linalg.norm(self.state - self.reference)
        done = False
        data_collection_done = False

        return x0, action, cost, done, dict(reference=self.reference, data_collection_done=data_collection_done)

    def reset(self):
        self.a_holder = self.action_space.sample()
        # self.state = self.x_init + np.random.normal(np.zeros_like(self.xs), self.xs * 0.001)
        self.state = self.x_init
        # self.state = self.xs
        self.t = 0
        self.time = 0
        self.flag = True
        return self.state

    def derivative(self, x, u):

        # u = self.get_action(flag)

        T = x[0]
        CA = x[1]

        f1 = (self.F / self.V) * (self.T0 - T) - self.DH1 * self.k10 * np.exp(-self.E1 / (self.R * T)) * CA / (
                self.rho * self.cp) - self.DH2 * self.k20 * np.exp(
            -self.E2 / (self.R * T)) * CA / (self.rho * self.cp) - self.DH3 * self.k30 * np.exp(
            -self.E3 / (self.R * T)) * CA / (self.rho * self.cp)
        f2 = (self.F / self.V) * (self.CA0 - CA) - self.k10 * np.exp(-self.E1 / (self.R * T)) * x[
            1] - self.k20 * np.exp(-self.E2 / (self.R * T)) * CA \
             - self.k30 * np.exp(-self.E3 / (self.R * T)) * CA
        d1 = f1 + u / (self.rho * self.cp * self.V)
        d2 = f2 + (self.F / self.V) * self.w

        F = np.array([d1, d2])

        return F

    def render(self, mode='human'):
        return

    def get_action(self, flag):
        # a = -5000 * (x[0] - self.xs[0])
        if flag:
            a = 1e4
        else:
            a = 5e3

        action_noise = np.random.normal(np.zeros_like(self.an), self.an)
        action_noise = np.clip(action_noise, -self.bn, self.bn)

        a = a + action_noise

        return a

    # def get_noise(self):
    #     scale = 0.1 * self.xs
    #     return np.random.normal(np.zeros_like(self.xs), scale)


if __name__ == '__main__':

    env = CSTR_system()
    T = 300
    path = []
    a_path = []
    t1 = []
    s = env.reset()
    for i in range(int(T)):
        # action = env.us
        s, a, r, done, info = env.step(i)
        path.append(s)
        a_path.append(a)
        t1.append(i)
    path = np.array(path)

    # 调整action维度
    a_path = np.array(a_path)
    a_path = a_path.flatten()
    # print(a_path)
    state_dim = s.shape[0]
    fig, ax = plt.subplots(state_dim, sharex=True, figsize=(15, 15))
    t = range(T)
    for i in range(state_dim):
        ax[i].plot(t, path[:, i], color='red')

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(t1, a_path)
    # ax.plot(a_path)
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.show()
    print('done')
