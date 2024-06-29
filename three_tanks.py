#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File : three_tanks.py
@Author : Yan Mingxue
@Software : PyCharm
"""

import math
import torch
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
from input import generate_input, generate_testinput
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


class three_tank_system(gym.Env):

    def __init__(self, args, ref=np.array([0.18, 0.67, 480.3, 0.19, 0.65, 472.8, 0.06, 0.67, 474.9])):
        self.args = args
        self.t = 0
        self.action_sample_period = 20
        self.sampling_period = 0.005
        self.h = torch.tensor(0.001).to(self.args['device'])
        self.sampling_steps = int(self.sampling_period / self.h)
        self.delay = 5
        # self.delay = 0

        self.s2hr = torch.tensor(3600).to(self.args['device'])
        self.MW = torch.tensor(250e-3).to(self.args['device'])
        self.sum_c = torch.tensor(2E3).to(self.args['device'])
        self.T10 = torch.tensor(300).to(self.args['device'])
        self.T20 = torch.tensor(300).to(self.args['device'])
        self.F10 = torch.tensor(5.04).to(self.args['device'])
        self.F20 = torch.tensor(5.04).to(self.args['device'])
        self.Fr = torch.tensor(50.4).to(self.args['device'])
        self.Fp = torch.tensor(0.504).to(self.args['device'])
        self.V1 = torch.tensor(1).to(self.args['device'])
        self.V2 = torch.tensor(0.5).to(self.args['device'])
        self.V3 = torch.tensor(1).to(self.args['device'])
        self.E1 = torch.tensor(5e4).to(self.args['device'])
        self.E2 = torch.tensor(6e4).to(self.args['device'])
        self.k1 = torch.tensor(2.77e3).to(self.args['device']) * self.s2hr
        self.k2 = torch.tensor(2.6e3).to(self.args['device']) * self.s2hr
        self.dH1 = -torch.tensor(6e4).to(self.args['device']) / self.MW
        self.dH2 = -torch.tensor(7e4).to(self.args['device']) / self.MW
        self.aA = torch.tensor(3.5).to(self.args['device'])
        self.aB = torch.tensor(1).to(self.args['device'])
        self.aC = torch.tensor(0.5).to(self.args['device'])
        self.Cp = torch.tensor(4.2e3).to(self.args['device'])
        self.R = torch.tensor(8.314).to(self.args['device'])
        self.rho = torch.tensor(1000).to(self.args['device'])
        self.xA10 = torch.tensor(1).to(self.args['device'])
        self.xB10 = torch.tensor(0).to(self.args['device'])
        self.xA20 = torch.tensor(1).to(self.args['device'])
        self.xB20 = torch.tensor(0).to(self.args['device'])
        self.Hvap1 = -torch.tensor(35.3E3).to(self.args['device']) * self.sum_c
        self.Hvap2 = -torch.tensor(15.7E3).to(self.args['device']) * self.sum_c
        self.Hvap3 = -torch.tensor(40.68E3).to(self.args['device']) * self.sum_c

        self.kw = torch.tensor([0.7, 0.7, 3.5, 0.7, 0.7, 3.5, 0.7, 0.7, 3.5]).to(self.args['device'])  # noise deviation
        self.bw = torch.tensor([5., 5., 10., 5., 5., 10., 5., 5., 10.]).to(self.args['device'])  # noise bound

        self.xs = torch.tensor([0.1763, 0.6731, 480.3165, 0.1965, 0.6536, 472.7863, 0.0651, 0.6703, 474.8877]).to(
            self.args['device'])
        self.us = 1.12 * np.array([2.9e9, 1.0e9, 2.9e9])

        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.state_high = torch.tensor([0.5, 1., 700., 0.5, 1., 700., 0.5, 1., 700.]).to(self.args['device'])
        self.state_low = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.]).to(self.args['device'])

        self.action_low = 0.2 * self.us
        self.action_high = 1.5 * self.us

        # 修改至 steady-state set-point
        self.reference = self.xs

        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_action(self, T):
        action = generate_input()
        signal = action.signal_generate(T)
        return signal

    def get_testaction(self, T):
        action_test = generate_testinput()
        signal = action_test.signal_generate(T)
        return signal

    def step(self, step, input):
        action = input[step - 1, :].to(self.args['device'])
        x0 = self.state.to(self.args['device'])
        for i in range(self.sampling_steps):
            # process_noise = torch.randn_like(self.kw).to(self.args['device'])
            process_noise = torch.normal(mean=0, std=self.kw).to(self.args['device'])
            process_noise = torch.clamp(process_noise, -self.bw, self.bw)
            x0 = x0 + self.derivative(x0, action) * self.h + process_noise * self.h
        self.state = x0
        self.t += 1
        # cost = np.linalg.norm(self.state - self.reference)
        cost = torch.norm(self.state - self.reference)
        done = False
        data_collection_done = False

        return x0, cost, done, dict(reference=self.reference, data_collection_done=data_collection_done)

    def reset(self):
        self.a_holder = self.action_space.sample()
        self.state = (torch.rand_like(self.xs) * 0.2 + 1) * self.xs + torch.randn_like(self.xs) * (self.xs * 0.01)
        self.t = 0
        self.time = 0
        return self.state.to(self.args['device'])

    def reset(self, test=False, seed_=1):
        self.a_holder = self.action_space.sample()
        self.state = (torch.rand_like(self.xs) * 0.2 + 1) * self.xs + torch.randn_like(self.xs) * (self.xs * 0.01)
        if test:
            np.random.seed(seed_)
            self.state = torch.tensor(
                [0.9599, 0.9039, 1.1200, 0.9726, 1.1643, 0.8727, 0.9055, 0.8582, 0.8544]) * self.xs.cpu()
        self.t = 0
        self.time = 0
        return self.state.to(self.args['device'])

    def derivative(self, x, us):

        xA1 = x[0]
        xB1 = x[1]
        T1 = x[2]

        xA2 = x[3]
        xB2 = x[4]
        T2 = x[5]

        xA3 = x[6]
        xB3 = x[7]
        T3 = x[8]

        Q1 = us[0]
        Q2 = us[1]
        Q3 = us[2]

        xC3 = 1 - xA3 - xB3
        x3a = self.aA * xA3 + self.aB * xB3 + self.aC * xC3

        xAr = self.aA * xA3 / x3a
        xBr = self.aB * xB3 / x3a
        xCr = self.aC * xC3 / x3a

        F1 = self.F10 + self.Fr
        F2 = F1 + self.F20
        F3 = F2 - self.Fr - self.Fp

        f1 = self.F10 * (self.xA10 - xA1) / self.V1 + self.Fr * (xAr - xA1) / self.V1 - self.k1 * torch.exp(
            -self.E1 / (self.R * T1)) * xA1
        f2 = self.F10 * (self.xB10 - xB1) / self.V1 + self.Fr * (xBr - xB1) / self.V1 + self.k1 * torch.exp(
            -self.E1 / (self.R * T1)) * xA1 - self.k2 * torch.exp(
            -self.E2 / (self.R * T1)) * xB1
        f3 = self.F10 * (self.T10 - T1) / self.V1 + self.Fr * (T3 - T1) / self.V1 - self.dH1 * self.k1 * torch.exp(
            -self.E1 / (self.R * T1)) * xA1 / self.Cp - self.dH2 * self.k2 * torch.exp(
            -self.E2 / (self.R * T1)) * xB1 / self.Cp + Q1 / (self.rho * self.Cp * self.V1)

        f4 = F1 * (xA1 - xA2) / self.V2 + self.F20 * (self.xA20 - xA2) / self.V2 - self.k1 * torch.exp(
            -self.E1 / (self.R * T2)) * xA2
        f5 = F1 * (xB1 - xB2) / self.V2 + self.F20 * (self.xB20 - xB2) / self.V2 + self.k1 * torch.exp(
            -self.E1 / (self.R * T2)) * xA2 - self.k2 * torch.exp(
            -self.E2 / (self.R * T2)) * xB2
        f6 = F1 * (T1 - T2) / self.V2 + self.F20 * (self.T20 - T2) / self.V2 - self.dH1 * self.k1 * torch.exp(
            -self.E1 / (self.R * T2)) * xA2 / self.Cp - self.dH2 * self.k2 * torch.exp(
            -self.E2 / (self.R * T2)) * xB2 / self.Cp + Q2 / (self.rho * self.Cp * self.V2)

        f7 = F2 * (xA2 - xA3) / self.V3 - (self.Fr + self.Fp) * (xAr - xA3) / self.V3
        f8 = F2 * (xB2 - xB3) / self.V3 - (self.Fr + self.Fp) * (xBr - xB3) / self.V3
        f9 = F2 * (T2 - T3) / self.V3 + Q3 / (self.rho * self.Cp * self.V3) + (self.Fr + self.Fp) * (
                    xAr * self.Hvap1 + xBr * self.Hvap2 + xCr * self.Hvap3) / (
                     self.rho * self.Cp * self.V3)

        F = torch.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9]).to(self.args['device'])

        return F

    def render(self, mode='human'):

        return

    #
    # def get_action(self):
    #
    #     if self.t % self.action_sample_period == 0:
    #         self.a_holder = self.action_space.sample()
    #     a = self.a_holder + np.random.normal(np.zeros_like(self.us), self.us*0.01)
    #     a = np.clip(a, self.action_low, self.action_high)
    #
    #     return a

    def get_noise(self):
        scale = 0.1 * self.xs
        return np.random.normal(np.zeros_like(self.xs), scale)


import my_args

if __name__ == '__main__':
    args = my_args.args
    args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = three_tank_system(args)
    T = 2000
    path = []
    t1 = []
    s = env.reset()
    action = env.get_testaction(T)
    for i in range(int(T)):
        s, r, done, info = env.step(i, action)
        path.append(s.cpu().numpy())
        t1.append(i)
    path = np.array(path)
    state_dim = 9
    fig, ax = plt.subplots(state_dim, sharex=True, figsize=(15, 15))
    t = range(T)
    for i in range(state_dim):
        ax[i].plot(t, path[:, i], color='red')

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    # ax.plot(t2, path2, color='red',label='1')
    #
    # ax.plot(t3, path3, color='black', label='0.01')
    # ax.plot(t4, path4, color='orange', label='0.001')
    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.show()
    print('done')