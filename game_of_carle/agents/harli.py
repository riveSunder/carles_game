import os
import time
import numpy as np
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from carle.env import CARLE
from carle.mcl import RND2D, AE2D 

from game_of_carle.agents.agent import Agent
from game_of_carle.agents.carla import CARLA

class HARLI(CARLA):
    """
    Hebbian Automata Reinforcement Learning Improviser
    """

    def __init__(self, **kwargs):
        super(HARLI, self).__init__(**kwargs)
        self.obs_dim = kwargs["obs_dim"] if "obs_dim" in kwargs.keys() else 256
        self.act_dim = kwargs["act_dim"] if "act_dim" in kwargs.keys() else 64

    def reset(self):

        self.ca_0 = nn.Conv2d(9,32, 1, bias=False).to(self.my_device)
        self.ca_1 = nn.Conv2d(32,1, 1, bias=False).to(self.my_device)

        for param in self.ca_0.named_parameters():
            param[1].requires_grad = False

        for param in self.ca_1.named_parameters():
            param[1].requires_grad = False


        self.zero_eligibility()

    def zero_eligibility(self):

        self.oi_oj_0 = torch.zeros(32, 9).to(self.my_device)
        self.oi_0 = torch.zeros(1, 9).to(self.my_device)
        self.oj_0 = torch.zeros(32, 1).to(self.my_device)

        self.oi_oj_1 = torch.zeros(1, 32).to(self.my_device)
        self.oi_1 = torch.zeros(1, 32).to(self.my_device)
        self.oj_1 = torch.zeros(1, 1).to(self.my_device)

    def initialize_policy(self):
        # initialize hebbian learning parameters

        self.perceive = nn.Conv2d(1, 9, 3, groups=1, padding=1, stride=1, \
                padding_mode="circular", bias=False)

        self.perceive.weight.requires_grad = False

        self.perceive.weight.data.fill_(0)

        for ii in range(9):
            self.perceive.weight[ii, :, ii//3, ii%3].fill_(1)

        self.perceive.to(self.my_device)

        self.eta_0 = nn.Parameter(torch.randn(32, 9) / 20.0, requires_grad=False).to(self.my_device)
        self.a_0 = nn.Parameter(torch.randn(32, 9), requires_grad=False).to(self.my_device)
        self.b_0 = nn.Parameter(torch.randn(1, 9), requires_grad=False).to(self.my_device)
        self.c_0 = nn.Parameter(torch.randn(32, 1), requires_grad=False).to(self.my_device)
        self.d_0 = nn.Parameter(torch.randn(32, 9), requires_grad=False).to(self.my_device)

        self.eta_1 = nn.Parameter(torch.randn(1, 32) / 20.0, requires_grad=False).to(self.my_device)
        self.a_1 = nn.Parameter(torch.randn(1, 32), requires_grad=False).to(self.my_device)
        self.b_1 = nn.Parameter(torch.randn(1, 32), requires_grad=False).to(self.my_device)
        self.c_1 = nn.Parameter(torch.randn(1, 1), requires_grad=False).to(self.my_device)
        self.d_1 = nn.Parameter(torch.randn(1, 32), requires_grad=False).to(self.my_device)
   
        self.act_0 = nn.Tanh()
        self.act_1 = nn.Sigmoid()

        self.reset()

        self.hallucinogen = torch.nn.Parameter(torch.rand(1)/10,\
                requires_grad=False).to(self.my_device)

        self.bias_1 = torch.nn.Parameter(-torch.rand(1)/10,\
                requires_grad=False).to(self.my_device)

    def hebbian_update(self):
        # update weights

        for param_0 in self.ca_0.parameters():

            param_0[:,:,0,0] += self.eta_0 * (\
                    self.a_0 * self.oi_oj_0 \
                    + self.b_0 * self.oi_0 \
                    + self.c_0 * self.oj_0 \
                    + self.d_0)

        for param_1 in self.ca_1.parameters():

            param_1[:,:,0,0] += self.eta_1 * (\
                    self.a_1 * self.oi_oj_1 \
                    + self.b_1 * self.oi_1 \
                    + self.c_1 * self.oj_1 \
                    + self.d_1)

        if (0):
            self.zero_eligibility()

    def crop_to_act(self, action):

        offset = (self.obs_dim - self.act_dim) // 2
        
        return action[:, :, offset:-offset, offset:-offset]


    def forward(self, obs):
        
        my_grid = self.hallucinate(obs)

        for jj in range(self.ca_steps):

            # eligibility traces calculated from pre/post synaptic
            # values after activation function is applied, but could 
            # just as easily use values from before activation

            # calculate non-deterministic neighborhoods
            my_grid_00 = self.perceive(my_grid)

            # first layer of ca policy rules
            my_grid_01 = self.ca_0(my_grid_00)
            my_grid_01 = F.dropout(my_grid_01, p =0.025)
            my_grid_01 = self.act_0(my_grid_01)

            # first layer eligibility traces 
            grid_00_mean = my_grid_00.mean(-1).mean(-1)
            grid_01_mean = my_grid_01.mean(-1).mean(-1)

            self.oi_0 += grid_00_mean.mean(0).unsqueeze(0)
            self.oi_oj_0 += torch.matmul(grid_01_mean.T, grid_00_mean)
            self.oj_0 += grid_01_mean.mean(0).unsqueeze(-1)

            # second layer of ca policy rules 
            my_grid_11 = self.ca_1(my_grid_01)
            my_grid_11 = F.dropout(my_grid_11, p =0.025)
            my_grid_11 = self.act_1(my_grid_11 - self.bias_1)

            # second layer eligibility traces
            grid_11_mean = my_grid_11.mean(-1).mean(-1)

            self.oi_1 += grid_01_mean.mean(0).unsqueeze(0)
            self.oi_oj_1 += torch.matmul(grid_11_mean.T, grid_01_mean)
            self.oj_1 += grid_11_mean.mean(0).unsqueeze(-1)

            my_grid = my_grid_11

            #alive_mask = (my_grid[:,3:4,:,:] > 0.05).float()
            #my_grid *= alive_mask

        action_probabilities = my_grid 
        
        action = (action_probabilities > 0.5).float()

        self.hebbian_update()

        return self.crop_to_act(action)

    def get_weights(self):

        weights = np.array([])

        for param_0 in self.ca_0.parameters():

            weights = np.append(weights, np.array(param_0).ravel())

        for param_1 in self.ca_1.parameters():

            weights = np.append(weights, np.array(param_0).ravel())

        return weights

    def get_params(self):
        params = np.array([])

        params = np.append(params, self.hallucinogen.cpu().numpy())
        params = np.append(params, self.bias_1.cpu().numpy())

        params = np.append(params, self.eta_0.detach().cpu().numpy().ravel())
        params = np.append(params, self.a_0.detach().cpu().numpy().ravel())
        params = np.append(params, self.b_0.detach().cpu().numpy().ravel())
        params = np.append(params, self.c_0.detach().cpu().numpy().ravel())
        params = np.append(params, self.d_0.detach().cpu().numpy().ravel())

        params = np.append(params, self.eta_1.detach().cpu().numpy().ravel())
        params = np.append(params, self.a_1.detach().cpu().numpy().ravel())
        params = np.append(params, self.b_1.detach().cpu().numpy().ravel())
        params = np.append(params, self.c_1.detach().cpu().numpy().ravel())
        params = np.append(params, self.d_1.detach().cpu().numpy().ravel())

        return params

    def set_params(self, my_params):

        param_start = 0

        self.hallucinogen = nn.Parameter(torch.Tensor(my_params[0:1]), \
                requires_grad=False).to(self.my_device)
        param_start = 1
        self.bias_1 = nn.Parameter(torch.Tensor(my_params[1:2]), \
                requires_grad=False).to(self.my_device)
        param_start = 2

        param_stop = param_start + reduce(lambda x,y: x*y, self.eta_0.shape)
        self.eta_0 = nn.Parameter(torch.Tensor(my_params[param_start:param_stop])\
                .reshape(self.eta_0.shape),\
                requires_grad=False).to(self.my_device)
        param_start = param_stop

        param_stop = param_start + reduce(lambda x,y: x*y, self.a_0.shape)
        self.a_0 = nn.Parameter(torch.Tensor(my_params[param_start:param_stop])\
                .reshape(self.a_0.shape),\
                requires_grad=False).to(self.my_device)
        param_start = param_stop

        param_stop = param_start + reduce(lambda x,y: x*y, self.b_0.shape)
        self.b_0 = nn.Parameter(torch.Tensor(my_params[param_start:param_stop])\
                .reshape(self.b_0.shape),\
                requires_grad=False).to(self.my_device)
        param_start = param_stop

        param_stop = param_start + reduce(lambda x,y: x*y, self.c_0.shape)
        self.c_0 = nn.Parameter(torch.Tensor(my_params[param_start:param_stop])\
                .reshape(self.c_0.shape),\
                requires_grad=False).to(self.my_device)
        param_start = param_stop

        param_stop = param_start + reduce(lambda x,y: x*y, self.d_0.shape)
        self.d_0 = nn.Parameter(torch.Tensor(my_params[param_start:param_stop])\
                .reshape(self.d_0.shape),\
                requires_grad=False).to(self.my_device)
        param_start = param_stop

        param_stop = param_start + reduce(lambda x,y: x*y, self.eta_1.shape)
        self.eta_1 = nn.Parameter(torch.Tensor(my_params[param_start:param_stop])\
                .reshape(self.eta_1.shape),\
                requires_grad=False).to(self.my_device)
        param_start = param_stop

        param_stop = param_start + reduce(lambda x,y: x*y, self.a_1.shape)
        self.a_1 = nn.Parameter(torch.Tensor(my_params[param_start:param_stop])\
                .reshape(self.a_1.shape),\
                requires_grad=False).to(self.my_device)
        param_start = param_stop

        param_stop = param_start + reduce(lambda x,y: x*y, self.b_1.shape)
        self.b_1 = nn.Parameter(torch.Tensor(my_params[param_start:param_stop])\
                .reshape(self.b_1.shape),\
                requires_grad=False).to(self.my_device)
        param_start = param_stop

        param_stop = param_start + reduce(lambda x,y: x*y, self.c_1.shape)
        self.c_1 = nn.Parameter(torch.Tensor(my_params[param_start:param_stop])\
                .reshape(self.c_1.shape),\
                requires_grad=False).to(self.my_device)
        param_start = param_stop

        param_stop = param_start + reduce(lambda x,y: x*y, self.d_1.shape)
        self.d_1 = nn.Parameter(torch.Tensor(my_params[param_start:param_stop])\
                .reshape(self.d_1.shape),\
                requires_grad=False).to(self.my_device)
        param_start = param_stop

