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


class CARLA(Agent):
    def __init__(self, **kwargs):
        super(CARLA, self).__init__(**kwargs)

        self.use_grad = False
        self.population_size = 4
        self.generation = 0
        self.episodes = 0
        self.max_episodes = 1
        self.ca_steps = 8

        self.agents = []
        self.fitness = []

        self.save_path = kwargs["save_path"] if "save_path" in kwargs.keys() else "" 
        self.instances = kwargs["instances"] if "instances" in kwargs.keys() else 1 

    def initialize_policy(self):

        self.perceive = nn.Conv2d(1, 9, 3, groups=1, padding=1, stride=1, \
                padding_mode="circular", bias=False)

        self.perceive.weight.requires_grad = False

        self.perceive.weight.data.fill_(0)

        for ii in range(9):
            self.perceive.weight[ii, :, ii//3, ii%3].fill_(1)

        self.perceive.to(self.my_device)


        self.cellular_rules = nn.Sequential(nn.Conv2d(9, 32, 1, bias=False),\
                nn.ReLU(),
                nn.Conv2d(32, 1, 1, bias=True),
                nn.Sigmoid()).to(self.my_device)

        for param in self.cellular_rules.named_parameters():
            param[1].requires_grad = False

            if "bias" in param[0]:
                param[1].data.fill_(param[1][0].item() - 0.075)

        self.hallucinogen = torch.nn.Parameter(torch.rand(1)/5,\
                requires_grad=False).to(self.my_device)

    def hallucinate(self, obs):

        obs = obs + (1.0 * (torch.rand_like(obs)  < self.hallucinogen))\
                .float().to(obs.device)

        obs[obs>1.0] = 1.0

        return obs 
        
    def forward(self, obs):
        
        obs = self.hallucinate(obs)

        for jj in range(self.ca_steps):

            my_grid = self.perceive(obs)

            my_grid = self.cellular_rules(my_grid)

            #alive_mask = (my_grid[:,3:4,:,:] > 0.05).float()
            #my_grid *= alive_mask

        off_x = (obs.shape[2] - 64) // 2
        off_y = (obs.shape[3] - 64) // 2

        action_probabilities = my_grid[:,0:1,off_x:-off_x,off_y:-off_y]
        
        action = (action_probabilities > 0.5).float()

        return action

    def get_params(self):
        params = np.array([])

        params = np.append(params, self.hallucinogen.cpu().numpy())

        for param in self.cellular_rules.named_parameters():
            params = np.append(params, param[1].detach().cpu().numpy().ravel())

        return params

    def set_params(self, my_params):

        param_start = 0

        my_params[0] = np.clip(my_params[0], 0.0025, .90)
        self.hallucinogen = nn.Parameter(torch.Tensor(my_params[0:1]), \
                requires_grad=False).to(self.my_device)


        param_start += 1

        for name, param in self.cellular_rules.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), \
                    requires_grad=self.use_grad), \
                    requires_grad=self.use_grad).to(param[:].device)

            param_start = param_stop


