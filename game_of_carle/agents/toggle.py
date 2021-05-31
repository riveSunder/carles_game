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


class Toggle(Agent):
    """
    agent for optimizing toggle parameters directly. 

    agent consists of parameters of the same dimension as the action space,
    at first step these are used as the action, all subsequent steps incur no 
    action
    """
    def __init__(self, **kwargs):
        super(Toggle, self).__init__(**kwargs)

        self.reset()

    def sigmoid(self,x):
        
        return 1 / (1 + np.exp(-x))

    def initialize_policy(self):

        self.toggle_width = 32
        assymetry_width = (self.action_width - self.toggle_width) % 2
        assymetry_height = (self.action_height - self.toggle_width) % 2

        width_padding = (self.action_width - self.toggle_width) // 2 
        height_padding = (self.action_height - self.toggle_width) // 2

        self.action_padding = nn.ZeroPad2d(padding=\
                (height_padding, height_padding + assymetry_height,\
                width_padding, width_padding + assymetry_width))

        self.toggles = (np.random.rand(1, 1, \
                self.toggle_width, self.toggle_width))


    def forward(self, obs):

        if self.first_step:
            action = torch.tensor(self.toggles > 0.5).float().to(self.my_device)
            action = self.action_padding(action)
            action *= torch.ones(self.instances, 1, \
                    self.action_height, self.action_width).to(self.my_device)

            self.first_step = False
        else: 
            action = torch.zeros(self.instances, 1, \
                    self.action_height, self.action_width).to(self.my_device)
            
        return action

    def get_params(self):

        return self.toggles.ravel()

    def set_params(self, params):
    
        self.toggles = params.reshape(1, 1, \
                self.toggle_width, self.toggle_width)

        self.toggles = np.clip(self.toggles, 0.0, 1.0)


    def reset(self):
        
        self.first_step = True

if __name__ == "__main__":
    
    env = CARLE()

    agent = Toggle()

    for jj in range(3):

        obs = env.reset()
        agent.reset()

        for ii in range(10):
            action = agent(obs)
            
            o, r, d, i = env.step(action)

            print(action.sum())
