import os
import time
import numpy as np
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from carle.env import CARLE
from carle.mcl import RND2D, AE2D 

class Agent(nn.Module):

    def __init__(self, **kwargs):
        super(Agent, self).__init__()

        self.use_grad = kwargs["use_grad"] \
                if "use_grad" in kwargs.keys() else False
        self.lr = kwargs["lr"] if "lr" in kwargs.keys() else 1e-4

        self.action_width = kwargs["action_width"] \
                if "action_width" in kwargs.keys()\
                else 64
        self.action_height = kwargs["action_height"] \
                if "action_height" in kwargs.keys()\
                else 64
        self.observation_width = kwargs["width"] \
                if "width" in kwargs.keys()\
                else 128
        self.observation_height = kwargs["height"] \
                if "height" in kwargs.keys()\
                else 128
        self.instances = kwargs["instances"] \
                if "instances" in kwargs.keys()\
                else 1

        self.my_device = torch.device(kwargs["device"]) if "device" in kwargs.keys()\
                else torch.device("cpu")

        self.initialize_policy()

    def initialize_policy(self):

        in_dim = self.observation_width * self.observation_height
        out_dim = self.action_width * self.action_height
        hid_dim = 256

        self.policy = torch.nn.Sequential(torch.nn.Linear(in_dim, hid_dim),\
                torch.nn.ReLU(),\
                torch.nn.Linear(hid_dim, out_dim))

    def initialize_optimizer(self):
        pass

    def step(self, reward):
        pass

    def reset(self):
        pass


    def forward(self, obs):

        instances = obs.shape[0]
        x = self.policy(obs.flatten())

        toggle_probability = x.reshape(instances, 1, self.action_height, self.action_width)
        
        action = 0.0 * (torch.rand_like(toggle_probability) <= toggle_probability)

        return action

