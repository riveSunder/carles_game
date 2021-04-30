import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from carle.env import CARLE
from carle.mcl import RND2D, AE2D 

import matplotlib.pyplot as plt


class DemoAgent(nn.Module):

    def __init__(self, **kwargs):
        super(DemoAgent, self).__init__()

        self.action_width = kwargs["action_width"] \
                if "action_width" in kwargs.keys()\
                else 64
        self.action_height = kwargs["action_height"] \
                if "action_height" in kwargs.keys()\
                else 64
        self.observation_width = kwargs["observation_width"] \
                if "observatoin_width" in kwargs.keys()\
                else 256
        self.observation_height = kwargs["observation_height"] \
                if "observation_height" in kwargs.keys()\
                else 256

        self.toggle_rate = 0.40
        
        in_dim = self.observation_width * self.observation_height
        out_dim = self.action_width * self.action_height
        hid_dim = 256

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, hid_dim),\
                torch.nn.ReLU(),\
                torch.nn.Linear(hid_dim, out_dim),\
                torch.nn.Sigmoid())

    def forward(self, obs):

        instances = obs.shape[0]
        x = self.mlp(obs.flatten())

        x = x.reshape(instances, 1, self.action_height, self.action_width)
        
        action = 1.0 * (x <= self.toggle_rate)

        return action

class SubmissionAgent(DemoAgent):

    def __init__(self, **kwargs):
        """
        Submission agent, must produce actions (binary toggles) when called
        """
        
        super(SubmissionAgent, self).__init__(**kwargs)

        
