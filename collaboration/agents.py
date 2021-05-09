import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from carle.env import CARLE
from carle.mcl import RND2D, AE2D 

import matplotlib.pyplot as plt


class TrainAgent(nn.Module):

    def __init__(self, **kwargs):
        super(TrainAgent, self).__init__()

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
        self.instances = kwargs["instances"] \
                if "instances" in kwargs.keys()\
                else 1


        self.initialize_policy()

    def initialize_policy(self):

        in_dim = self.observation_width * self.observation_height
        out_dim = self.action_width * self.action_height
        hid_dim = 256

        self.policy = torch.nn.Sequential(torch.nn.Linear(in_dim, hid_dim),\
                torch.nn.ReLU(),\
                torch.nn.Linear(hid_dim, out_dim),\
                torch.nn.Sigmoid())

    def forward(self, obs):

        instances = obs.shape[0]
        x = self.policy(obs.flatten())

        toggle_probability = x.reshape(instances, 1, self.action_height, self.action_width)
        
        action = 1.0 * (torch.rand_like(toggle_probability) <= toggle_probability)

        return action

class ConvGRNNAgent(TrainAgent):
    def __init__(self, **kwargs):
        """
        Submission agent, must produce actions (binary toggles) when called
        """
        super(ConvGRNNAgent, self).__init__(**kwargs)

    def reset_cells(self):

        self.cell_state *= 0

    def initialize_policy(self):
    
        self.hidden_channels = 16

        self.cells = torch.zeros(self.instances, 1, self.action_height // 4, \
                self.action_width // 4)

        self.feature_conv = nn.Sequential(\
                nn.Conv2d(1, self.hidden_channels, kernel_size=3, stride=2, padding=1), \
                nn.ReLU(), \
                nn.Conv2d(self.hidden_channels, self.hidden_channels, \
                        kernel_size=3, stride=2, padding=1), \
                nn.ReLU(), \
                nn.Conv2d(self.hidden_channels, self.hidden_channels, \
                        kernel_size=3, stride=2, padding=1), \
                nn.ReLU(), \

                nn.Conv2d(self.hidden_channels, 1, kernel_size=3, stride=2, padding=1), \
                nn.ReLU())

        self.gate_conv = nn.Sequential(\
                nn.Conv2d(2, self.hidden_channels, kernel_size=3, stride=1, padding=1), \
                nn.ReLU(), \
                nn.Conv2d(self.hidden_channels, 1, kernel_size=3, stride=1, padding=1), \
                nn.Sigmoid())

        self.action_conv = nn.Sequential(\
                nn.ConvTranspose2d(1, self.hidden_channels, kernel_size=2, stride=2, padding=0), \
                nn.ReLU(), \
                nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels, \
                        kernel_size=2, stride=2, padding=0), \
                nn.ReLU(), \
                nn.Conv2d(self.hidden_channels, 1, \
                        kernel_size=3, stride=1, padding=1), \
                nn.Sigmoid())


        nn.init.constant(self.gate_conv[2].bias, 2.0)
        nn.init.constant(self.action_conv[4].bias, -4.0)

    def forward(self, obs):

        instances = obs.shape[0]

        features = self.feature_conv(obs)
        gate_states = self.gate_conv(torch.cat([features, self.cells], dim=1))

        self.cells *= (1-gate_states)
        self.cells += gate_states * features

        x = self.action_conv(self.cells)

        toggle_probability = x.reshape(instances, 1, self.action_height, self.action_width)
        
        action = 1.0 * (torch.rand_like(toggle_probability) <= toggle_probability)

        return action
    
    def step(self):
        """
        update agent(s)
        
        this method is called everytime the CA universe is reset. 
        """
        return 0



if __name__ == "__main__":

    agent = ConvGRNNAgent() 

    obs = 1.0 * (torch.rand(1, 1, agent.observation_height, agent.observation_width) < 0.1)

    action = agent(obs)
