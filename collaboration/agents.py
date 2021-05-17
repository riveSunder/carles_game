import time
import numpy as np
from functools import reduce

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
        
        action = 0.0 * (torch.rand_like(toggle_probability) <= toggle_probability)

        return action

class ConvGRNNAgent(TrainAgent):
    def __init__(self, **kwargs):
        """
        Submission agent, must produce actions (binary toggles) when called
        """
        self.use_grad = False
        self.population_size = 4
        self.generation = 0
        self.episodes = 0
        self.max_episodes = 3
        self.agents = []
        self.fitness = []
        
        super(ConvGRNNAgent, self).__init__(**kwargs)

        params = self.get_params()

        self.initialize_policy()

        params = np.append(params[np.newaxis,:], \
                self.get_params()[np.newaxis,:], axis=0)

        means = np.mean(params, axis=0)
        var = np.var(params, axis=0)

        self.distribution = [means, np.diag(var)] 

    def reset_cells(self):

        self.cell_state *= 0

    def initialize_policy(self):
    
        self.hidden_channels = 8

        self.cells = nn.Parameter(torch.zeros(self.instances, 1, self.action_height // 4, \
                self.action_width // 4), requires_grad=False)

        self.feature_conv = nn.Sequential(\
                nn.Conv2d(1, self.hidden_channels, kernel_size=3, stride=2, padding=1), \
                nn.LeakyReLU(), \
                nn.Conv2d(self.hidden_channels, self.hidden_channels, \
                        kernel_size=3, stride=2, padding=1), \
                nn.LeakyReLU(), \
                nn.Conv2d(self.hidden_channels, self.hidden_channels, \
                        kernel_size=3, stride=2, padding=1), \
                nn.LeakyReLU(), \
                nn.Conv2d(self.hidden_channels, 1, kernel_size=3, stride=2, padding=1), \
                nn.Sigmoid())

        self.gate_conv = nn.Sequential(\
                nn.Conv2d(2, self.hidden_channels, kernel_size=3, stride=1, padding=1), \
                nn.LeakyReLU(), \
                nn.Conv2d(self.hidden_channels, 1, kernel_size=3, stride=1, padding=1), \
                nn.Sigmoid())

        self.action_conv = nn.Sequential(\
                nn.ConvTranspose2d(1, self.hidden_channels, kernel_size=2, stride=2, padding=0), \
                nn.LeakyReLU(), \
                nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels, \
                        kernel_size=2, stride=2, padding=0), \
                nn.LeakyReLU(), \
                nn.Conv2d(self.hidden_channels, 1, \
                        kernel_size=3, stride=1, padding=1), \
                nn.Sigmoid())


        nn.init.constant_(self.gate_conv[2].bias, 2.0 + np.random.randn()*1e-3)
        nn.init.constant_(self.action_conv[4].bias, -3.0 + np.random.randn()*1e-3)

        for params in [self.feature_conv.parameters(), \
                self.gate_conv.parameters(), \
                self.action_conv.parameters()]:

            for param in params: #self.lr_layers.parameters():
                param.requires_grad = self.use_grad

    def forward(self, obs):

        instances = obs.shape[0]

        features = self.feature_conv(obs)

        gate_states = self.gate_conv(torch.cat([features, self.cells], dim=1))

        self.cells *= (1-gate_states)
        self.cells += gate_states * features
        #self.cells += features

        x = self.action_conv(self.cells)

        toggle_probability = x.reshape(instances, 1, self.action_height, self.action_width)
        
        action = 1.0 * (torch.rand_like(toggle_probability) <= toggle_probability)

        return action
    
    def get_params(self):
        params = np.array([])

        for param in self.feature_conv.named_parameters():
            params = np.append(params, param[1].detach().cpu().numpy().ravel())

        for param in self.gate_conv.named_parameters():
            params = np.append(params, param[1].detach().cpu().numpy().ravel())

        for param in self.action_conv.named_parameters():
            params = np.append(params, param[1].detach().cpu().numpy().ravel())

        return params

    def set_params(self, my_params):

        param_start = 0

        for name, param in self.feature_conv.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), \
                    requires_grad=self.use_grad), \
                    requires_grad=self.use_grad).to(param[:].device)

            param_start = param_stop

        for name, param in self.gate_conv.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), \
                    requires_grad=self.use_grad), \
                    requires_grad=self.use_grad).to(param[:].device)

            param_start = param_stop

        for name, param in self.action_conv.named_parameters():

            param_stop = param_start + reduce(lambda x,y: x*y, param.shape)

            param[:] = torch.nn.Parameter(torch.tensor(\
                    my_params[param_start:param_stop].reshape(param.shape), \
                    requires_grad=self.use_grad), \
                    requires_grad=self.use_grad).to(param[:].device)

            param_start = param_stop


class CMAPopulation():
    def __init__(self, agent_fn, device="cpu"):
        """
        """
        self.use_grad = False
        self.population_size = 16
        self.meta_index = 0
        self.generation = 0
        self.episodes = 0
        self.device = torch.device(device)

        self.agents = []
        self.fitness = []
        
        self.population = [agent_fn() for ii in range(self.population_size)]

        for kk in range(self.population_size):
            self.population[kk].to(self.device)

        params = self.population[0].get_params()[np.newaxis,:]

        for jj in range(1,len(self.population)):
            params = np.append(params, \
                    self.population[jj].get_params()[np.newaxis,:], axis=0)

        means = np.mean(params, axis=0)
        var = np.var(params, axis=0)

        self.distribution = [means, np.diag(var)] 

    def __call__(self, obs, agent_index=0):

        action = self.population[agent_index + self.meta_index](obs)

        return action

    def update(self):
        """
        select champions and update the population distribution according to fitness

        """
        print("updating policy distribution")

        self.agents = [agent.get_params() for agent in self.population] 

        sorted_indices = list(np.argsort(np.array(self.fitness)))

        sorted_indices.reverse()

        sorted_params = np.array(self.agents)[sorted_indices]

        keep = self.population_size // 4

        elite_means = np.mean(sorted_params[0:keep], axis=0, keepdims=True)

        covariance = np.matmul(\
                (elite_means - self.distribution[0]).T,
                (elite_means - self.distribution[0]))

        self.distribution = [elite_means.squeeze(), covariance]


        np.save("../policies/grnn_mean_gen{}.npy".format(self.generation), \
                self.distribution[0])
        np.save("../policies/grnn_covar_gen{}.npy".format(self.generation), \
                self.distribution[1])

        np.save("../policies/grnn_best_gen{}.npy".format(self.generation),\
                self.agents[0])

        self.agents = []
        self.fitness = [] 

        print("gen. {} updated policy distribution".format(self.generation))
        self.generation += 1

    def step(self, rewards=[0,0,0,0.]):
        """
        update agent(s)
        this method is called everytime the CA universe is reset. 
        """ 

        self.fitness.extend(rewards)

        if len(self.fitness) >= self.population_size :
            self.update()
        else:
            self.meta_index = len(self.fitness)


if __name__ == "__main__":

    agent = ConvGRNNAgent() 

    obs = 0.0 * (torch.rand(1, 1, agent.observation_height, agent.observation_width) < 0.1)

    action = agent(obs)
    
    for ii in range(4):
        agent.step(rewards=torch.randn(10,))
        print(len(agent.fitness))

    agent.step(rewards=np.random.randn(np.random.randint(10),))

