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


class ConvGRNN(Agent):
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

        self.save_path = kwargs["save_path"] if "save_path" in kwargs.keys() else "" 
        
        super(ConvGRNN, self).__init__(**kwargs)

        params = self.get_params()

        self.initialize_policy()

        params = np.append(params[np.newaxis,:], \
                self.get_params()[np.newaxis,:], axis=0)

        means = np.mean(params, axis=0)
        var = np.var(params, axis=0)

        self.distribution = [means, np.diag(var)] 
        self.tag = int(time.time())

    def reset_cells(self):

        self.cell_state *= 0

    def initialize_policy(self):
    
        self.hidden_channels = 8

        self.cells = nn.Parameter(torch.zeros(self.instances, 1, self.observation_height // 16, \
                self.observation_width // 16), requires_grad=False)

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
                nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels, \
                        kernel_size=2, stride=2, padding=0), \
                nn.LeakyReLU(), \
                nn.Conv2d(self.hidden_channels, 1, \
                        kernel_size=3, stride=1, padding=1), \
                nn.Sigmoid())


        nn.init.constant_(self.gate_conv[2].bias, 2.0 + np.random.randn())
        nn.init.constant_(self.action_conv[4].bias, -1.0 + np.random.randn())

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

    def update(self):
        """
        select champions and update the population distribution according to fitness

        """
        print("updating policy distribution")

        sorted_indices = list(np.argsort(np.array(self.fitness)))

        sorted_indices.reverse()

        sorted_params = np.array(self.agents)[sorted_indices]

        keep = self.population_size // 2

        elite_means = np.mean(sorted_params[0:keep], axis=0, keepdims=True)

        covariance = np.matmul(\
                (elite_means - self.distribution[0]).T,
                (elite_means - self.distribution[0]))

        self.distribution = [elite_means.squeeze(), covariance]

        save_mean_path = os.path.join(self.save_path, "grnn{}_mean_gen{}.npy"\
                .format(self.tag, self.generation))
        save_covar_path = os.path.join(self.save_path, "grnn{}_covar_gen{}.npy"\
                .format(self.tag, self.generation))
        save_params_path = os.path.join(self.save_path, "grnn{}_params_gen{}.npy"\
                .format(self.tag, self.generation))

        np.save(save_mean_path, self.distribution[0])
        np.save(save_covar_path, self.distribution[1])
        np.save(save_params_path, self.agents[0])

        self.agents = []
        self.fitness = [] 

        print("gen. {} updated policy distribution".format(self.generation))
        self.generation += 1

    def step(self, rewards=None):
        """
        update agent(s)
        this method is called everytime the CA universe is reset. 
        """ 


        if rewards is not None:
            if type(rewards) == torch.Tensor:
                fitness = np.sum(np.array(rewards.detach().cpu()))
            else:
                fitness = np.sum(np.array(rewards))
        else:
            fitness = np.random.randn(1)
        
        self.agents.append(self.get_params())
        self.fitness.append(fitness)

        if len(self.fitness) >= self.population_size:
            self.update()
            
        if self.episodes <= self.max_episodes:
            new_params = np.random.multivariate_normal(\
                    self.distribution[0], self.distribution[1])
            self.set_params(new_params)

            self.episodes = 0

        else:

            self.episodes += 1
