import os
import time
import numpy as np
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from carle.env import CARLE
from carle.mcl import RND2D, AE2D 

import matplotlib.pyplot as plt


class CMAPopulation():
    def __init__(self, agent_fn, device="cpu", **kwargs):
        """
        """
        self.use_grad = False
        self.population_size = kwargs["population_size"] \
                if "population_size" in kwargs.keys() else 16
        self.meta_index = 0
        self.generation = 0
        self.l2_penalty = 1e0
        self.lr = 1e-1

        self.elitism = True

        self.my_device = torch.device(device)
        self.device = device

        self.episodes = kwargs["episodes"] if "episodes" in kwargs.keys() else 1
        self.save_path = kwargs["save_path"] if "save_path" in kwargs.keys() else "" 

        self.agents = []
        self.fitness = []
        
        self.population = [agent_fn(device=self.device) for ii in range(self.population_size)]

        self.tag = int(time.time())

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

        action = self.population[(agent_index + self.meta_index) \
                % self.population_size](obs)

        return action

    def update(self):
        """
        select champions and update the population distribution according to fitness

        """
        print("updating policy distribution")

        self.agents = [agent.get_params() for agent in self.population]
        
        for hh in range(len(self.agents)):
            self.fitness[hh] = self.fitness[hh] - self.l2_penalty * \
                    np.sum((self.agents[hh])**2)


        sorted_indices = list(np.argsort(np.array(self.fitness)))

        sorted_indices.reverse()

        sorted_params = np.array(self.agents)[sorted_indices]

        keep = self.population_size // 4

        elite_means = np.mean(sorted_params[0:keep], axis=0, keepdims=True)

        covariance = np.matmul(\
                (elite_means - self.distribution[0]).T,
                (elite_means - self.distribution[0]))

        new_means = (1-self.lr) * self.distribution[0] \
                + self.lr * elite_means.squeeze()
            
        self.distribution = [new_means, covariance]

        save_mean_path = self.save_path + f"mean_gen{self.generation}.npy"
        save_covar_path = self.save_path + f"covar_gen{self.generation}.npy"
        save_params_path = self.save_path + f"best_params_gen{self.generation}.npy"

        np.save(save_mean_path, self.distribution[0])
        np.save(save_covar_path, self.distribution[1])
        np.save(save_params_path, sorted_params[0])

        self.agents = []
        self.fitness = [] 

        print("gen. {} updated policy distribution".format(self.generation))

        #for mm in range(2):
            # elitism: keep the best 25% of agents

        #    self.population[mm].set_params(sorted_params[mm])

        for nn in range(2, self.population_size):
            self.population[nn].set_params(np.random.multivariate_normal(\
                    self.distribution[0], self.distribution[1]))

        self.generation += 1
        self.meta_index = 0

    def step(self, rewards=[0,0,0,0.]):
        """
        update agent(s)
        this method is called everytime the CA universe is reset. 
        """ 

        if type(rewards) == list:
            self.fitness.extend(rewards)
        else:
            self.fitness.append(np.sum(rewards)/rewards.shape[0])


        if len(self.fitness) >= (self.population_size * self.episodes):
            if self.episodes > 1:
                self.fitness = np.array(self.fitness).reshape(-1, self.population_size)
                self.fitness = np.mean(self.fitness, axis=0)
                self.update()
            else:
                self.update()
        else:
            self.population[self.meta_index % self.population_size].reset()
            self.meta_index = len(self.fitness)

        # reset the next agent. used for random parameter initialization 
        # for Hebbian policies.
        self.population[(self.meta_index )% self.population_size].reset()
