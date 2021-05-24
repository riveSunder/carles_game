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
                if "observation_width" in kwargs.keys()\
                else 256
        self.observation_height = kwargs["observation_height"] \
                if "observation_height" in kwargs.keys()\
                else 256
        self.instances = kwargs["instances"] \
                if "instances" in kwargs.keys()\
                else 1

        self.my_device = torch.device(kwargs["device"]) if "device" in kwargs.keys()\
                else torch.device("cpu")

        self.initialize_policy()

    def reset(self):
        pass

    def initialize_policy(self):

        in_dim = self.observation_width * self.observation_height
        out_dim = self.action_width * self.action_height
        hid_dim = 256

        self.policy = torch.nn.Sequential(torch.nn.Linear(in_dim, hid_dim),\
                torch.nn.ReLU(),\
                torch.nn.Linear(hid_dim, out_dim))

    def forward(self, obs):

        instances = obs.shape[0]
        x = self.policy(obs.flatten())

        toggle_probability = x.reshape(instances, 1, self.action_height, self.action_width)
        
        action = 0.0 * (torch.rand_like(toggle_probability) <= toggle_probability)

        return action

class CARLA(TrainAgent):
    def __init__(self, **kwargs):
        super(CARLA, self).__init__(**kwargs)

        self.use_grad = False
        self.population_size = 4
        self.generation = 0
        self.episodes = 0
        self.max_episodes = 1
        self.ca_steps = 32

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
                param[1].data.fill_(param[1][0].item() - 0.025)

        self.hallucinogen = torch.nn.Parameter(torch.rand(1)/10,\
                requires_grad=False).to(self.my_device)
            

    def hallucinate(self, obs):

        obs = obs + (1.0 * (torch.rand_like(obs)  < self.hallucinogen)).float()
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


class HARLI(CARLA):
    """
    Hebbian Automata Reinforcement Learning Improviser
    """

    def __init__(self, **kwargs):
        super(HARLI, self).__init__(**kwargs)

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
            my_grid_01 = self.act_0(my_grid_01)

            # first layer eligibility traces 
            grid_00_mean = my_grid_00.mean(-1).mean(-1)
            grid_01_mean = my_grid_01.mean(-1).mean(-1)

            self.oi_0 += grid_00_mean.mean(0).unsqueeze(0)
            self.oi_oj_0 += torch.matmul(grid_01_mean.T, grid_00_mean)
            self.oj_0 += grid_01_mean.mean(0).unsqueeze(-1)

            # second layer of ca policy rules 
            my_grid_11 = self.ca_1(my_grid_01)
            my_grid_11 = self.act_1(my_grid_11 - self.bias_1)

            # second layer eligibility traces
            grid_11_mean = my_grid_11.mean(-1).mean(-1)

            self.oi_1 += grid_01_mean.mean(0).unsqueeze(0)
            self.oi_oj_1 += torch.matmul(grid_11_mean.T, grid_01_mean)
            self.oj_1 += grid_11_mean.mean(0).unsqueeze(-1)

            my_grid = my_grid_11

            #alive_mask = (my_grid[:,3:4,:,:] > 0.05).float()
            #my_grid *= alive_mask

        off_x = (obs.shape[2] - 64) // 2
        off_y = (obs.shape[3] - 64) // 2

        action_probabilities = my_grid[:,0:1,off_x:-off_x,off_y:-off_y]
        
        action = (action_probabilities > 0.5).float()

        self.hebbian_update()

        return action

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

        self.save_path = kwargs["save_path"] if "save_path" in kwargs.keys() else "" 
        
        super(ConvGRNNAgent, self).__init__(**kwargs)

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


        nn.init.constant_(self.gate_conv[2].bias, 2.0 + np.random.randn())
        nn.init.constant_(self.action_conv[4].bias, -5.0 + np.random.randn())

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



if __name__ == "__main__":


    env = CARLE(instances=4, use_cuda=True)

    agent = CARLA(device="cuda")

    obs = env.reset()

    params = agent.get_params()

    agent.set_params(params)

    t2 = time.time()
    for kk in range(100):
        action = agent(obs)

        print(action.sum())
        obs, r, d, i = env.step(action)
    
    t3 = time.time()
    
   

    agent = HARLI(device="cuda")

    obs = env.reset()

    params = agent.get_params()

    agent.set_params(params)

    t0 = time.time()
    for kk in range(100):
        action = agent(obs)

        print(action.sum())
        obs, r, d, i = env.step(action)

    t1 = time.time()

    print(t1-t0, t3-t2)
    import pdb; pdb.set_trace()
    
    
