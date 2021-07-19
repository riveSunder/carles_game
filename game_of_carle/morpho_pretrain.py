import os
import sys
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from carle.env import CARLE
from carle.mcl import MorphoBonus


from game_of_carle.agents.carla import CARLA 

def get_rle_pattern(env, rle_path, dim=16):

    pad_1 = nn.ZeroPad2d((int(dim - 6) // 2, 1, (dim - 6) // 2,1))

    my_rle = env.read_rle(rle_path)
    pattern = pad_1(env.rle_to_grid(my_rle))[:dim, :dim]

    pattern = pattern.unsqueeze(0).unsqueeze(0)
    
    return pattern

def corrupt_input(my_input, degradation=0.0, batch_size=32):

    #corrupt_input = torch.abs(my_input - (torch.rand_like(my_input) < degradation).float())
    corrupt_input = my_input * (torch.rand_like(my_input) > degradation).float()

    return corrupt_input
    

def train():

    num_epochs = 262144
    degradation = 0.001
    degradation_rate = 0.0001
    my_device = "cuda:0"
    lr = 3e-4
    batch_size = 4 

    file_path = os.path.split(os.path.abspath(__file__))[0]

    file_path = os.path.split(file_path)[0]

    pattern_list = ["spaceship_duck.rle"] #["glider_2.rle", "glider_1.rle"]

    env = CARLE(instances=len(pattern_list) * 6 * batch_size, use_grad=True, device=my_device)
    loss_fn = torch.nn.MSELoss()

    pattern =  get_rle_pattern(env, os.path.join(file_path, pattern_list[0]))
    patterns = pattern
    patterns = torch.cat([patterns, pattern.flip(2)])
    patterns = torch.cat([patterns, pattern.flip(3)])
    patterns = torch.cat([patterns, pattern.transpose(2,3).flip(2)])
    patterns = torch.cat([patterns, pattern.transpose(2,3).flip(3)])
    patterns = torch.cat([patterns, pattern.transpose(2,3)])

    for pattern_name in pattern_list[1:]:
        patterns = torch.cat([pattern, get_rle_pattern(env, os.path.join(file_path, pattern_name))])
        patterns = torch.cat([patterns, pattern.flip(2)])
        patterns = torch.cat([patterns, pattern.flip(3)])
        patterns = torch.cat([patterns, pattern.transpose(2,3).flip(2)])
        patterns = torch.cat([patterns, pattern.transpose(2,3).flip(3)])
        patterns = torch.cat([patterns, pattern.transpose(2,3)])


    for ii in range(batch_size):
        patterns  = torch.cat([patterns, patterns])

    pattern = patterns.to(my_device)

    agent = CARLA(instances=patterns.shape[0], \
            use_grad=True, device=my_device, lr=lr) 

    for epoch in range(num_epochs):

        my_input = corrupt_input(pattern, degradation)

        output = agent(my_input, get_prob=True)

        loss = loss_fn(output, pattern)

        agent.step(loss)

        if loss < 0.009:
            degradation = min([0.75, degradation+degradation_rate])
        else:
            degradation = max([0.001, degradation-degradation_rate])

        if epoch % 1000 == 0:
            print(f"loss at epoch {epoch} = {loss:.3}, degradation = {degradation:.3}")

            my_file_path = os.path.abspath(os.path.dirname(__file__))
            my_save_path = os.path.join(os.path.split(my_file_path)[0], \
                    "policies", "carla_pre_train_00.npy")
            np.save(my_save_path, agent.get_params())

        
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    train()
