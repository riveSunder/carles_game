import time

import numpy as np

import torch
import torch.nn as nn

from carle.env import CARLE
from carle.mcl import AE2D, RND2D

from collaboration.agents import ConvGRNNAgent 

def train():

    max_generations = int(1e5)
    max_steps = 1024
    my_instances = 4
    number_steps = 0

    # defin environment and exploration bonus wrappers
    env = CARLE(instances = my_instances, use_cuda = True)

    if torch.cuda.is_available():
        my_device = torch.device("cuda")
    else:
        my_device = torch.device("cpu")

    env = RND2D(env)
    env = AE2D(env)


    agent = ConvGRNNAgent(instances=my_instances)
    agent.initialize_policy()
    agent.to(my_device)
    

    agent.population_size = 8
    agent.max_episodes = 1

    for generation in range(max_generations):

        t0 = time.time()

        obs = env.reset()

        rewards = torch.Tensor([]).to(my_device)
        reward_sum = []
        number_steps = 0
        temp_generation = 0.0 + agent.generation

        while agent.generation <= temp_generation:
        #len(agent.fitness) <= (agent.population_size * agent.max_episodes):
            
            if number_steps >= max_steps:
                number_steps = 0
                agent.step(rewards)
                reward_sum.append(np.sum(rewards.detach().cpu().numpy()))

                obs = env.reset()
                rewards = torch.Tensor([]).to(my_device)

            action = agent(obs)

            obs, reward, done, info = env.step(action)

            rewards = torch.cat([rewards, reward])
            number_steps += 1

        t1 = time.time()


        print("generation {}, mean, max, min, std. dev. fitness: ".format(generation), \
                 "{:.3e}, {:.3e}, {:.3e}, {:.3e}".format(\
                np.mean(reward_sum), np.max(reward_sum), np.min(reward_sum), \
                np.std(reward_sum)))
        print("steps per second = {:.4e}".format(\
                (env.inner_env.instances * max_steps \
                * agent.max_episodes * agent.population_size) / (t1 - t0)))
            


if __name__ == "__main__":

    train()
