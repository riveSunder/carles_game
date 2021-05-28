import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from carle.env import CARLE
from carle.mcl import AE2D, RND2D, CornerBonus, PufferDetector, SpeedDetector

from game_of_carle.agents.grnn import ConvGRNN
from game_of_carle.agents.carla import CARLA 
from game_of_carle.agents.harli import HARLI 
from game_of_carle.algos.cma import CMAPopulation

def train(wrappers = [CornerBonus], agent_fns=[HARLI, CARLA, ConvGRNNAgent], seeds=[13, 42]):

    max_generations = int(256)
    max_steps = 512
    my_instances = 1
    number_steps = 0


    # define environment and exploration bonus wrappers
    env = CARLE(instances = my_instances, use_cuda = True, height=128, width=128)

    my_device = env.my_device

    for wrapper in wrappers:

        env = wrapper(env)

    test_rules = ["B3/S012345678", "B3/S345678"]

    validation_rules = [\
            "B3/S145678",\
#            "B3/S12345678",\
#            "B3/S0245678",\
#            "B3/S01345678"\
            ]

    training_rules = [\
#            "B3/S245678",\
#            "B3/S2345678",\
#            "B3/S1345678",\
#            "B3/S1245678",\
#            "B3/S045678",\
#            "B3/S0345678",\
#            "B3/S02345678",\
            "B3/S0145678",\
            #"B3/S01245678",\
            ]

    for my_seed in seeds:

        np.random.seed(my_seed)
        torch.manual_seed(my_seed)

        for agent_fn in agent_fns:

            time_stamp = int(time.time())

            agent = CMAPopulation(agent_fn, device="cuda", episodes=4)

            experiment_name = agent.population[0].__class__.__name__ + f"_{my_seed}_{time_stamp}"
            my_file_path = os.path.abspath(os.path.dirname(__file__))
            my_directory_path = os.path.join(my_file_path, "../policies/")
            my_save_path = os.path.join(my_directory_path, experiment_name)

            agent.save_path = my_save_path
            
            agent.population_size = 16

            # with a vectorization of 4, don't need to repeat "episodes"
            agent.max_episodes = 4

            writer_path = "/".join(my_file_path.split("/")[:-1])
            writer_path = os.path.join(writer_path, f"experiments/logs/{experiment_name}")

            
            writer = SummaryWriter(writer_path)
            print(f"tensorboard logging to {writer_path}")

            results = {"generation": [],\
                    "fitness": [],\
                    "fitness_max": [],\
                    "fitness_min": [],\
                    "fitness_mean": [],\
                    "fitness std. dev.": []}

            tag = str(int(time.time()))
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
                        
                        if (agent.meta_index % agent.population_size) == 0:
                            my_rules = np.random.choice(training_rules, \
                                    p=[1/len(training_rules)]*len(training_rules))

                            env.rules_from_string(my_rules)

                        number_steps = 0
                        reward_sum.append(np.sum(rewards.detach().cpu().numpy()))
                        agent.step(reward_sum[-1])

                        obs = env.reset()
                        rewards = torch.Tensor([]).to(my_device)

                    action = agent(obs)

                    obs, reward, done, info = env.step(action)

                    rewards = torch.cat([rewards, reward])
                    number_steps += 1

                t1 = time.time()

                results["generation"].append(generation)
                results["fitness_max"].append(np.max(reward_sum))
                results["fitness_min"].append(np.min(reward_sum))
                results["fitness_mean"].append(np.mean(reward_sum))
                results["fitness std. dev."].append(np.std(reward_sum))

                # training summary writer adds

                max_fit = np.max(reward_sum)
                mean_fit = np.mean(reward_sum)
                min_fit = np.min(reward_sum)
                std_dev_fit = np.std(reward_sum)

                writer.add_scalar("max_fit/train", max_fit, generation)
                writer.add_scalar("mean_fit/train", mean_fit, generation)
                writer.add_scalar("min_fit/train", min_fit, generation)
                writer.add_scalar("std_dev_fit/train", std_dev_fit, generation)

                print(f"generation {generation}, mean, max, min, std. dev. fitness: "\
                         f"{mean_fit}, {max_fit}, "\
                         f"{min_fit}, {std_dev_fit}")

                steps_per_second = (env.inner_env.instances*max_steps*agent.population_size)/(t1-t0)
                print(f"steps per second = {steps_per_second} s per generation: {t1-t0}")

                if generation % 16 == 0:

                    rewards = torch.Tensor([]).to(my_device)
                    reward_sum = []
                    number_steps = 0
                    agent_count = 0

                    for my_rules in validation_rules:
                        agent.fitness = []

                        env.rules_from_string(my_rules)

                        while agent_count < agent.population_size:

                            if number_steps >= max_steps:

                                agent_count += 1
                                agent.meta_index = agent_count
                                
                                number_steps = 0
                                reward_sum.append(np.sum(rewards.detach().cpu().numpy()))

                                obs = env.reset()
                                rewards = torch.Tensor([]).to(my_device)

                            action = agent(obs)

                            obs, reward, done, info = env.step(action)

                            rewards = torch.cat([rewards, reward])

                            number_steps += 1

                    # validation summary writer adds
                    max_fit = np.max(reward_sum)
                    mean_fit = np.mean(reward_sum)
                    min_fit = np.min(reward_sum)
                    std_dev_fit = np.std(reward_sum)

                    writer.add_scalar("max_fit/val", max_fit, generation)
                    writer.add_scalar("mean_fit/val", mean_fit, generation)
                    writer.add_scalar("min_fit/val", min_fit, generation)
                    writer.add_scalar("std_dev_fit/val", std_dev_fit, generation)

                    print(f"{generation} validation, mean, max, min, std. dev. fitness: "\
                            f"{mean_fit}, {max_fit}, {min_fit}, {std_dev_fit}")
                    np.save(f"{writer_path}_results{tag}.npy", results, allow_pickle=True)


if __name__ == "__main__":

    train()
