import os
import sys
import time
import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from carle.env import CARLE
from carle.mcl import AE2D, \
        PredictionBonus, \
        SurpriseBonus, \
        RND2D, \
        CornerBonus, \
        PufferDetector, \
        SpeedDetector, \
        MorphoBonus


from game_of_carle.agents.toggle import Toggle
from game_of_carle.agents.grnn import ConvGRNN
from game_of_carle.agents.carla import CARLA 
from game_of_carle.agents.harli import HARLI 
from game_of_carle.algos.cma import CMAPopulation
from game_of_carle.algos.ges import GESPopulation

WRAPPER_DICT = { \
        "predictionbonus": PredictionBonus, \
        "surprisebonus": SurpriseBonus, \
        "cornerbonus": CornerBonus, \
        "ae2d": AE2D, \
        "rnd2d": RND2D, \
        "pufferdetector": PufferDetector, \
        "speeddetector": SpeedDetector, \
        "morphobonus": MorphoBonus \
        }

AGENT_DICT = { \
        "harli": HARLI, \
        "carla": CARLA, \
        "convgrnn": ConvGRNN, \
        "toggle": Toggle \
        }

ALGO_DICT = { \
        "cmapopulation": CMAPopulation, \
        "gespopulation": GESPopulation\
        }

def train(args): 
    
    # parse arguments
    episodes = args.episodes
    max_generations =  args.max_generations
    max_steps = args.max_steps
    population_size = args.population_size
    selection_mode = args.selection_mode
    my_instances = args.vectorization
    seeds = args.seeds
    device = args.device
    use_grad = args.use_grad
    env_dimension = args.env_dimension
    agents = [AGENT_DICT[key.lower()] for key in args.agents]
    wrappers = [WRAPPER_DICT[key.lower()] for key in args.wrappers]
    training_rules = args.training_rules
    validation_rules = args.validation_rules
    testing_rules = args.testing_rules

    algo = ALGO_DICT[args.algorithm.lower()]


    if use_grad:
        print(f"use grad == {use_grad} is true")

    # define environment and exploration bonus wrappers
    env = CARLE(instances = my_instances, device=device, \
            height=env_dimension, width=env_dimension)

    my_device = env.my_device

    for wrapper in wrappers:
        env = wrapper(env)

    for my_seed in seeds:

        np.random.seed(my_seed)
        torch.manual_seed(my_seed)

        for agent_fn in agents:

            time_stamp = int(time.time())

            agent = algo(agent_fn, device=device, \
                    episodes=episodes, \
                    population_size=population_size, \
                    selection_mode=selection_mode, \
                    use_grad=use_grad)

            tag = args.tag + str(int(time.time()))
            experiment_name = agent.population[0].__class__.__name__ + \
                    f"_{my_seed}_{tag}"
            my_file_path = os.path.abspath(os.path.dirname(__file__))
            my_directory_path = os.path.join(my_file_path, "../policies/")
            my_save_path = os.path.join(my_directory_path, experiment_name)

            my_meta_path = os.path.join(\
                    os.path.sep.join(my_file_path.split(os.path.sep)[:-1]), \
                    "experiments", f"args_{experiment_name}")

            with open(my_meta_path, "w") as f:
                my_module = sys.argv[0].split(os.path.sep)[-1]
                f.write(my_module)
                for my_arg in sys.argv[1:]:
                    f.write(f" {my_arg} ")

            agent.save_path = my_save_path
            agent.population_size = population_size

            writer_path = os.path.sep.join(my_file_path.split(os.path.sep)[:-1])
            writer_path = os.path.join(writer_path, f"experiments/logs/{experiment_name}")
            
            writer = SummaryWriter(writer_path)
            print(f"tensorboard logging to {writer_path}")

            results = {"generation": [],\
                    "fitness": [],\
                    "fitness_max": [],\
                    "fitness_min": [],\
                    "fitness_mean": [],\
                    "fitness std. dev.": []}


            my_rules = np.random.choice(training_rules, \
                    p=[1/len(training_rules)]*len(training_rules))
            env.rules_from_string(my_rules)

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
                        reward_sum.append(np.mean(rewards.detach().cpu().numpy()))

                        agent.step(reward_sum[-1])

                        obs = env.reset()
                        rewards = torch.Tensor([]).to(my_device)
                    
                    action = agent(obs)

                    obs, reward, done, info = env.step(action)

                    #agent.population[agent.meta_index].step(reward)

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
                         f"{mean_fit:.4}, {max_fit:.4}, "\
                         f"{min_fit:.4}, {std_dev_fit:.4}")

                steps_per_second = (agent.episodes * \
                        env.inner_env.instances \
                        * max_steps*agent.population_size) \
                        /(t1-t0)
                print(agent.episodes)
                print(f"steps per second = {steps_per_second:.4} s per generation: {(t1-t0):.4}")

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
                                reward_sum.append(np.mean(rewards.detach().cpu().numpy()))

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

    parser = argparse.ArgumentParser()

    parser.add_argument("-mg", "--max_generations", type=int, default=10)
    parser.add_argument("-ms", "--max_steps", type=int, default=512)
    parser.add_argument("-p", "--population_size", type=int, default=16)
    parser.add_argument("-sm", "--selection_mode", type=int, default=1) 
    parser.add_argument("-v", "--vectorization", type=int, default=1)
    parser.add_argument("-s", "--seeds", type=int, nargs="+", default=[13])
    parser.add_argument("-e", "--episodes", type=int, default=1)
    parser.add_argument("-d", "--device", type=str, default="cuda:1")
    parser.add_argument("-u", "--use_grad", dest="use_grad",\
            action="store_true")

    parser.add_argument("-dim", "--env_dimension", type=int, default=256)
    parser.add_argument("-a", "--agents", type=str, nargs="+", \
            default=["Toggle", "HARLI"], \
            help="agent(s) to train in experiment, can be several")
    parser.add_argument("-g", "--algorithm", type=str, \
            default="CMAPopulation", \
            help="algorithm to use for training")
    parser.add_argument("-w", "--wrappers", type=str, nargs="+", \
            default=["CornerBonus"], help="reward wrappers to train with")
    parser.add_argument("-tr", "--training_rules", type=str, nargs="+", \
            default=["B3/S23"], \
            help="B/S string(s) defining CA rules to use during training")
    parser.add_argument("-vr", "--validation_rules", type=str, nargs="+", \
            default=["B3/S23"], \
            help="B/S string(s) defining CA rules to use during validation")
    parser.add_argument("-xr", "--testing_rules", type=str, nargs="+", \
            default=["B3/S23"], \
            help="B/S string(s) defining CA rules to use during testing")
    parser.add_argument("-tag", "--tag", type=str, default="default_tag", \
            help="a tag to identify the experiment")

    parser.set_defaults(use_grad=False)

    args = parser.parse_args()

    train(args)
