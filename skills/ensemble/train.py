import time
import pickle
import random
import os
import csv
import argparse
import shutil
from collections import deque

import torch
import seeding
import numpy as np
import pfrl
from matplotlib import pyplot as plt

from skills import utils
from skills.agents.ensemble import EnsembleAgent
from skills.option_utils import SingleOptionTrial


class TrainEnsembleOfSkills(SingleOptionTrial):
    """
    a class for running experiments to train an option
    """
    def __init__(self):
        super().__init__()
        args = self.parse_args()
        self.params = self.load_hyperparams(args)
        self.setup()

    def parse_args(self):
        """
        parse the inputted argument
        """
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            parents=[self.get_common_arg_parser()]
        )
        # goal state
        parser.add_argument("--goal_state", type=str, default="middle_ladder_bottom.npy",
                            help="a file in info_dir that stores the image of the agent in goal state")
        parser.add_argument("--goal_state_pos", type=str, default="middle_ladder_bottom_pos.txt",
                            help="a file in info_dir that store the x, y coordinates of goal state")

        # ensemble
        parser.add_argument("--num_policies", type=int, default=8,
                            help="number of policies in the ensemble")
        
        # training
        parser.add_argument("--steps", type=int, default=100000,
                            help="number of training steps")
        parser.add_argument("--explore_epsilon", type=float, default=0.1,
                            help="epsilon for epsilon-greedy exploration")
        parser.add_argument("--warmup_steps", type=int, default=1024,
                            help="number of steps for warming up before updating the network")
        parser.add_argument("--epochs_per_step", type=int, default=1,
                            help="how many epochs to train the embedding the policy network per step in environment")
        parser.add_argument("--batch_size", type=int, default=64,
                            help="batch size for training")
        parser.add_argument("--saving_freq", type=int, default=5000,
                            help="how often to save the trained model")
        parser.add_argument("--q_target_update_interval", type=int, default=40,
                            help="how often to update the target network in number of steps")
        
        parser.add_argument("--verbose", action="store_true", default=False,
                            help="whether to print the training loss")
        args = self.parse_common_args(parser)
        return args

    def check_params_validity(self):
        """
        check whether the params entered by the user is valid
        """
        pass
    
    def setup(self):
        """
        do set up for the experiment
        """
        self.check_params_validity()

        # setting random seeds
        seeding.seed(self.params['seed'], random, np)
        pfrl.utils.set_random_seed(self.params['seed'])

        # torch benchmark
        torch.backends.cudnn.benchmark = True

        # create the saving directories
        self.saving_dir = os.path.join(self.params['results_dir'], self.params['experiment_name'])
        if os.path.exists(self.saving_dir):  # remove all existing contents
            shutil.rmtree(self.saving_dir)
        utils.create_log_dir(self.saving_dir)
        self.params['saving_dir'] = self.saving_dir
        self.params['plots_dir'] = os.path.join(self.saving_dir, 'plots')
        os.mkdir(self.params['plots_dir'])

        # save the hyperparams
        utils.save_hyperparams(os.path.join(self.saving_dir, "hyperparams.csv"), self.params)

        # set up env and its goal
        if self.params['agent_space']:
            goal_state_path = self.params['info_dir'].joinpath(self.params['goal_state_agent_space'])
        else:
            goal_state_path = self.params['info_dir'].joinpath(self.params['goal_state'])
        goal_state_pos_path = self.params['info_dir'].joinpath(self.params['goal_state_pos'])
        self.params['goal_state'] = np.load(goal_state_path)
        self.params['goal_state_position'] = tuple(np.loadtxt(goal_state_pos_path))
        print(f"aiming for goal location {self.params['goal_state_position']}")
        self.env = self.make_env(self.params['environment'], self.params['seed'], goal=self.params['goal_state_position'])

        # set up ensemble
        self.ensemble_agent = EnsembleAgent(
            device=self.params['device'],
            warmup_steps=self.params['warmup_steps'],
            batch_size=self.params['batch_size'],
            update_interval=4,
            q_target_update_interval=self.params['q_target_update_interval'],
            explore_epsilon=self.params['explore_epsilon'],
            num_modules=self.params['num_policies'],
            num_output_classes=self.env.action_space.n,
            plot_dir=self.params['plots_dir'],
            verbose=self.params['verbose']
        )

        # results
        self.total_reward = 0
        self.success_rates = deque(maxlen=10)

    def train_option(self):
        """
        run the actual experiment to train one option
        """
        start_time = time.time()

        # train loop
        step_number = 0
        episode_number = 0
        state = self.env.reset()
        while step_number < self.params['steps']:
            # action selection: epsilon greedy
            action = self.ensemble_agent.act(state)
            
            # step
            next_state, reward, done, info = self.env.step(action)
            self.ensemble_agent.observe(state, action, reward, next_state, done)
            state = next_state
            if done:
                self.save_success_rate(done and reward == 1, episode_number)
                episode_number += 1
                state = self.env.reset()
            
            self.save_total_reward(reward, step_number)
            self.save_results(step_number)
            step_number += 1

        end_time = time.time()

        print("Time taken: ", end_time - start_time)
    
    def save_success_rate(self, success, episode_number, save_every=1):
        """
        log the average success rate during training every 5 episodes
        the success rate at every episode is the average success rate over the last 10 episodes
        """
        save_file = os.path.join(self.saving_dir, "success_rate.csv")
        img_file = os.path.join(self.saving_dir, "success_rate.png")
        self.success_rates.append(success)
        if episode_number % save_every == 0:
            # write to csv
            open_mode = 'w' if episode_number == 0 else 'a'
            with open(save_file, open_mode) as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([episode_number, np.mean(self.success_rates)])
            # plot it as well
            with open(save_file, 'r') as f:
                reader = csv.reader(f)
                data = np.array([row for row in reader])
                epsidoes = data[:, 0].astype(int)
                success_rates = data[:, 1].astype(np.float32)
                plt.plot(epsidoes, success_rates)
                plt.title("Success rate")
                plt.xlabel("Episode")
                plt.ylabel("Success rate")
                plt.savefig(img_file)
                plt.close()
    
    def save_total_reward(self, r, step_number, save_every=50):
        """
        log the total reward achieved during training every 50 steps
        """
        save_file = os.path.join(self.saving_dir, "total_reward.csv")
        img_file = os.path.join(self.saving_dir, "total_reward.png")
        self.total_reward += r
        if step_number % save_every == 0:
            # write to csv
            open_mode = 'w' if step_number == 0 else 'a'
            with open(save_file, open_mode) as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([step_number, self.total_reward])
            # plot it as well
            with open(save_file, 'r') as f:
                csv_reader = csv.reader(f, delimiter=',')
                data = np.array([row for row in csv_reader])  # (step_number, 2)
                steps = data[:, 0].astype(int)
                total_reward = data[:, 1].astype(np.float32)
                plt.plot(steps, total_reward)
                plt.title("training reward")
                plt.xlabel("steps")
                plt.ylabel("total reward")
                plt.savefig(img_file)
                plt.close()
    
    def save_results(self, step_number):
        """
        save the trained model
        """
        if step_number % self.params['saving_freq'] == 0:
            self.ensemble_agent.save(self.saving_dir)
            print(f"model saved at step {step_number}")


def main():
    trial = TrainEnsembleOfSkills()
    trial.train_option()


if __name__ == "__main__":
    main()
