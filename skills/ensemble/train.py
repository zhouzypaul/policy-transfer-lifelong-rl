import time
import pickle
import random
import os
import argparse
import shutil

import torch
import seeding
import numpy as np
import pfrl

from skills import utils
from skills.ensemble.policy_ensemble import PolicyEnsemble
from skills.agents.replay_buffer import ReplayBuffer, Transition
from skills.ensemble.aggregate import choose_most_popular
from skills.option_utils import SingleOptionTrial, make_done_state_plot
from skills.plot import main as plot_learning_curve


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
        
        # training
        parser.add_argument("--steps", type=int, default=100000,
                            help="number of training steps")
        parser.add_argument("--explore_epsilon", type=float, default=0.1,
                            help="epsilon for epsilon-greedy exploration")
        parser.add_argument("--warmup_steps", type=int, default=1024,
                            help="number of steps for warming up before updating the network")
        parser.add_argument("--epochs_per_step", type=int, default=1,
                            help="how many epochs to train the embedding the policy network per step in environment")
        parser.add_argument("--batch_size", type=int, default=16,
                            help="batch size for training")
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
        self.replay_buffer = ReplayBuffer(max_memory=10000)
        self.policy_ensemble = PolicyEnsemble(
            device=self.params['device'],
            num_output_classes=self.env.action_space.n,
        )

    def train_option(self):
        """
        run the actual experiment to train one option
        """
        start_time = time.time()

        # train loop
        step_number = 0
        state = self.env.reset()
        while step_number < self.params['steps']:
            # action selection: epsilon greedy
            actions = self.policy_ensemble.predict_actions(torch.from_numpy(np.array(state)).float())
            if random.random() < self.params['explore_epsilon']:
                action = random.randint(0, self.env.action_space.n-1)
            else:
                action = choose_most_popular(actions)
            
            # step
            next_state, reward, done, info = self.env.step(action)
            self.replay_buffer.add(Transition(state, action, reward, next_state, done))
            state = next_state
            if done:
                state = self.env.reset()

            # update
            if len(self.replay_buffer) > self.params['warmup_steps']:
                # sample from replay buffer and split into batches
                dataset = self.replay_buffer.sample(self.params['warmup_steps'])
                dataset = [dataset[i:i+self.params['batch_size']] for i in range(0, len(dataset), self.params['batch_size'])]
                self.policy_ensemble.train_embedding(dataset=dataset, epochs=self.params['epochs_per_step'])
                self.policy_ensemble.train_policy(dataset=dataset, epochs=self.params['epochs_per_step'])

        end_time = time.time()

        print("Time taken: ", end_time - start_time)
    
    def save_results(self, success_curves_file_name='success_curves.pkl'):
        """
        save the results into csv files
        """
        # save success curve
        success_curve_save_path = os.path.join(self.saving_dir, success_curves_file_name)
        with open(success_curve_save_path, 'wb+') as f:
            pickle.dump(self.option.success_rates, f)
        # save trained policy and classifiers
        self.option.policy_net.save(dirname=os.path.join(self.saving_dir, 'saved_model'))
        if self.option.termination_classifier is not None:
            self.option.termination_classifier.save_to_file(os.path.join(self.saving_dir, 'termination_classifier'))
        if self.option.initiation_classifier is not None:
            self.option.initiation_classifier.save_to_file(os.path.join(self.saving_dir, 'initiation_classifier'))


def main():
    trial = TrainEnsembleOfSkills()
    trial.train_option()


if __name__ == "__main__":
    main()
