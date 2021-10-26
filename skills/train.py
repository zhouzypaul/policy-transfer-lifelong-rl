import time
import pickle
import random
import os
import argparse
import shutil
from pathlib import Path

import torch
import seeding
import numpy as np

from skills import utils
from skills.option_utils import SingleOptionTrial, make_done_state_plot
from skills.option import Option
from skills.plot import main as plot_learning_curve


class TrainOptionTrial(SingleOptionTrial):
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
        parser.add_argument("--goal_state_agent_space", type=str, default="middle_ladder_bottom_agent_space.npy",
                            help="a file in info_dir that store the agent space image of agent in goal state")
        parser.add_argument("--goal_state_pos", type=str, default="middle_ladder_bottom_pos.txt",
                            help="a file in info_dir that store the x, y coordinates of goal state")
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
        self.env = self.make_env(self.params['environment'], self.params['seed'])
        if self.params['agent_space']:
            goal_state_path = self.params['info_dir'].joinpath(self.params['goal_state_agent_space'])
        else:
            goal_state_path = self.params['info_dir'].joinpath(self.params['goal_state'])
        goal_state_pos_path = self.params['info_dir'].joinpath(self.params['goal_state_pos'])
        self.params['goal_state'] = np.load(goal_state_path)
        self.params['goal_state_position'] = np.loadtxt(goal_state_pos_path)
        print(f"aiming for goal location {self.params['goal_state_position']}")

        # setup global option and the only option that needs to be learned
        self.option = Option(name='only-option', 
                            env=self.env, 
                            gestation_period=self.params['gestation_period'],
                            buffer_length=self.params['buffer_length'],
                            goal_state=self.params['goal_state'],
                            goal_state_position=self.params['goal_state_position'],
                            epsilon_within_goal=self.params['epsilon_within_goal'],
                            death_reward=self.params['death_reward'],
                            goal_reward=self.params['goal_reward'],
                            step_reward=self.params['step_reward'],
                            max_episode_len=self.params['max_episode_len'],
                            policy_net_lr=self.params['lr'],
                            policy_net_final_epsilon=self.params['final_epsilon'],
                            policy_net_final_exploration_frames=self.params['final_exploration_frames'],
                            policy_net_replay_start_size=self.params['replay_start_size'],
                            policy_net_target_update_interval=self.params['target_update_interval'],
                            policy_net_update_interval=self.params['update_interval'],
                            saving_dir=self.saving_dir,
                            seed=self.params['seed'],
                            logging_frequency=self.params['logging_frequency'],
                            device=self.params['device'])

    def train_option(self):
        """
        run the actual experiment to train one option
        """
        start_time = time.time()
        
        # create the option
        step_number = 0
        episode_idx = 0
        while self.option.get_training_phase() == "gestation":
            print(f"starting episode {episode_idx} at step {step_number}")
            use_directed_rollout = True if episode_idx < 50 else False
            if use_directed_rollout:
                print("directed rolling out episode")
            option_transitions, total_reward = self.option.rollout(
                step_number=step_number, 
                directed_rollout=use_directed_rollout,
                eval_mode=False, 
                rendering=self.params['render']
            )
            step_number += len(option_transitions)
            episode_idx += 1

            # plot the done state
            done_state_dir = Path(self.saving_dir).joinpath('done_states_plot')
            done_state_dir.mkdir(exist_ok=True)
            if not use_directed_rollout:
                make_done_state_plot(replay_buffer=option_transitions, episode_idx=episode_idx, save_dir=done_state_dir)

            # save the results
            if episode_idx % self.params['saving_frequency'] == 0:
                success_curves_file_name = 'success_curves.pkl'
                self.save_results()
                plot_learning_curve(self.params['experiment_name'], log_file_name=success_curves_file_name)

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
    trial = TrainOptionTrial()
    trial.train_option()


if __name__ == "__main__":
    main()
