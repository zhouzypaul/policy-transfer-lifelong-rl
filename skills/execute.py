import pickle
import random
import argparse
import os
import shutil

import torch
import seeding
import numpy as np

from skills import utils
from skills.option import Option
from skills.option_utils import SingleOptionTrial


class ExecuteOptionTrial(SingleOptionTrial):
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
        parser.add_argument(
            "--saved_option_dir",
            type=str,
            default='results/right-ladder',
            help='path to a stored trained policy network')
        args = self.parse_common_args(parser)
        return args

    def setup(self):
        """
		do set up for the experiment
		"""
        # setting random seeds
        seeding.seed(self.params['seed'], random, np, torch)

        # set up env and the forwarding target
        self.env = self.make_env(self.params['environment'], self.params['seed'])

        # create the saving directories
        self.saving_dir = os.path.join(self.params['results_dir'], self.params['experiment_name'])
        utils.create_log_dir(self.saving_dir, remove_existing=True)
        self.params['saving_dir'] = self.saving_dir

        # setup global option and the only option that needs to be learned
        self.option = Option(
            name='excute-option',
            env=self.env,
            load_from=self.params['saved_option_dir'],
            gestation_period=None,
            buffer_length=None,
            goal_state=None,
            goal_state_position=None,
            epsilon_within_goal=None,
            death_reward=self.params['death_reward'],
            goal_reward=self.params['goal_reward'],
            step_reward=self.params['step_reward'],
            max_episode_len=self.params['max_episode_len'],
            saving_dir=self.saving_dir,
            seed=self.params['seed'],
            logging_frequency=self.params['logging_frequency'],
            device=self.params['device'],
        )

    def exec_option(self):
        """
		run the actual experiment to train one option
		"""
        step_number = 0
        for _ in range(5):
            print(f"step {step_number}")
            option_transitions, total_reward = self.option.rollout(
                step_number=step_number, eval_mode=True, rendering=self.params['render']
            )
            step_number += len(option_transitions)


def main():
    trial = ExecuteOptionTrial()
    trial.exec_option()


if __name__ == "__main__":
    main()
