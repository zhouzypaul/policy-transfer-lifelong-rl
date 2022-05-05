import os
import argparse
from pathlib import Path

import pfrl
import numpy as np

from skills import utils
from skills.option_utils import SingleOptionTrial
from skills.ensemble.train import train_ensemble_agent
from skills.agents.ensemble import EnsembleAgent


class TransferTrial(SingleOptionTrial):
    """
    load a trained agent, and try to retrain it on another starting spot
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
        parser.add_argument("--load", type=str, required=True,
                            help="the experiment_name of the trained agent so we know where to look for loading it")
        
        # testing params
        parser.add_argument("--steps", type=int, default=1000000,
                            help="max number of steps to train the agent for")

        args = self.parse_common_args(parser)
        return args
    
    def check_params_validity(self):
        assert self.params['start_state'] is not None

    def setup(self):
        self.check_params_validity()
        
        # setting random seeds
        pfrl.utils.set_random_seed(self.params['seed'])

        # get the hyperparams
        hyperparams_file = Path(self.params['results_dir']) / self.params['load'] / 'hyperparams.csv'
        saved_params = utils.load_hyperparams(hyperparams_file)

        # create the saving directories
        self.saving_dir = Path(self.params['results_dir']).joinpath(self.params['experiment_name'])
        utils.create_log_dir(self.saving_dir, remove_existing=True)
        self.params['plots_dir'] = os.path.join(self.saving_dir, 'plots')
        os.mkdir(self.params['plots_dir'])
        self.params['saving_dir'] = self.saving_dir

        # env
        self.env = self.make_env(saved_params['environment'], saved_params['seed'])

        # agent
        agent_file = Path(self.params['results_dir']) / self.params['load'] / 'agent.pkl'
        self.agent = EnsembleAgent.load(agent_file, plot_dir=self.params['plots_dir'], reset=True)
    
    def run(self):
        """
        test the loaded agent
        """
        train_ensemble_agent(
            self.agent,
            self.env,
            max_steps=self.params['steps'],
            saving_dir=self.saving_dir,
            success_rate_save_freq=self.params['success_rate_save_freq'],
            reward_save_freq=self.params['reward_logging_freq'],
            agent_save_freq=self.params['saving_freq'],
            success_threshold_for_stopping=self.params['success_threshold_for_stopping'],
        )


def main():
    trial = TransferTrial()
    trial.run()


if __name__ == '__main__':
    main()
