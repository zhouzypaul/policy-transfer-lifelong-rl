import os
import csv
import argparse
from pathlib import Path

import pfrl
import numpy as np
from matplotlib import pyplot as plt

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
        parser.add_argument("--load", "-l", type=str, required=True,
                            help="the experiment_name of the trained agent so we know where to look for loading it")
        parser.add_argument("--target", "-t", type=str, required=True, 
                            nargs='+', default=[],
                            help="a list of target start_state to transfer to")
        parser.add_argument("--plot", "-p", action='store_true',
                            help="only do the plotting. Use this after the agent has been trained on transfer tasks.")
        
        # testing params
        parser.add_argument("--steps", type=int, default=50000,
                            help="max number of steps to train the agent for")

        args = self.parse_common_args(parser)
        return args

    def _set_experiment_name(self):
        """
        the experiment name shall the the combination of the loading state as well as all the transfer targets
        """
        connector = '->'
        exp_name = self.params['load']
        for target in self.params['target']:
            exp_name += connector + target
        self.params['experiment_name'] = exp_name
    
    def check_params_validity(self):
        # check that all the target start_states are valid
        for target_path in self.params['target']:
            start_state_path = self.params['ram_dir'].joinpath(target_path + '.npy')
            if not os.path.exists(start_state_path):
                raise FileNotFoundError(f"{target_path} does not exist")
        print(f"Targetting {len(self.params['target'])} transfer targets: {self.params['target']}")
        # set experiment name
        self._set_experiment_name()
        # log more frequently because it takes less time to train
        self.params['reward_logging_freq'] = 100
        self.params['success_rate_save_freq'] = max(1, int(self.params['steps'] / 200))

    def setup(self):
        self.check_params_validity()
        
        # setting random seeds
        pfrl.utils.set_random_seed(self.params['seed'])

        # get the hyperparams
        hyperparams_file = Path(self.params['results_dir']) / self.params['load'] / 'hyperparams.csv'
        self.saved_params = utils.load_hyperparams(hyperparams_file)

        # create the saving directories
        self.saving_dir = Path(self.params['results_dir']).joinpath(self.params['experiment_name'])
        if self.params['plot']:
            utils.create_log_dir(self.saving_dir, remove_existing=False)
        else:
            utils.create_log_dir(self.saving_dir, remove_existing=True)
        self.params['saving_dir'] = self.saving_dir

    def plot_results(self):
        """
        just plot the results after the agent has been trained on transfer tasks
        """
        plot_when_well_trained(self.params['target'], self.saving_dir)
        plot_average_success_rate(self.params['target'], self.saving_dir)
    
    def transfer(self):
        """
        sequentially train the agent on each of the targets
        loaded agent -> first target state
        first target state trained -> second target state
        second target state trained -> third target state
        ...
        """
        # training
        trained = self.params['load']
        for i, target in enumerate(self.params['target']):
            print(f"Training {trained} -> {target}")
            # make env
            env = self.make_env(self.saved_params['environment'], self.saved_params['seed'], start_state=target)
            # find loaded agent
            if trained == self.params['load']:
                agent_file = Path(self.params['results_dir']) / self.params['load'] / 'agent.pkl'
            else:
                agent_file = sub_saving_dir / 'agent.pkl'
            # make saving dir
            exp_name = trained + '->' + target
            sub_saving_dir = self.saving_dir.joinpath(exp_name)
            sub_saving_dir.mkdir()
            # make agent
            plots_dir = sub_saving_dir / 'plots'
            plots_dir.mkdir()
            agent = EnsembleAgent.load(agent_file, plot_dir=plots_dir)
            # train
            train_ensemble_agent(
                agent,
                env,
                max_steps=self.params['steps'],
                saving_dir=sub_saving_dir,
                success_rate_save_freq=self.params['success_rate_save_freq'],
                reward_save_freq=self.params['reward_logging_freq'],
                agent_save_freq=self.params['saving_freq'],
            )
            # advance to next target
            trained = target
        
        # meta learning statistics
        self.plot_results()
    
    def run(self):
        if self.params['plot']:
            self.plot_results()
        else:
            self.transfer()


def plot_when_well_trained(targets, saving_dir):
    steps_when_well_trained = np.zeros(len(targets))
    episode_when_well_trained = np.zeros(len(targets))

    # descend into the sub saving dirs to find the well_trained csv file
    for subdir in os.listdir(saving_dir):
        if not os.path.isdir(saving_dir.joinpath(subdir)):
            continue
        well_trained_file = saving_dir / subdir / 'finish_training_time.csv'
        with open(well_trained_file, 'r') as f:
            csv_reader = csv.reader(f)
            data = list(csv_reader)[-1]
            episode = int(data[0])
            step = int(data[1])
            target = subdir.split('->')[1]
            steps_when_well_trained[targets.index(target)] = step
            episode_when_well_trained[targets.index(target)] = episode

    steps_file = saving_dir / 'steps_when_well_trained.png'
    plt.plot(steps_when_well_trained)
    plt.xticks(range(len(targets)), targets)
    plt.xlabel('target')
    plt.ylabel('steps till skill is well trained')
    plt.savefig(steps_file)
    plt.close()

    episode_file = saving_dir / 'episode_when_well_trained.png'
    plt.plot(episode_when_well_trained)
    plt.xticks(range(len(targets)), targets)
    plt.xlabel('target')
    plt.ylabel('episode till skill is well trained')
    plt.savefig(episode_file)
    plt.close()


def plot_average_success_rate(targets, saving_dir):
    average_success_rates = np.zeros(len(targets))
    # descend into the sub saving dirs to find the success rates file
    for subdir in os.listdir(saving_dir):
        if not os.path.isdir(saving_dir.joinpath(subdir)):
            continue
        success_rates_file = Path(saving_dir) / subdir / 'success_rate.csv'
        with open(success_rates_file, 'r') as f:
            csv_reader = csv.reader(f)
            success_rates = [float(row[1]) for row in csv_reader]
            avg_success_rate = np.mean(success_rates)
            target = subdir.split('->')[-1]
            average_success_rates[targets.index(target)] = avg_success_rate
    # plot
    plt.plot(average_success_rates)
    plt.xticks(range(len(targets)), targets)
    plt.xlabel('target')
    plt.ylabel('average success rate')
    plt.savefig(saving_dir / 'average_success_rate.png')
    plt.close()


def main():
    trial = TransferTrial()
    trial.run()


if __name__ == '__main__':
    main()
