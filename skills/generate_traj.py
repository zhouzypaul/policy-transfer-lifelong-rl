import os
import random
import pickle
import argparse

import seeding
import numpy as np
from matplotlib import pyplot as plt

from skills import utils
from skills.option_utils import SingleOptionTrial
from skills.ale_utils import get_player_position


class GenerateTrajectory(SingleOptionTrial):
    """
    use the class to generate trajectories using manually-inputted actions, and
    save the trajectory to a file
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
        parser.add_argument("--file_name", type=str,
                            help="name of the output file")
        args = self.parse_common_args(parser)
        return args
    
    def setup(self):
        # override params
        self.params['experiment_name'] = 'trajectories'
        self.params['suppress_action_prunning'] = True

        # setting random seeds
        seeding.seed(self.params['seed'], np)

        # saving
        self.saving_dir = os.path.join(self.params['results_dir'], self.params['experiment_name'])
        self.saving_file = os.path.join(self.saving_dir, self.params['file_name'])
        self.random_traj_file = os.path.join(self.saving_dir, 'random_traj.pkl')
        self.plot_file = os.path.join(self.saving_dir, 'random_traj_positions.png')
        utils.create_log_dir(self.saving_dir, remove_existing=True)

        # make env
        self.env = self.make_env(self.params['environment'], self.params['seed'], self.params['start_state'])
    
    def generate_traj(self):
        """
        generate trajectories by human control
        """
        print([(i, meaning) for i, meaning in enumerate(self.env.unwrapped.get_action_meanings())])
        meaning_to_action = {meaning.lower(): i for i, meaning in enumerate(self.env.unwrapped.get_action_meanings())}
        state = self.env.reset()
        trajectories = []
        trajectory = []
        try:
            while True:
                # render env
                self.env.unwrapped.render()

                # user input an action to take
                action_input = input()
                if action_input.isnumeric():
                    action = int(action_input)
                else:
                    action = meaning_to_action[action_input.lower()]
                    print(action)

                # record the state and ram
                ram = self.env.unwrapped.ale.getRAM()
                trajectory.append((np.array(state), ram))

                # take the action
                next_state, r, done, info = self.env.step(action)
                state = next_state
                if done:
                    state = self.env.reset()
                    trajectories.append(trajectory)
                    print(f"finished a trajectory, length: {len(trajectory)}")
                    trajectory = []
                    self.save_traj(trajectories, file_path=self.saving_file)
        except:
            # save on error
            trajectories.append(trajectory)
            self.save_traj(trajectories, file_path=self.saving_file)
    
    def generate_random_traj(self, total_steps=10000):
        """
        generate random trajectory with random control
        """
        state = self.env.reset()
        trajectories = []
        trajectory = []
        step_idx = 0
        while step_idx < total_steps:
            # render env
            self.env.unwrapped.render()

            # record the state and ram
            ram = self.env.unwrapped.ale.getRAM()
            trajectory.append((np.array(state), ram))

            # take the action
            action = random.randint(0, self.env.action_space.n - 1)
            next_state, r, done, info = self.env.step(action)
            state = next_state
            if done:
                state = self.env.reset()
                trajectories.append(trajectory)
                print(f"finished a trajectory, length: {len(trajectory)}")
                trajectory = []
            step_idx += 1
        trajectories.append(trajectory)
        self.save_traj(trajectories, file_path=self.random_traj_file)
        self.plot_traj_positions(trajectories)
    
    def plot_traj_positions(self, trajectories):
        """
        plot the position of all the positons in a certain list of trajectories
        """
        all_pos = [get_player_position(ram) for traj in trajectories for state, ram in traj]
        plt.scatter(*zip(*all_pos))
        plt.xlim(15, 140)
        plt.ylim(140, 260)
        plt.title('player positions')
        plt.show()
        plt.savefig(self.plot_file)
        plt.close()
    
    def save_traj(self, trajectories, file_path):
        """
        save a list of trajectories
        """
        with open(file_path, 'wb') as f:
            pickle.dump(trajectories, f)


def main():
    game = GenerateTrajectory()
    game.generate_random_traj()
    game.generate_traj()


if __name__ == "__main__":
    main()
