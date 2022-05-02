import os

import numpy as np
from gym import Wrapper

from skills.option_utils import get_player_position, get_player_room_number


class GoalsCollection:
    def __init__(self, room, file_names, goal_file_dir='resources/monte_info'):
        """
        room: the room number these goals are in
        file_names: a list of filenames of the np arrays that store the goal positions
        goal_file_dir: where the goal files are stored
        """
        self.room = room
        self.goals = [np.loadtxt(os.path.join(goal_file_dir, f)) for f in file_names]
    
    def __len__(self):
        return len(self.goals)
    
    def __contains__(self, item):
        """
        make sure the argument item is a numpy array
        """
        return item in self.goals
    
    def is_within_goal_position(self, room_number, player_pos, tol):
        """
        check that whether the player is within an epsilon tolerance to one of 
        the goal positions
        args:
            player_pos: make sure this is np array: (x, y)
            tol: pixel-wise tolerance from the ground truth goal
        """
        if room_number != self.room:
            return False
        for goal in self.goals:
            if np.linalg.norm(player_pos - goal) < tol:
                return True
        return False


room_to_goals = {
    1: GoalsCollection(1, [
        'middle_ladder_bottom_pos.txt',
        'right_ladder_bottom_pos.txt',
        'left_ladder_bottom_pos.txt',
    ]),
    2: GoalsCollection(2, [
        # TODO:
    ]),
}


class MonteLadderGoalWrapper(Wrapper):
    """
    for training a "go to the bottom of a ladder" skill
    The goals are defined to be the bottom of every ladder in the room the agent
    started out in.
    when the goal is hit, done will be true and the reward will be 1. The other
    default rewards, such as getting a key, are overwritten to be 0.
    """
    def __init__(self, env, epsilon_tol=4):
        """
        Args:
            epsilon_tol: tolerance of nearness to goal, count as within goal 
                            if inside this epsilon ball to the goal
        """
        super().__init__(env)
        self.env = env
        self.epsilon_tol = epsilon_tol
        self.room_number = get_player_room_number(self.env.unwrapped.ale.getRAM())
        self.goal_regions = room_to_goals[self.room_number]
    
    def step(self, action):
        """
        override done and reward
        """
        next_state, reward, done, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()
        player_pos = get_player_position(ram)
        room = get_player_room_number(ram)
        if self.goal_regions.is_within_goal_position(room, player_pos, self.epsilon_tol):
            reward = 1
            done = True
        else:
            reward = 0  # override reward, such as when got key
        return next_state, reward, done, info
