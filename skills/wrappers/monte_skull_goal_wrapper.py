import os

import numpy as np
from gym import Wrapper

from skills.option_utils import get_player_position, get_player_room_number, get_skull_position


room_to_skull_y = {
    1: 148,
}


class MonteSkullGoalWrapper(Wrapper):
    """
    for training a "jump over skull" skill.
    The agent finishes the skill if its y pos aligns with the floor of the skull and 
    its x pos is on the other side of the skull.

    currently, default to the player starts on the right side of the skull, and try to jump to the left of it
    """
    def __init__(self, env, epsilon_tol=6):
        """
        Args:
            epsilon_tol: the agent dies within 6 of the skull (according to pix2sim)
        """
        super().__init__(env)
        self.env = env
        self.epsilon_tol = epsilon_tol
        self.room_number = get_player_room_number(self.env.unwrapped.ale.getRAM())
        self.y = room_to_skull_y[self.room_number]
    
    def step(self, action):
        """
        override done and reward
        """
        next_state, reward, done, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()
        player_x, player_y = get_player_position(ram)
        skull_x = get_skull_position(ram)
        if player_y == self.y and player_x < skull_x - self.epsilon_tol:
            done = True
            reward = 1
        else:
            reward = 0  # override reward, such as when got key
        # override needs_real_reset for EpisodicLifeEnv
        self.env.unwrapped.needs_real_reset = done or info.get("needs_reset", False)
        return next_state, reward, done, info
