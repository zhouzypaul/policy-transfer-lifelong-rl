import numpy as np
from gym import Wrapper

from skills.option_utils import get_player_position


class MonteNewGoalWrapper(Wrapper):
    """
    define a goal in Monte to faciliate the training of one skill
    when the goal is hit, done will be true and the reward will be 1
    """
    def __init__(self, env, goal_pos, epsilon_tol=4):
        """
        define a new goal in Monte for dense reward
        Args:
            goal_pos: (x, y)
            epsilon_tol: tolerance of nearness to goal, count as within goal 
                            if inside this epsilon ball to the goal
        """
        super().__init__(env)
        self.env = env
        self.goal_pos = np.array(goal_pos)
        self.epsilon_tol = epsilon_tol
    
    def step(self, action):
        """
        override done and reward
        """
        next_state, reward, done, info = self.env.step(action)
        player_pos = get_player_position(self.env.unwrapped.ale.getRAM())
        if np.linalg.norm(np.array(player_pos) - self.goal_pos) < self.epsilon_tol:
            reward = 1
            done = True
        else:
            reward = 0  # override reward, such as when got key
        return next_state, reward, done, info
