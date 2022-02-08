from gym import Wrapper

from skills.option_utils import get_player_position


class MonteNewGoalWrapper(Wrapper):
    """
    define a goal in Monte to faciliate the training of one skill
    when the goal is hit, done will be true and the reward will be 1
    """
    def __init__(self, env, goal_pos):
        """
        define a new goal in Monte for dense reward
        Args:
            goal_pos: (x, y)
        """
        super().__init__(env)
        self.env = env
        self.goal_pos = goal_pos
    
    def step(self, action):
        """
        override done and reward
        """
        next_state, reward, done, info = self.env.step(action)
        player_pos = get_player_position(self.env.unwrapped.ale.getRAM())
        if player_pos == self.goal_pos:
            reward = 1
            done = True
        return next_state, reward, done, info
