import gym


class ProcgenAgentWrapper(gym.Wrapper):
    """
    include some helpful info in the info dict of how the agent is doing
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['reached_goal'] = done and reward > 0
        info['dead'] = done and reward < 0
        return obs, reward, done, info
