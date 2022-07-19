import gym


class ProcgenGymWrapper(gym.Wrapper):
    """
    make procgen environments more like gym envs
    """
    def __init__(self, env, agent_space=False):
        super().__init__(env)
        self.agent_space = agent_space

        # rendering
        self.env.viewer = None
        
        # action meanings
        env.unwrapped.get_action_meanings = lambda : [
                "DOWNLEFT",
                "LEFT",
                "UPLEFT",
                "DOWN",
                "",
                "UP",
                "DOWNRIGHT",
                "RIGHT",
                "UPRIGHT",
                "D",
                "A",
                "W",
                "S",
                "Q",
                "E",
        ]
    
    def render(self, mode='human'):
        if self.agent_space:
            raise NotImplementedError
        else:
            img = self.env.render(mode='rgb_array')
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.env.viewer is None:
                self.env.viewer = rendering.SimpleImageViewer()
            self.env.viewer.imshow(img)
            return self.env.viewer.isopen
