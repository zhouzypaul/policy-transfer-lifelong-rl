import cv2
import gym
from gym import spaces
import numpy as np


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
    
    def seed(self, seed=None):
        """
        does nothing, because apparently procgen envs don't need it
        they seed when creating the environment
        this is just here to make a common API with atari envs, so code doesn't bug out
        """
        pass


class GrayscaleWrapper(gym.ObservationWrapper):
    def __init__(self, env, grayscale=True, channel_order="chw"):
        """make then images into grayscale and change the channel order of the observation"""
        super().__init__(env)
        self.width = 64
        self.height = 64
        self.grayscale = grayscale
        self.channel_order = channel_order
        num_channels = 1 if grayscale else 3
        shape = {
            'hwc': (self.height, self.width, num_channels),
            'chw': (num_channels, self.height, self.width),
        }
        self.observation_space = spaces.Box(
            low=0, high=255, shape=shape[channel_order], dtype=np.uint8
        )
    
    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return frame.reshape(self.observation_space.low.shape)
        else: 
            order = (0,1,2) if self.channel_order == "hwc" else (2,0,1)
            return frame.transpose(order)
