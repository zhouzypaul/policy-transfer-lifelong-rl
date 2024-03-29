import torch
import numpy as np
from gym import Wrapper
from skimage import color

from skills.classifiers.portable_set import EnsembleClassifier
from skills.wrappers.agent_wrapper import crop_agent_space


class MonteTerminationSetWrapper(Wrapper):
    """
    a wrapper that uses the portable EnsembleClassifier to determine whether a skill is done or not
    """
    def __init__(self, env, confidence_based_reward=False, device="cuda"):
        """
        when using confidence_based_reward, the reward received when done is exactly the confidence of the 
        termination classifier
        else the reward would be either 1 or 0
        """
        super().__init__(env)
        self.env = env
        self.confidence_based_reward = confidence_based_reward
        # load saved classifier
        clf_path = 'resources/classifier'  # hard coded for now
        self.clf = EnsembleClassifier(device=device)
        self.clf.load(clf_path)

    def step(self, action):
        """
        override done and reward
        """
        next_state, reward, done, info = self.env.step(action)
        # get the agent space input here
        # build the frame stack
        agent_space_next_state = np.zeros((1, 4, 56, 40))
        for i, frame in enumerate(self.env.unwrapped.original_stacked_frames):
            player_pos = self.env.unwrapped.stacked_agent_position[i]
            obs = crop_agent_space(frame, player_pos)
            assert obs.shape == (56, 40, 3)
            obs = color.rgb2gray(obs)  # also concerts to floats
            obs = np.expand_dims(obs, axis=0)  # add channel dimension, (1, 56, 40)
            agent_space_next_state[:, i, :, :] = obs
        tensor_next_state = torch.from_numpy(np.array(agent_space_next_state)).float()
        assert tensor_next_state.shape == (1, 4, 56, 40), tensor_next_state.shape  # make sure it's agent space observation
        votes, vote_confs = self.clf.get_votes(tensor_next_state)
        # aggregate the votes, vote yes if one of them is yes
        done = np.sum(votes) > 0  # votes are all either 0 or 1
        reward = 1 if done else 0
        if self.confidence_based_reward and done:
            reward = vote_confs[np.argmax(votes==1)]
        # override needs_real_reset for EpisodicLifeEnv
        self.env.unwrapped.needs_real_reset = done or info.get("needs_reset", False)
        return next_state, reward, done, info
