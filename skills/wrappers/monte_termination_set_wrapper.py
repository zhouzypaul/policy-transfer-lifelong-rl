import cv2
import torch
import numpy as np
from gym import Wrapper

from skills.classifiers.portable_set import EnsembleClassifier
from skills.wrappers.agent_wrapper import crop_agent_space


class MonteTerminationSetWrapper(Wrapper):
    """
    a wrapper that uses the portable EnsembleClassifier to determine whether a skill is done or not
    """
    def __init__(self, env, confidence_based_reward=False):
        """
        when using confidence_based_reward, the reward received when done is exactly the confidence of the 
        termination classifier
        else the reward would be either 1 or 0
        """
        super().__init__(env)
        self.env = env
        self.confidence_based_reward = confidence_based_reward
        # load saved classifier
        clf_path = 'resources/classifiers'  # hard coded for now
        self.clf = EnsembleClassifier(device='cuda')
        self.clf.load(clf_path)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        tensor_next_state = torch.from_numpy(np.array(next_state)).float().unsqueeze(0)  # add batch dimension
        try:
            assert tensor_next_state.shape == (1, 4, 56, 40)  # make sure it's agent space observation
        except AssertionError:
            # policy is in problem space, get the agent space input here
            # build the frame stack
            agent_space_next_state = np.zeros((1, 4, 56, 40))
            for i, frame in enumerate(self.env.unwrapped.original_stacked_frames):
                player_pos = self.env.unwrapped.stacked_agent_position[i]
                obs = crop_agent_space(frame, player_pos)
                assert obs.shape == (56, 40, 3)
                obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
                obs = np.expand_dims(obs, axis=0)  # add channel dimension, (1, 56, 40)
                agent_space_next_state[:, i, :, :] = obs
            tensor_next_state = torch.from_numpy(np.array(agent_space_next_state)).float()
            assert tensor_next_state.shape == (1, 4, 56, 40), tensor_next_state.shape  # make sure it's agent space observation
        votes, vote_confs = self.clf.get_votes(tensor_next_state)
        # aggregate the votes, vote yes if one of them is yes
        done = np.sum(votes) > 0  # votes are all either 0 or 1
        if self.confidence_based_reward and done:
            reward = vote_confs[np.argmax(votes==1)]
        # override needs_real_reset for EpisodicLifeEnv
        self.env.unwrapped.needs_real_reset = done or info.get("needs_reset", False)
        return next_state, reward, done, info
