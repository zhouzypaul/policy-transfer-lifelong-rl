import numpy as np
from gym import Wrapper

from skills.classifiers.portable_set import EnsembleClassifier
from skills.wrappers.agent_wrapper import build_agent_space_image_stack


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
        clf_path = 'resources/classifier/termination'  # hard coded for now
        self.clf = EnsembleClassifier(device=device)
        self.clf.load(clf_path)

    def step(self, action):
        """
        override done and reward
        """
        next_state, reward, done, info = self.env.step(action)
        # get the agent space input here
        # build the frame stack
        tensor_next_state = build_agent_space_image_stack(self.env)
        votes, vote_confs = self.clf.get_votes(tensor_next_state)
        # aggregate the votes, vote yes if one of them is yes
        done = np.sum(votes) > 0  # votes are all either 0 or 1
        reward = 1 if done else 0
        if self.confidence_based_reward and done:
            reward = vote_confs[np.argmax(votes==1)]
        # override needs_real_reset for EpisodicLifeEnv
        self.env.unwrapped.needs_real_reset = done or info.get("needs_reset", False)
        return next_state, reward, done, info
