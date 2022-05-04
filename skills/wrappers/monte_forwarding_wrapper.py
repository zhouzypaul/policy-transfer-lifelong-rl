from pathlib import Path

import numpy as np
from gym import Wrapper

from skills.option_utils import set_player_ram


class MonteForwarding(Wrapper):
	"""
	forwards the agent to another state when the agent starts
	this just overrides the reset method and make it start in another position
	"""
	def __init__(self, env, forwarding_target: Path):
		"""
		forward the agent to start in state `forwarding_target`
		Args:
			forwarding_target: a previously saved .npy file that contains the encoded start state ram
		"""
		super().__init__(env)
		self.env = env
		self.target_ram = np.load(forwarding_target)
	
	def reset(self):
		self.env.reset()
		obs = set_player_ram(self.env, self.target_ram)
		return obs