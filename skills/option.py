import itertools
import os
from copy import deepcopy
from collections import deque
from pathlib import Path

import numpy as np
from thundersvm import SVC, OneClassSVM
import matplotlib.pyplot as plt

from skills.option_utils import get_player_position, make_chunked_value_function_plot, \
		last_in_framestack
from skills.agents.dqn import make_dqn_agent


class Option:
	"""
	the base class for option that all Option class shoud inherit from
	"""
	def __init__(self, 
				name, 
				env,
				gestation_period,
				buffer_length,
				goal_state,
				goal_state_position,
				epsilon_within_goal,
				death_reward,
				goal_reward,
				step_reward,
				max_episode_len,
				saving_dir,
				seed,
				logging_frequency,
				load_from=None,
				policy_net_lr=1e-4,
				policy_net_final_epsilon=0.01,
				policy_net_final_exploration_frames=10000,
				policy_net_replay_start_size=1000000,
				policy_net_target_update_interval=100,
				policy_net_update_interval=4,
				device='cuda:1'):
		self.name = name
		self.env = env
		self.goal_state_position = goal_state_position
		self.epsilon_within_goal = epsilon_within_goal
		self.death_reward = death_reward
		self.goal_reward = goal_reward
		self.step_reward = step_reward
		self.goal_state = goal_state
		self.max_episode_len = max_episode_len
		self.saving_dir = saving_dir
		self.seed = seed
		self.logging_frequency = logging_frequency
		self.load_from = load_from
		self.buffer_length = buffer_length

		self.success_curve = deque([], maxlen=10)
		self.success_rates = {}

		self.gestation_period = gestation_period
		self.num_goal_hits = 0
		self.num_executions = 0

		# used to store trajectories of positions and finally used for plotting value function
		self.position_buffer = deque([], maxlen=buffer_length)

		# learner for the value function 
		self.policy_net = make_dqn_agent(
			q_agent_type="DQN",
			arch="custom",
			n_actions=self.env.action_space.n,
			lr=policy_net_lr,
			noisy_net_sigma=None,
			buffer_length=buffer_length,
			final_epsilon=policy_net_final_epsilon,
			final_exploration_frames=policy_net_final_exploration_frames,
			use_gpu=-1 if device == 'cpu' else 0,
			replay_start_size=policy_net_replay_start_size,
			target_update_interval=policy_net_target_update_interval,
			update_interval=policy_net_update_interval,
		)

		# load policy network and classifiers
		if self.load_from is not None:
			self.policy_net.load(os.path.join(self.load_from, 'saved_model'))
			self.termination_classifier = SVC()
			self.termination_classifier.load_from_file(os.path.join(self.load_from, 'termination_classifier'))
			self.initiation_classifier = SVC()
			self.initiation_classifier.load_from_file(os.path.join(self.load_from, 'initiation_classifier'))
			# other stats
			self.gestation_period = 0
		else:
			# init classifiers
			self.initiation_classifier = None
			self.termination_classifier = None
			self.initiation_positive_examples = deque([], maxlen=buffer_length)
			self.initiation_negative_examples = deque([], maxlen=buffer_length)
			self.termination_positive_examples = deque([], maxlen=buffer_length)
			self.termination_negative_examples = deque([], maxlen=buffer_length)

	# ------------------------------------------------------------
	# Learning Phase Methods
	# ------------------------------------------------------------

	def get_training_phase(self):
		"""
		determine the training phase, which could only be one of two
		"""
		if self.num_goal_hits < self.gestation_period:
			return "gestation"
		return "initiation_done"

	def is_init_true(self, state):
		"""
		whether the initaition condition is true
		"""
		# initation is always true for an option during training
		if self.get_training_phase() == "gestation":
			return True
		
		# initiation is true if we are at the start state
		copied_env = deepcopy(self.env)
		if np.array_equal(state, copied_env.reset()):
			return True

		return self.initiation_classifier.predict(last_in_framestack(state))[0] == 1

	def is_term_true(self, state, is_dead, eval_mode=False):
		"""
		whether the termination condition is true
		"""
		if self.load_from is not None:
			assert eval_mode
		
		if is_dead:
			# ensure the agent is not dead when hitting the goal
			return False
		if not eval_mode:
			# termination is always true if the state is near the goal
			position = get_player_position(self.env.unwrapped.ale.getRAM())
			distance_to_goal = np.linalg.norm(position - self.goal_state_position)
			if distance_to_goal < self.epsilon_within_goal:
				return True
			else:
				return False
		# if termination classifier isn't initialized, and state is not goal state
		if self.termination_classifier is None:
			return False
		return self.termination_classifier.predict(last_in_framestack(state))[0] == 1

	# ------------------------------------------------------------
	# Control Loop Methods
	# ------------------------------------------------------------
	def reward_function(self, state, is_dead, eval_mode=False):
		"""
		override the env.step() reward, so that options that hit the subgoal
		get a reward (the original monte environment gives no rewards)
		"""
		if self.load_from is not None:
			assert eval_mode

		if is_dead:
			return self.death_reward
		elif self.is_term_true(state, is_dead=is_dead, eval_mode=eval_mode):
			return self.goal_reward
		else:
			return self.step_reward

	def act(self, state, eval_mode=False):
		"""
		return an action for the specified state according to an epsilon greedy policy
		"""
		if self.load_from is not None:
			assert eval_mode

		if eval_mode:
			with self.policy_net.eval_mode():
				return self.policy_net.act(state)
		else:
			return self.policy_net.act(state)
	
	def rollout(self, step_number, eval_mode=False, directed_rollout=False, rendering=False):
		"""
		main control loop for option execution
		"""
		if self.load_from is not None:
			assert eval_mode

		# reset env
		state = self.env.reset()
		is_dead = False
		terminal = self.is_term_true(state, is_dead=is_dead, eval_mode=eval_mode)

		assert self.is_init_true(state)

		num_steps = 0
		total_reward = 0
		visited_states = []
		option_transitions = []
		if not eval_mode:
			goal = self.goal_state
			# state is likely not a LazyFrame because of the MonteForwarding wrapper
			assert goal.shape == state.shape

		# print(f"[Step: {step_number}] Rolling out {self.name}, from {state} targeting {goal}")

		self.num_executions += 1

		# main while loop
		while not terminal:
			# control
			if directed_rollout:
				action = 4  # go down
			else:
				action = self.act(np.array(state), eval_mode=eval_mode)
			next_state, reward, done, info = self.env.step(action)
			is_dead = int(info['ale.lives']) < 6
			done = self.is_term_true(next_state, is_dead=is_dead, eval_mode=eval_mode)
			terminal = done or is_dead or info.get('needs_reset', False) # epsidoe is done if agent dies
			reward = self.reward_function(next_state, is_dead=is_dead, eval_mode=eval_mode)
			if num_steps >= self.max_episode_len:
				terminal = True
			
			# udpate policy if necessary
			if eval_mode:
				with self.policy_net.eval_mode():
					self.policy_net.observe(np.array(state), action, reward, np.array(next_state), terminal)
			else:
				self.policy_net.observe(np.array(state), action, reward, np.array(next_state), terminal)

			# rendering
			if rendering or eval_mode:
				episode_dir = Path(self.saving_dir).joinpath('episode_rendering').joinpath(f'episode_{self.num_executions}')
				episode_dir.mkdir(exist_ok=True, parents=True)
				save_path = episode_dir.joinpath(f"state_at_step_{step_number}.jpeg")
				plt.imsave(save_path, last_in_framestack(next_state))

			# logging
			num_steps += 1
			step_number += 1
			total_reward += reward
			visited_states.append(last_in_framestack(state))
			option_transitions.append((np.array(state), action, reward, np.array(next_state), done))
			state_pos = get_player_position(self.env.unwrapped.ale.getRAM())
			self.position_buffer.append(state_pos)
			state = next_state
			if step_number % self.logging_frequency == 0 and not eval_mode:
				value_func_plots_dir = Path(self.saving_dir).joinpath('value_function_plots')
				value_func_plots_dir.mkdir(exist_ok=True)
				make_chunked_value_function_plot(self.policy_net, 
													step_number, 
													self.seed, 
													value_func_plots_dir, 
													pos_replay_buffer=self.position_buffer)
		visited_states.append(last_in_framestack(state))

		# more logging
		if not directed_rollout:
			self.success_curve.append(self.is_term_true(state, is_dead=is_dead, eval_mode=eval_mode))
			self.success_rates[step_number] = {'success': self.get_success_rate()}
			if self.is_term_true(state, is_dead=is_dead, eval_mode=eval_mode):
				self.num_goal_hits += 1
				print(f"num goal hits increased to {self.num_goal_hits}")
		
		# training classifiers
		if not eval_mode:
			self.derive_positive_and_negative_examples(visited_states, final_state_is_dead=is_dead)
			# refining your initiation/termination classifier
			self.fit_classifier(self.initiation_positive_examples, self.initiation_negative_examples, 'initiation')
			self.fit_classifier(self.termination_positive_examples, self.termination_negative_examples, 'termination')
		
		return option_transitions, total_reward


	# ------------------------------------------------------------
	# Classifiers
	# ------------------------------------------------------------
	def derive_positive_and_negative_examples(self, visited_states, final_state_is_dead):
		"""
		derive positive and negative examples used to train classifiers
		"""
		start_state = visited_states[0]
		final_state = visited_states[-1]

		if self.is_term_true(final_state, is_dead=final_state_is_dead):
			# positive
			positive_states = [start_state] + visited_states[-self.buffer_length:]
			self.initiation_positive_examples += positive_states
			self.termination_positive_examples.append(final_state)
		else:
			negative_examples = [start_state]
			self.initiation_negative_examples += negative_examples
			self.termination_negative_examples.append(final_state)
		
		# all states along the trajectory are negative examples for termination
		self.termination_negative_examples += visited_states[:-1]

	def construct_feature_matrix(self, examples):
		states = list(itertools.chain.from_iterable(examples))
		states = np.array(states).reshape(len(states), -1)  # reshape to (batch_size, state_size)
		return np.array(states)
	
	def fit_classifier(self, positive_examples, negative_examples, classifier):
		"""
		fit the initiation/termination classifier using positive and negative examples
		"""
		assert classifier == 'initiation' or 'termination'
		if len(negative_examples) > 0 and len(positive_examples) > 0:
			self.train_two_class_classifier(classifier, positive_examples, negative_examples)
		elif len(positive_examples) > 0:
			self.train_one_class_svm(classifier, positive_examples)

	def train_one_class_svm(self, classifier_class, positive_examples, nu=0.1):
		positive_feature_matrix = self.construct_feature_matrix(positive_examples)
		classifier = OneClassSVM(kernel="rbf", nu=nu)  # or nu=nu/10. for pessimestic
		classifier.fit(positive_feature_matrix)
		if classifier_class == 'initiation':
			self.initiation_classifier = classifier
		else:
			self.termination_classifier = classifier

	def train_two_class_classifier(self, classifier_class, positive_examples, negative_examples, nu=0.1):
		positive_feature_matrix = self.construct_feature_matrix(positive_examples)
		negative_feature_matrix = self.construct_feature_matrix(negative_examples)
		positive_labels = [1] * positive_feature_matrix.shape[0]
		negative_labels = [0] * negative_feature_matrix.shape[0]

		X = np.concatenate((positive_feature_matrix, negative_feature_matrix))
		Y = np.concatenate((positive_labels, negative_labels))

		if negative_feature_matrix.shape[0] >= 10:
			kwargs = {"kernel": "rbf", "gamma": "auto", "class_weight": "balanced"}
		else:
			kwargs = {"kernel": "rbf", "gamma": "auto"}

		classifier = SVC(**kwargs)
		classifier.fit(X, Y)
		if classifier_class == 'initiation':
			self.initiation_classifier = classifier
		else:
			self.termination_classifier = classifier
	
	# ------------------------------------------------------------
	# testing
	# ------------------------------------------------------------

	def get_success_rate(self):
		if len(self.success_curve) == 0:
			return 0.
		return np.mean(self.success_curve)
