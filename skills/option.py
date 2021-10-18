import itertools
from copy import deepcopy
from collections import deque
from pathlib import Path

import numpy as np
from thundersvm import SVC, OneClassSVM
import matplotlib.pyplot as plt

from skills.option_utils import get_player_position, make_chunked_value_function_plot
from skills.agents.dqn import make_dqn_agent


class Option:
	"""
	the base class for option that all Option class shoud inherit from
	"""
	def __init__(self, name, env, params):
		self.name = name
		self.env = env
		self.params = params

		self.initiation_classifier = None
		self.termination_classifier = None
		self.initiation_positive_examples = deque([], maxlen=self.params['buffer_length'])
		self.initiation_negative_examples = deque([], maxlen=self.params['buffer_length'])
		self.termination_positive_examples = deque([], maxlen=self.params['buffer_length'])
		self.termination_negative_examples = deque([], maxlen=self.params['buffer_length'])

		self.success_curve = deque([], maxlen=self.params['buffer_length'])
		self.success_rates = {}

		self.gestation_period = params['gestation_period']
		self.num_goal_hits = 0
		self.num_executions = 0

		# used to store trajectories of positions and finally used for plotting value function
		self.position_buffer = deque([], maxlen=self.params['buffer_length'])

		# learner for the value function 
		self.policy_net = make_dqn_agent(
			q_agent_type="DQN",
			arch="custom",
			n_actions=self.env.action_space.n,
			lr=self.params['lr'],
			noisy_net_sigma=None,
			buffer_length=self.params['buffer_length'],
			final_epsilon=self.params['final_epsilon'],
			final_exploration_frames=self.params['final_exploration_frames'],
			use_gpu=-1 if self.params['device'] == 'cpu' else 0,
			replay_start_size=self.params['replay_start_size'],
			target_update_interval=self.params['target_update_interval'],
			update_interval=self.params['update_interval'],
		)

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

		return self.initiation_classifier.predict([state])[0] == 1

	def is_term_true(self, state, is_dead, eval_mode=False):
		"""
		whether the termination condition is true
		"""
		if is_dead:
			# ensure the agent is not dead when hitting the goal
			return False
		if not eval_mode:
			# termination is always true if the state is near the goal
			position = get_player_position(self.env.unwrapped.ale.getRAM())
			distance_to_goal = np.linalg.norm(position - self.params['goal_state_position'])
			if distance_to_goal < self.params['epsilon_within_goal']:
				return True
			else:
				return False
		# if termination classifier isn't initialized, and state is not goal state
		if self.termination_classifier is None:
			return False
		return self.termination_classifier.predict([state])[0] == 1

	# ------------------------------------------------------------
	# Control Loop Methods
	# ------------------------------------------------------------
	def reward_function(self, state, is_dead, eval_mode=False):
		"""
		override the env.step() reward, so that options that hit the subgoal
		get a reward (the original monte environment gives no rewards)
		"""
		if is_dead:
			return self.params['death_reward']
		elif self.is_term_true(state, is_dead=is_dead, eval_mode=eval_mode):
			return self.params['goal_reward']
		else:
			return self.params['step_reward']

	def act(self, state, eval_mode=False):
		"""
		return an action for the specified state according to an epsilon greedy policy
		"""
		if eval_mode:
			with self.policy_net.eval_mode():
				return self.policy_net.act(state)
		else:
			return self.policy_net.act(state)
	
	def rollout(self, step_number, eval_mode=False, rendering=False):
		"""
		main control loop for option execution
		"""
		# reset env
		state = self.env.reset()
		terminal = False
		is_dead = False

		assert self.is_init_true(state)

		num_steps = 0
		total_reward = 0
		visited_states = []
		option_transitions = []
		if not eval_mode:
			goal = self.params['goal_state']
			# channel doesn't need to match, in case we down sampled
			assert goal.shape[:-1] == np.array(state).shape[:-1]

		# print(f"[Step: {step_number}] Rolling out {self.name}, from {state} targeting {goal}")

		self.num_executions += 1

		# main while loop
		while not self.is_term_true(state.flatten(), is_dead=is_dead, eval_mode=eval_mode) and not terminal:
			# control
			action = self.act(state, eval_mode=eval_mode)
			next_state, reward, done, info = self.env.step(action)
			is_dead = int(info['ale.lives']) < 6
			done = self.is_term_true(next_state.flatten(), is_dead=is_dead, eval_mode=eval_mode)
			terminal = done or is_dead or info.get('needs_reset', False) # epsidoe is done if agent dies
			reward = self.reward_function(next_state.flatten(), is_dead=is_dead, eval_mode=eval_mode)
			if num_steps >= self.params['max_episode_len']:
				terminal = True
			
			# udpate policy if necessary
			self.policy_net.observe(next_state, reward, done, terminal)

			# rendering
			if rendering or eval_mode:
				episode_dir = Path(self.params['saving_dir']).joinpath(f'episode_{self.num_executions}')
				episode_dir.mkdir(exist_ok=True)
				save_path = episode_dir.joinpath(f"state_at_step_{step_number}.jpeg")
				try:
					plt.imsave(save_path, next_state)
				except ValueError:
					# cannot plot because next_state is a framestack of 4
					plt.imsave(save_path, next_state[0])

			# logging
			num_steps += 1
			step_number += 1
			total_reward += reward
			visited_states.append(state.flatten())
			option_transitions.append((state.flatten(), action, reward, next_state.flatten(), done))
			state_pos = get_player_position(self.env.unwrapped.ale.getRAM())
			self.position_buffer.append(state_pos)
			state = next_state
			if step_number % self.params['logging_frequency'] == 0 and not eval_mode:
				value_func_plots_dir = Path(self.params['saving_dir']).joinpath('value_function_plots')
				value_func_plots_dir.mkdir(exist_ok=True)
				make_chunked_value_function_plot(self.policy_net, 
													step_number, 
													self.params['seed'], 
													value_func_plots_dir, 
													pos_replay_buffer=self.position_buffer)
		visited_states.append(state.flatten())

		# more logging
		self.success_curve.append(self.is_term_true(state.flatten(), is_dead=is_dead, eval_mode=eval_mode))
		self.success_rates[step_number] = {'success': self.get_success_rate()}
		if self.is_term_true(state.flatten(), is_dead=is_dead, eval_mode=eval_mode):
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
			positive_states = [start_state] + visited_states[-self.params['buffer_length']:]
			self.initiation_positive_examples += positive_states
			self.termination_positive_examples.append(final_state)
		else:
			negative_examples = [start_state]
			self.initiation_negative_examples += negative_examples
			self.termination_negative_examples.append(final_state)

	def construct_feature_matrix(self, examples):
		states = list(itertools.chain.from_iterable(examples))
		states = np.array(states).reshape(len(states), -1)  # reshape to (batch_size, state_size)
		return np.array(states)
	
	def fit_classifier(self, positive_examples, negative_examples, classifier):
		"""
		fit the initiation/termination classifier using positive and negative examples
		"""
		assert classifier is 'initiation' or 'termination'
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
