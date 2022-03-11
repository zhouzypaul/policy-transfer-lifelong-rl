import random

import torch
import numpy as np
from pfrl import explorers
from pfrl.replay_buffers import ReplayBuffer
from pfrl.replay_buffer import ReplayUpdater, batch_experiences
from pfrl.utils.batch_states import batch_states

from skills.ensemble.policy_ensemble import PolicyEnsemble
from skills.ensemble.aggregate import choose_most_popular


class EnsembleAgent():
    """
    an Agent that keeps an ensemble of policies
    an agent needs to support two methods: observe() and act()

    this class currently doesn't support batched observe() and act()
    """
    def __init__(self, 
                device, 
                warmup_steps,
                batch_size,
                phi,
                buffer_length=100000,
                update_interval=4,
                q_target_update_interval=40,
                embedding_output_size=64, 
                embedding_learning_rate=1e-4, 
                policy_learning_rate=1e-2, 
                explore_epsilon=0.1,
                final_epsilon=0.01,
                final_exploration_frames=10 ** 6,
                discount_rate=0.9,
                num_modules=8, 
                num_output_classes=18,
                plot_dir=None,
                embedding_plot_freq=10000,
                verbose=False,):
        # vars
        self.device = device
        self.phi = phi
        self.warmup_steps = warmup_steps
        self.num_data_for_update = warmup_steps
        self.batch_size = batch_size
        self.q_target_update_interval = q_target_update_interval
        self.update_interval = update_interval
        self.explore_epsilon = explore_epsilon
        self.num_output_classes = num_output_classes
        self.step_number = 0
        self.update_epochs_per_step = 1
        self.embedding_plot_freq = embedding_plot_freq
        self.discount_rate = discount_rate
        
        # ensemble
        self.policy_ensemble = PolicyEnsemble(
            device=device,
            embedding_output_size=embedding_output_size,
            embedding_learning_rate=embedding_learning_rate,
            policy_learning_rate=policy_learning_rate,
            discount_rate=discount_rate,
            num_modules=num_modules,
            num_output_classes=num_output_classes,
            plot_dir=plot_dir,
            verbose=verbose,
        )

        # explorer
        self.explorer = explorers.LinearDecayEpsilonGreedy(
            1.0,
            final_epsilon,
            final_exploration_frames,
            lambda: np.random.randint(num_output_classes),
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_length)
        self.replay_updater = ReplayUpdater(
            replay_buffer=self.replay_buffer,
            update_func=self.update,
            batchsize=batch_size,
            episodic_update=False,
            episodic_update_len=None,
            n_times_update=1,
            replay_start_size=warmup_steps,
            update_interval=update_interval,
        )
    
    def observe(self, obs, action, reward, next_obs, terminal):
        """
        store the experience tuple into the replay buffer
        and update the agent if necessary
        """
        transition = {
            "state": obs,
            "action": action,
            "reward": reward,
            "next_state": next_obs,
            "next_action": None,
            "is_state_terminal": terminal,
        }
        self.replay_buffer.append(**transition)
        if terminal:
            self.replay_buffer.stop_current_episode()

        self.replay_updater.update_if_necessary(self.step_number)
        self.step_number += 1

    def update(self, experiences):
        """
        update the model
        accepts as argument a list of transition dicts
        args:
            transitions (list): List of lists of dicts.
                For DQN, each dict must contains:
                  - state (object): State
                  - action (object): Action
                  - reward (float): Reward
                  - is_state_terminal (bool): True iff next state is terminal
                  - next_state (object): Next state
                  - weight (float, optional): Weight coefficient. It can be
                    used for importance sampling.
        """
        exp_batch = batch_experiences(
            experiences,
            device=self.device,
            phi=self.phi,
            gamma=self.discount_rate,
            batch_states=batch_states,
        )
        update_target_net =  self.step_number % self.q_target_update_interval == 0
        self.policy_ensemble.train_embedding(exp_batch, epochs=self.update_epochs_per_step, plot_embedding=(self.step_number % self.embedding_plot_freq == 0))
        self.policy_ensemble.train_q_network(exp_batch, epochs=self.update_epochs_per_step, update_target_network=update_target_net)

    def act(self, obs):
        """
        epsilon-greedy policy
        """
        obs = batch_states([obs], self.device, self.phi)
        actions = self.policy_ensemble.predict_actions(obs)
        # epsilon-greedy
        a = self.explorer.select_action(
            self.step_number,
            greedy_action_func=lambda: choose_most_popular(actions),
        )   
        return a

    def save(self, path):
        self.policy_ensemble.save(path)

    def load(self, path):
        self.policy_ensemble = PolicyEnsemble.load(path)
        # still need to load replay buffer and other vars
