import os
import random

import torch
import numpy as np

from skills.ensemble.policy_ensemble import PolicyEnsemble
from skills.ensemble.aggregate import choose_most_popular
from skills.agents.replay_buffer import ReplayBuffer, Transition


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
                update_interval=4,
                q_target_update_interval=40,
                embedding_output_size=64, 
                embedding_learning_rate=1e-4, 
                policy_learning_rate=1e-2, 
                explore_epsilon=0.1,
                discount_rate=0.9,
                num_modules=8, 
                batch_k=4, 
                normalize=False, 
                num_output_classes=18,
                plot_dir=None):
        # vars
        self.warmup_steps = warmup_steps
        self.num_data_for_update = warmup_steps
        self.batch_size = batch_size
        self.q_target_update_interval = q_target_update_interval
        self.update_interval = update_interval
        self.explore_epsilon = explore_epsilon
        self.num_output_classes = num_output_classes
        self.step_number = 0
        self.update_epochs_per_step = 1
        
        # ensemble and replay buffer
        self.policy_ensemble = PolicyEnsemble(
            device=device,
            embedding_output_size=embedding_output_size,
            embedding_learning_rate=embedding_learning_rate,
            policy_learning_rate=policy_learning_rate,
            discount_rate=discount_rate,
            num_modules=num_modules,
            batch_k=batch_k,
            normalize=normalize,
            num_output_classes=num_output_classes,
            plot_dir=plot_dir,
        )
        self.replay_buffer = ReplayBuffer(max_memory=10000)
    
    def observe(self, obs, action, reward, next_obs, termimal):
        """
        store the experience tuple into the replay buffer
        and update the agent if necessary
        """
        self.replay_buffer.add(Transition(obs, action, reward, next_obs, termimal))
        # update
        if len(self.replay_buffer) > self.warmup_steps and self.step_number % self.update_interval == 0:
            dataset = self.replay_buffer.sample(self.warmup_steps)
            dataset = [dataset[i:i+self.batch_size] for i in range(0, len(dataset), self.batch_size)]
            update_target_net =  self.step_number % self.q_target_update_interval == 0
            self.policy_ensemble.train_embedding(dataset=dataset, epochs=self.update_epochs_per_step)
            self.policy_ensemble.train_q_network(dataset=dataset, epochs=self.update_epochs_per_step, update_target_network=update_target_net)
        self.step_number += 1

    def act(self, obs):
        """
        epsilon-greedy policy
        """
        actions = self.policy_ensemble.predict_actions(torch.from_numpy(np.array(obs)).float())
        if random.random() < self.explore_epsilon:
            a = random.randint(0, self.num_output_classes-1)
        else:
            a = choose_most_popular(actions)
        return a

    def save(self, path):
        self.policy_ensemble.save(path)

    def load(self, path):
        self.policy_ensemble = PolicyEnsemble.load(path)
        # still need to load replay buffer and other vars
