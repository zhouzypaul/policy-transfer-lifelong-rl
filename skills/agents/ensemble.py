import os
import lzma
import dill

import torch
import numpy as np
from pfrl import explorers
from pfrl import replay_buffers
from pfrl.replay_buffer import ReplayUpdater, batch_experiences
from pfrl.utils.batch_states import batch_states

from skills.agents.abstract_agent import Agent, evaluating
from skills.ensemble.value_ensemble import ValueEnsemble
from skills.ensemble.aggregate import choose_most_popular, choose_leader


class EnsembleAgent(Agent):
    """
    an Agent that keeps an ensemble of policies
    an agent needs to support two methods: observe() and act()
    """
    def __init__(self, 
                ensemble_model: ValueEnsemble,
                device, 
                warmup_steps,
                batch_size,
                action_selection_strategy,
                prioritized_replay_anneal_steps,
                phi=lambda x: x,
                buffer_length=100000,
                update_interval=4,
                q_target_update_interval=40,
                final_epsilon=0.01,
                final_exploration_frames=10**6,
                discount_rate=0.9,
                num_modules=8, 
                num_output_classes=18,
                embedding_plot_freq=10000,):
        # vars
        self.device = device
        self.phi = phi
        self.action_selection_strategy = action_selection_strategy
        print(f"using action selection strategy: {self.action_selection_strategy}")
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.q_target_update_interval = q_target_update_interval
        self.update_interval = update_interval
        self.num_modules = num_modules
        self.step_number = 0
        self.episode_number = 0
        self.n_updates = 0
        self.action_leader = np.random.choice(self.num_modules)
        self.learner_accumulated_reward = np.ones((self.num_modules,))  # laplace smoothing
        self.embedding_plot_freq = embedding_plot_freq
        self.discount_rate = discount_rate
        
        # ensemble
        self.value_ensemble = ensemble_model

        # explorer
        self.explorer = explorers.LinearDecayEpsilonGreedy(
            1.0,
            final_epsilon,
            final_exploration_frames,
            lambda: np.random.randint(num_output_classes),
        )

        # Prioritized Replay
        # Anneal beta from beta0 to 1 throughout training
        # taken from https://github.com/pfnet/pfrl/blob/master/examples/atari/reproduction/rainbow/train_rainbow.py
        self.replay_buffer = replay_buffers.PrioritizedReplayBuffer(
            capacity=buffer_length,
            alpha=0.5,  # Exponent of errors to compute probabilities to sample
            beta0=0.4,  # Initial value of beta
            betasteps=prioritized_replay_anneal_steps,  # Steps to anneal beta to 1
            normalize_by_max="memory",  # method to normalize the weight
        )
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

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        if self.training:
            return self._batch_observe_train(
                batch_obs, batch_reward, batch_done, batch_reset
            )
        else:
            return self._batch_observe_eval(
                batch_obs, batch_reward, batch_done, batch_reset
            )
    
    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        for i in range(len(batch_obs)):
            self.step_number += 1

            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                transition = {
                    "state": self.batch_last_obs[i],
                    "action": self.batch_last_action[i],
                    "reward": batch_reward[i],
                    "next_state": batch_obs[i],
                    "next_action": None,
                    "is_state_terminal": batch_done[i],
                }
                self.replay_buffer.append(env_id=i, **transition)
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)

            self.replay_updater.update_if_necessary(self.step_number)

        # action leader
        self.learner_accumulated_reward[self.action_leader] += batch_reward.sum()
        if batch_reset.any() or batch_done.any():
            self.episode_number += np.logical_or(batch_reset, batch_done).sum()
            self._choose_new_leader()  # may be buggy

    def _batch_observe_eval(self, batch_obs, batch_reward, batch_done, batch_terminal):
        # need to do stuff if recurrent 
        pass
    
    def observe(self, obs, action, reward, next_obs, terminal):
        """
        store the experience tuple into the replay buffer
        and update the agent if necessary
        """
        self.step_number += 1

        # update replay buffer 
        if self.training:
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
            self.learner_accumulated_reward[self.action_leader] += reward
            
        # new episode
        if terminal:
            self.episode_number += 1
            self._choose_new_leader()
    
    def _choose_new_leader(self):
        if self.action_selection_strategy == 'uniform_leader':
            self.action_leader = np.random.choice(self.num_modules)
        elif self.action_selection_strategy == 'leader':
            acc_reward = np.array([self.learner_accumulated_reward[l] for l in range(self.num_modules)])
            normalized_reward = acc_reward - acc_reward.max()
            probability = np.exp(normalized_reward) / np.exp(normalized_reward).sum()  # softmax
            self.action_leader = np.random.choice(self.num_modules, p=probability)

    def update(self, experiences, errors_out=None):
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
            errors_out (list or None): If set to a list, then TD-errors
                computed from the given experiences are appended to the list.
        """
        if self.training:
            has_weight = "weight" in experiences[0][0]
            exp_batch = batch_experiences(
                experiences,
                device=self.device,
                phi=self.phi,
                gamma=self.discount_rate,
                batch_states=batch_states,
            )
            # get weights for prioritized experience replay
            if has_weight:
                exp_batch["weights"] = torch.tensor(
                    [elem[0]["weight"] for elem in experiences],
                    device=self.device,
                    dtype=torch.float32,
                )
                if errors_out is None:
                    errors_out = []
            # actual update
            update_target_net =  self.step_number % self.q_target_update_interval == 0
            self.value_ensemble.train(exp_batch, errors_out, update_target_net, plot_embedding=(self.step_number % self.embedding_plot_freq == 0))
            # update prioritiy
            if has_weight:
                assert isinstance(self.replay_buffer, replay_buffers.PrioritizedReplayBuffer)
                self.replay_buffer.update_errors(errors_out)
            
            self.n_updates += 1

    def batch_act(self, batch_obs):
        with torch.no_grad(), evaluating(self):
            batch_xs = batch_states(batch_obs, self.device, self.phi)
            actions = self.value_ensemble.predict_actions(batch_xs, return_q_values=False)  # (num_envs, num_modules)
        # action selection strategy
        if self.action_selection_strategy == 'vote':
            action_selection_func = choose_most_popular
        elif self.action_selection_strategy in ['leader', 'uniform_leader']:
            action_selection_func = lambda a: choose_leader(a, leader=self.action_leader)
        else:
            raise NotImplementedError("action selection strat not supported")
        # epsilon-greedy
        if self.training:
            batch_action = [
                self.explorer.select_action(
                    self.step_number,
                    greedy_action_func=lambda: action_selection_func(actions[i]),
                ) for i in range(len(batch_obs))
            ]
            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else:
            batch_action = [
                action_selection_func(actions[i]) 
                for i in range(len(batch_obs))
            ]
        return np.array(batch_action)

    def act(self, obs, return_ensemble_info=False):
        """
        epsilon-greedy policy
        args:
            obs (object): Observation from the environment.
            return_ensemble_info (bool): when set to true, this function returns
                (action_selected, actions_selected_by_each_learner, q_values_of_each_actions_selected)
        """
        with torch.no_grad(), evaluating(self):
            obs = batch_states([obs], self.device, self.phi)
            actions, q_vals = self.value_ensemble.predict_actions(obs, return_q_values=True)
            actions, q_vals = actions[0], q_vals[0]  # get rid of batch dimension
        # action selection strategy
        if self.action_selection_strategy == 'vote':
            action_selection_func = choose_most_popular
        elif self.action_selection_strategy in ['leader', 'uniform_leader']:
            action_selection_func = lambda a: choose_leader(a, leader=self.action_leader)
        else:
            raise NotImplementedError("action selection strat not supported")
        # epsilon-greedy
        if self.training:
            a = self.explorer.select_action(
                self.step_number,
                greedy_action_func=lambda: action_selection_func(actions),
            )
        else:
            a = action_selection_func(actions)
        if return_ensemble_info:
            return a, actions, q_vals
        return a
    
    def get_statistics(self):
        return []

    def save(self, save_dir):
        path = os.path.join(save_dir, "agent.pkl")
        with lzma.open(path, 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, load_path, plot_dir=None):
        with lzma.open(load_path, 'rb') as f:
            agent = dill.load(f)
        # hack to change the plot_dir of the agent
        agent.value_ensemble.embedding.plot_dir = plot_dir
        return agent
