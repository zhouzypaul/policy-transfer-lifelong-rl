import os
import argparse
from pathlib import Path

import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import pfrl
from pfrl.utils import set_random_seed

from skills import utils
from skills.wrappers.atari_wrappers import make_atari, wrap_deepmind
from skills.wrappers.agent_wrapper import MonteAgentWrapper
from skills.wrappers.procgen_wrapper import ProcgenGymWrapper, ChannelOrderWrapper
from skills.wrappers.procgen_agent_wrapper import ProcgenAgentWrapper
from skills.wrappers.monte_forwarding_wrapper import MonteForwarding
from skills.wrappers.monte_termination_set_wrapper import MonteTerminationSetWrapper
from skills.wrappers.monte_pruned_actions import MontePrunedActions
from skills.wrappers.monte_ladder_goal_wrapper import MonteLadderGoalWrapper
from skills.wrappers.monte_skull_goal_wrapper import MonteSkullGoalWrapper
from skills.wrappers.monte_spider_goal_wrapper import MonteSpiderGoalWrapper
from skills.wrappers.monte_snake_goal_wrapper import MonteSnakeGoalWrapper

cv2.ocl.setUseOpenCL(False)


class BaseTrial:
    """
    a base class for running experiments
    """
    def __init__(self):
        pass

    def get_common_arg_parser(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False,
        )
        # common args
        # system 
        parser.add_argument("--experiment_name", "-e", type=str,
                            help="Experiment Name, also used as the directory name to save results")
        parser.add_argument("--results_dir", type=str, default='results',
                            help='the name of the directory used to store results')
        parser.add_argument("--device", type=str, default='cuda',
                            help="cpu/cuda/cuda:0/cuda:1")
        # environments
        parser.add_argument("--environment", type=str,
                            help="name of the gym environment")
        parser.add_argument("--seed", type=int, default=0,
                            help="Random seed")
        # hyperparams
        parser.add_argument('--params_dir', type=Path, default='hyperparams',
                            help='the hyperparams directory')
        parser.add_argument('--hyperparams', type=str, default='atari',
                            help='which hyperparams csv file to use')
        return parser

    def parse_common_args(self, parser):
        args, unknown = parser.parse_known_args()
        other_args = {
            (utils.remove_prefix(key, '--'), val)
            for (key, val) in zip(unknown[::2], unknown[1::2])
        }
        args.other_args = other_args
        return args

    def load_hyperparams(self, args):
        """
        load the hyper params from args to a params dictionary
        """
        params = utils.load_hyperparams(args.params_dir.joinpath(args.hyperparams + '.csv'))
        for arg_name, arg_value in vars(args).items():
            # if arg_name == 'hyperparams':
            #     continue
            params[arg_name] = arg_value
        for arg_name, arg_value in args.other_args:
            utils.update_param(params, arg_name, arg_value)
        return params
    
    def make_deterministic(self, seed, force_deterministic=False):
        set_random_seed(seed)
        torch.backends.cudnn.benchmark = False  # make alg selection determistics, but may be slower
        if force_deterministic:
            torch.use_deterministic_algorithms(True)  # must use deterministic algorithms

    def make_env(self, env_name, env_seed):
        if self.params['use_deepmind_wrappers']:
            env = pfrl.wrappers.atari_wrappers.make_atari(env_name, max_frames=30*60*60)  # 30 min with 60 fps
            env = pfrl.wrappers.atari_wrappers.wrap_deepmind(
                env,
                episode_life=True,
                clip_rewards=True,
                frame_stack=True,
                scale=False,
                fire_reset=True,
                channel_order="chw",
                flicker=False,
            )
        else:
            env = gym.make(env_name)
        print(f'making environment {env_name}')
        env.seed(env_seed)
        env.action_space.seed(env_seed)
        return env


class SingleOptionTrial(BaseTrial):
    """
    a base class for every class that deals with training/executing a single option
    This class should only be used for training on Montezuma
    """
    def __init__(self):
        super().__init__()
    
    def setup(self):
        self._expand_agent_name()
    
    def check_params_validity(self):
        if self.params['skill_type'] == 'ladder':
            print(f"changing epsilon to ladder specific one: {self.params['ladder_epsilon_tol']}")
            self.params['goal_epsilon_tol'] = self.params['ladder_epsilon_tol']

    def get_common_arg_parser(self):
        parser = super().get_common_arg_parser()
        # defaults
        parser.set_defaults(experiment_name='train', 
                            environment='MontezumaRevenge', 
                            hyperparams='atari')
    
        # environments
        parser.add_argument("--render", action='store_true', default=False, 
                            help="save the images of states while training")
        parser.add_argument("--agent_space", action='store_true', default=False,
                            help="train with the agent space")
        parser.add_argument("--use_deepmind_wrappers", action='store_true', default=True,
                            help="use the deepmind wrappers")
        parser.add_argument("--suppress_action_prunning", action='store_true', default=True,
                            help='do not prune the action space of monte')

        # ensemble
        parser.add_argument("--action_selection_strat", type=str, default="ucb_leader",
                            choices=['vote', 'uniform_leader', 'greedy_leader', 'ucb_leader', 'add_qvals'],
                            help="the action selection strategy when using ensemble agent")
        
        # skill type
        parser.add_argument("--skill_type", "-s", type=str, default="enemy", 
                            choices=['skull', 'snake', 'spider', 'enemy', 'ladder', 'finish_game'], 
                            help="the type of skill to train")

        # start state
        parser.add_argument("--info_dir", type=Path, default="resources/monte_info",
                            help="where the goal state files are stored")
        parser.add_argument("--ram_dir", type=Path, default="resources/monte_ram",
                            help="where the monte ram (encoded) files are stored")

        parser.add_argument("--start_state", type=str, default=None,
                            help="""filename that saved the starting state RAM. 
                                    This should not include the whole path or the .npy extension.
                                    e.g: room1_right_ladder_top""")
        
        # termination classifiers
        parser.add_argument("--termination_clf", "-c", action='store_true', default=False,
                            help="whether to use the trained termination classifier to determine episodic done.")
        parser.add_argument("--confidence_based_reward", action='store_true', default=False,
                            help="whether to use the confidence based reward when using the trained termination classifer")
        return parser
    
    def find_start_state_ram_file(self, start_state):
        """
        given the start state string, find the corresponding RAM file in ram_dir
        when the skill type is enemey, look through all the enemies directories
        """
        real_skill_type = self._get_real_skill_type(start_state)
        start_state_path = self.params['ram_dir'] / real_skill_type / f'{start_state}.npy'
        if not start_state_path.exists():
            raise FileNotFoundError(f'{start_state_path} does not exist')
        return start_state_path
    
    def _get_real_skill_type(self, saved_ram_file):
        """
        when the entered skill type is enemy, the real skil type is the actual enemy type
        """
        if self.params['skill_type'] == 'enemy':
            enemies = ['skull', 'snake', 'spider']
            for enemy in enemies:
                saved_ram_path = self.params['ram_dir'] / f'{enemy}' / f'{saved_ram_file}.npy'
                if saved_ram_path.exists():
                    real_skill = enemy
            try:
                real_skill
            except NameError:
                raise RuntimeError(f"Could not find real skill type for {saved_ram_file} and entered skill {self.params['skill_type']}")
        else:
            real_skill = self.params['skill_type']
        self.real_skill_type = real_skill
        return real_skill
    
    def _expand_agent_name(self):
        """
        expand the agent name to include information such as whether using agent space.
        """
        agent = self.params['agent']
        if agent == 'ensemble':
            agent += f"-{self.params['num_policies']}"
        if self.params['agent_space']:
            agent += '-agent-space'
        if self.params['action_selection_strat'] == 'add_qvals':
            agent += '-add-qvals'

        self.detailed_agent_name = agent

        if self.params['termination_clf']:
            agent += '-termclf'
            agent += f"-highconf-{self.params['termination_num_agreeing_votes']}"
        if self.params['confidence_based_reward']:
            agent += '-cbr'
        self.expanded_agent_name = agent
    
    def _set_saving_dir(self):
        self._expand_agent_name()
        return Path(self.params['results_dir']).joinpath(self.params['experiment_name']).joinpath(self.expanded_agent_name)

    def make_env(self, env_name, env_seed, start_state=None):
        """
        Make a monte environemnt for training skills
        Args:
            goal: None or (x, y)
        """
        if env_name == 'MontezumaRevenge':
            # ContinuingTimeLimit, NoopResetEnv, MaxAndSkipEnv
            env = make_atari(f"{env_name}NoFrameskip-v4", max_frames=30*60*60)  # 30 min with 60 fps
            # make agent space
            if self.params['agent_space']:
                print('using the agent space to train the option right now')
            env = MonteAgentWrapper(env, agent_space=self.params['agent_space'])
            if self.params['use_deepmind_wrappers']:
                env = wrap_deepmind(
                    env,
                    warp_frames=not self.params['agent_space'],
                    episode_life=True,
                    clip_rewards=True,
                    frame_stack=True,
                    scale=False,
                    fire_reset=False,
                    channel_order="chw",
                    flicker=False,
                )
            # prunning actions
            if not self.params['suppress_action_prunning']:
                env = MontePrunedActions(env)
            # starting state wrappers
            if start_state is not None:
                start_state_path = self.find_start_state_ram_file(start_state)
                # MonteForwarding should be after EpisodicLifeEnv so that reset() is correct
                # this does not need to be enforced once test uses the timeout wrapper
                env = MonteForwarding(env, start_state_path)
            # termination wrappers
            if self.params['termination_clf']:
                env = MonteTerminationSetWrapper(env, confidence_based_reward=self.params['confidence_based_reward'], device=self.params['device'])
                print('using trained termination classifier')
            # skills and goals
            self._get_real_skill_type(start_state)
            info_only = self.params['termination_clf']
            if self.real_skill_type == 'ladder':
                # ladder goals
                # should go after the forwarding wrappers, because the goals depend on the position of 
                # the agent in the starting state
                env = MonteLadderGoalWrapper(env, epsilon_tol=self.params['goal_epsilon_tol'], info_only=info_only)
                print('pursuing ladder skills')
            elif self.real_skill_type == 'skull':
                env = MonteSkullGoalWrapper(env, epsilon_tol=self.params['goal_epsilon_tol'], info_only=info_only)
                print('pursuing skull skills')
            elif self.real_skill_type == 'spider':
                env = MonteSpiderGoalWrapper(env, epsilon_tol=self.params['goal_epsilon_tol'], info_only=info_only)
                print('pursuing spider skills')
            elif self.real_skill_type == 'snake':
                env = MonteSnakeGoalWrapper(env, epsilon_tol=self.params['goal_epsilon_tol'], info_only=info_only)
                print('pursuing snake skills')
            env.seed(env_seed)
        else:
            # procgen environments 
            # env = ProcgenEnv(
            #     num_envs=1,
            #     env_name='coinrun',
            #     rand_seed=env_seed,
            #     num_levels=0,
            #     start_level=0,
            #     distribution_mode='easy',
            #     render_mode='rgb_array',
            #     center_agent=True,
            # )
            env = gym.make(
                f'procgen:procgen-{env_name}-v0', 
                rand_seed=env_seed,
                center_agent=True,
                num_levels=0,
                start_level=0,
                distribution_mode='easy',
                render_mode='rgb_array',
            )
            env = ProcgenGymWrapper(env, agent_space=self.params['agent_space'])
            env = ProcgenAgentWrapper(env)
            env = ChannelOrderWrapper(env, grayscale=False, channel_order='chw')
        print(f'making environment {env_name}')
        env.action_space.seed(env_seed)
        return env


def extract(input, idx, idx_dim, batch_dim=0):
    '''
Extracts slices of input tensor along idx_dim at positions
specified by idx.
Notes:
    idx must have the same size as input.shape[batch_dim].
    Output tensor has the shape of input with idx_dim removed.
Args:
    input (Tensor): the source tensor
    idx (LongTensor): the indices of slices to extract
    idx_dim (int): the dimension along which to extract slices
    batch_dim (int): the dimension to treat as the batch dimension
Example::
    >>> t = torch.arange(24, dtype=torch.float32).view(3,4,2)
    >>> i = torch.tensor([1, 3, 0], dtype=torch.int64)
    >>> extract(t, i, idx_dim=1, batch_dim=0)
        tensor([[ 2.,  3.],
                [14., 15.],
                [16., 17.]])
'''
    if idx_dim == batch_dim:
        raise RuntimeError('idx_dim cannot be the same as batch_dim')
    if len(idx) != input.shape[batch_dim]:
        raise RuntimeError(
            "idx length '{}' not compatible with batch_dim '{}' for input shape '{}'".format(
                len(idx), batch_dim, list(input.shape)))
    viewshape = [
        1,
    ] * input.ndimension()
    viewshape[batch_dim] = input.shape[batch_dim]
    idx = idx.view(*viewshape).expand_as(input)
    result = torch.gather(input, idx_dim, idx).mean(dim=idx_dim)
    return result


def make_done_state_plot(replay_buffer, episode_idx, save_dir):
    """
    visualize the done positions
    """
    if not replay_buffer:
        # if replay buffer is empty
        return
    states = np.array([exp[0] for exp in replay_buffer])
    dones = np.array([exp[-1] for exp in replay_buffer])
    done_states = states[dones]

    for i in range(len(done_states)):
        s = np.array(done_states[i])
        frame = s[-1, :, :]  # final frame in framestack
        file_name = save_dir.joinpath(f"done_state_plot_at_episode_{episode_idx}__{i}.jpg")
        plt.imsave(file_name, frame)


def make_chunked_value_function_plot(solver, step, seed, save_dir, pos_replay_buffer, chunk_size=1000):
    """
    helper function to visualize the value function
    """
    replay_buffer = solver.replay_buffer
    states = np.array([exp[0]['state'] for exp in replay_buffer.memory])
    actions = np.array([exp[0]['action'] for exp in replay_buffer.memory])

    # Chunk up the inputs so as to conserve GPU memory
    num_chunks = int(np.ceil(states.shape[0] / chunk_size))

    if num_chunks == 0:
        return 0.

    state_chunks = np.array_split(states, num_chunks, axis=0)
    action_chunks = np.array_split(actions, num_chunks, axis=0)
    qvalues = np.zeros((states.shape[0],))
    current_idx = 0

    for state_chunk, action_chunk in zip(state_chunks, action_chunks):
        state_chunk = torch.from_numpy(state_chunk).float().to(solver.device)
        action_chunk = torch.from_numpy(action_chunk).float().to(solver.device)
        with torch.no_grad():
            chunk_qvalues = solver.model(state_chunk).q_values.cpu().numpy()
            actions_taken = list(map(lambda a: int(a), action_chunk.cpu().numpy()))
            chunk_action_taken_qvalues = [chunk_qvalues[i, idx_a] for i, idx_a in enumerate(actions_taken)]
        current_chunk_size = len(state_chunk)
        qvalues[current_idx:current_idx + current_chunk_size] = chunk_action_taken_qvalues
        current_idx += current_chunk_size

    x_pos = np.array([pos[0] for pos in pos_replay_buffer])
    y_pos = np.array([pos[1] for pos in pos_replay_buffer])
    try:
        plt.scatter(x_pos, y_pos, c=qvalues)
    except ValueError:
        num_points = min(len(x_pos), len(qvalues))
        x_pos = x_pos[:num_points]
        y_pos = y_pos[:num_points]
        qvalues = qvalues[:num_points]
        plt.scatter(x_pos, y_pos, c=qvalues)
    plt.xlim(0, 160)  # set the limits to the monte frame
    plt.ylim(145, 240)
    plt.colorbar()
    file_name = f"value_function_seed_{seed}_step_{step}.png"
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    plt.close()

    return qvalues.max()


def visualize_positive_reward_state(state, reward, step_number, save_dir):
    """
    when the reward is positive, visualize it to see what the agent is doing
    """
    if reward > 0:
        frame_stack = np.array(state)
        plt.imsave(os.path.join(save_dir, f"{step_number}_0.png"), frame_stack[0])
        plt.imsave(os.path.join(save_dir, f"{step_number}_1.png"), frame_stack[1])
        plt.imsave(os.path.join(save_dir, f"{step_number}_2.png"), frame_stack[2])
        plt.imsave(os.path.join(save_dir, f"{step_number}_3.png"), frame_stack[3])
        print(f"plotted at step {step_number}")
