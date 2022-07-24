"""
acknowledgement:
this code is adapted from 
https://github.com/lerrytang/train-procgen-pfrl/
"""

import os
import time
import argparse
from pathlib import Path
from collections import deque

import torch
import numpy as np
from procgen import ProcgenEnv
from pfrl.utils import set_random_seed

from skills.vec_env import VecExtractDictObs, VecMonitor, VecNormalize, VecChannelOrder
from skills.agents import PPO, EnsembleAgent
from skills.option_utils import BaseTrial
from skills.models.impala import ImpalaCNN
from skills.baseline import logger
from skills import utils
from skills.baseline.plot import plot_reward_curve


class ProcgenTrial(BaseTrial):
    """
    trial for training procgen
    """
    def __init__(self):
        super().__init__()
        args = self.parse_args()
        self.params = self.load_hyperparams(args)
        self.setup()

    def parse_args(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            parents=[self.get_common_arg_parser()]
        )
        # defaults
        parser.set_defaults(hyperparams='procgen_ppo')

        # procgen environment
        parser.add_argument('--env', type=str, required=True,
                            help='name of the procgen environment')
        parser.add_argument('--distribution_mode', '-d', type=str, default='easy',
                            choices=['easy', 'hard', 'exploration, memeory', 'extreme'],
                            help='distribution mode of procgen')
        parser.add_argument('--num_envs', type=int, default=64,
                            help='number of environments to run in parallel')
        parser.add_argument('--num-threads', type=int, default=4)
        parser.add_argument('--num_levels', type=int, default=200,
                            help='number of different levels to generate during training')
        parser.add_argument('--start_level', type=int, default=0,
                            help='seed to start level generation')
        
        # agent
        parser.add_argument('--agent', type=str, default='ppo',
                            choices=['ppo', 'ensemble'])
        parser.add_argument('--load', '-l', type=str, default=None,
                            help='path to load agent')
        
        args = self.parse_common_args(parser)
        # auto fill
        if args.experiment_name is None:
            args.experiment_name = args.env
        if args.agent == 'ppo':
            args.hyperparams = 'procgen_ppo'
        elif args.agent == 'ensemble':
            args.hyperparams = 'procgen_ensemble'
        return args

    def check_params_validity(self):
        """
        check whether the params entered by the user is valid
        """
        if self.params['agent'] == 'ensemble':
            try:
                assert self.params['target_update_interval'] == self.params['ensemble_target_update_interval'] * self.params['update_interval']
            except AssertionError:
                new_interval = self.params['ensemble_target_update_interval'] * self.params['update_interval']
                print(f"updating target_update_interval to be {new_interval}")
                self.params['target_update_interval'] = new_interval
    
    def make_vector_env(self, eval=False):
        venv = ProcgenEnv(
            num_envs=self.params['num_envs'],
            env_name=self.params['env'],
            num_levels=0 if eval else self.params['num_levels'],
            start_level=0 if eval else self.params['start_level'],
            distribution_mode=self.params['distribution_mode'],
            num_threads=self.params['num_threads'],
            center_agent=True,
        )
        venv = VecChannelOrder(venv, channel_order='chw')
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        venv = VecNormalize(venv=venv, ob=False)
        return venv
    
    def make_agent(self, env):
        if self.params['agent'] == 'ppo':
            # Create policy.
            policy = ImpalaCNN(
                obs_space=env.observation_space,
                num_outputs=env.action_space.n,
            )

            # Create agent and train.
            optimizer = torch.optim.Adam(policy.parameters(), lr=self.params['learning_rate'], eps=1e-5)
            ppo_agent = PPO(
                model=policy,
                optimizer=optimizer,
                gpu=-1 if self.params['device']=='cpu' else 0,
                gamma=self.params['gamma'],
                lambd=self.params['lambda'],
                value_func_coef=self.params['value_function_coef'],
                entropy_coef=self.params['entropy_coef'],
                update_interval=self.params['nsteps'] * self.params['num_envs'],
                minibatch_size=self.params['batch_size'],
                epochs=self.params['nepochs'],
                clip_eps=self.params['clip_range'],
                clip_eps_vf=self.params['clip_range'],
                max_grad_norm=self.params['max_grad_norm'],
            )
            return ppo_agent
        elif self.params['agent'] == 'ensemble':
            agent = EnsembleAgent(
                device=self.params['device'],
                phi=lambda x: x.astype(np.float32),
                action_selection_strategy=self.params['action_selection_strat'],
                warmup_steps=self.params['warmup_steps'],
                batch_size=self.params['batch_size'],
                prioritized_replay_anneal_steps=self.params['max_steps'] / self.params['update_interval'],
                buffer_length=self.params['buffer_length'],
                update_interval=self.params['update_interval'],
                q_target_update_interval=self.params['target_update_interval'],
                num_modules=self.params['num_policies'],
                num_output_classes=env.action_space.n,
                plot_dir=self.params['plots_dir'],
                verbose=False,
            )
            return agent
        else:
            raise NotImplementedError('Unsupported agent')
    
    def _expand_agent_name(self):
        agent = self.params['agent']
        if agent == 'ensemble':
            agent += f"-{self.params['num_policies']}"
        self.expanded_agent_name = agent
    
    def _set_saving_dir(self):
        self._expand_agent_name()
        return Path(self.params['results_dir'], self.params['experiment_name'], self.expanded_agent_name)

    def make_logger(self, log_dir):
        logger.configure(dir=log_dir, format_strs=['csv', 'stdout'])
    
    def setup(self):
        self.check_params_validity()
        set_random_seed(self.params['seed'])
        torch.backends.cudnn.benchmark = True

        # set up saving dir
        self.saving_dir = self._set_saving_dir()
        utils.create_log_dir(self.saving_dir, remove_existing=True)
        self.params['saving_dir'] = self.saving_dir
        self.params['plots_dir'] = os.path.join(self.saving_dir, 'plots')
        os.mkdir(self.params['plots_dir'])

        # save hyperparams
        utils.save_hyperparams(self.saving_dir.joinpath('hyperparams.csv'), self.params)

        # logger
        self.logger = self.make_logger(self.saving_dir)

        # env
        self.train_env = self.make_vector_env(eval=False)
        self.eval_env = self.make_vector_env(eval=True)

        # agent
        self.agent = self.make_agent(self.train_env)
    
    def train(self):
        train_with_eval(
            agent=self.agent,
            train_env=self.train_env,
            test_env=self.eval_env,
            num_envs=self.params['num_envs'],
            nsteps=self.params['nsteps'],
            nepochs=self.params['nepochs'],
            max_steps=self.params['max_steps'],
            batch_size=self.params['batch_size'],
            model_dir=self.saving_dir,
            save_interval=self.params['save_interval'],
            model_file=self.params['load'],
        )
        plot_reward_curve(self.saving_dir)


def safe_mean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def rollout_one_step(agent, env, obs, steps, env_max_steps=1000):

    # Step once.
    action = agent.batch_act(obs)
    new_obs, reward, done, infos = env.step(action)
    steps += 1
    reset = steps == env_max_steps
    steps[done] = 0

    # Save experience.
    agent.batch_observe(
        batch_obs=new_obs,
        batch_reward=reward,
        batch_done=done,
        batch_reset=reset,
    )

    # Get rollout statistics.
    epinfo = []
    for info in infos:
        maybe_epinfo = info.get('episode')
        if maybe_epinfo:
            epinfo.append(maybe_epinfo)

    return new_obs, steps, epinfo


def train_with_eval(
    agent,
    train_env, 
    test_env,
    num_envs,
    nsteps,
    nepochs,
    max_steps,
    batch_size,
    model_dir,
    save_interval,
    model_file=None,
):

    if model_file is not None:
        load_agent(agent, model_file, plot_dir=os.path.join(model_dir, 'plots'))
    else:
        logger.info('Train agent from scratch.')

    train_epinfo_buf = deque(maxlen=100)
    train_obs = train_env.reset()
    train_steps = np.zeros(num_envs, dtype=int)

    test_epinfo_buf = deque(maxlen=100)
    test_obs = test_env.reset()
    test_steps = np.zeros(num_envs, dtype=int)

    nbatch = num_envs * nsteps
    # Due to some bug-like code in baseline.ppo2,
    # (and I modified PFRL accordingly) the true batch size is
    # nbatch // batch_size.
    n_ops_per_update = nbatch * nepochs / (nbatch // batch_size)
    nupdates = max_steps // nbatch
    max_steps = max_steps // num_envs

    logger.info('Start training for {} steps (approximately {} updates)'.format(
        max_steps, nupdates))

    tstart = time.perf_counter()
    for step_cnt in range(max_steps):

        # Roll-out in the training environments.
        assert agent.training
        train_obs, train_steps, train_epinfo = rollout_one_step(
            agent=agent,
            env=train_env,
            obs=train_obs,
            steps=train_steps,
        )
        train_epinfo_buf.extend(train_epinfo)

        # Roll-out in the test environments.
        with agent.eval_mode():
            assert not agent.training
            test_obs, test_steps, test_epinfo = rollout_one_step(
                agent=agent,
                env=test_env,
                obs=test_obs,
                steps=test_steps,
            )
            test_epinfo_buf.extend(test_epinfo)

        assert agent.training
        num_ppo_updates = agent.n_updates // n_ops_per_update

        if (step_cnt + 1) % nsteps == 0:
            tnow = time.perf_counter()
            fps = int(nbatch / (tnow - tstart))

            logger.logkv('steps', step_cnt + 1)
            logger.logkv('total_steps', (step_cnt + 1) * num_envs)
            logger.logkv('fps', fps)
            logger.logkv('num_ppo_update', num_ppo_updates)
            logger.logkv('ep_reward_mean',
                         safe_mean([info['r'] for info in train_epinfo_buf]))
            logger.logkv('ep_len_mean',
                         safe_mean([info['l'] for info in train_epinfo_buf]))
            logger.logkv('eval_ep_reward_mean',
                         safe_mean([info['r'] for info in test_epinfo_buf]))
            logger.logkv('eval_ep_len_mean',
                         safe_mean([info['l'] for info in test_epinfo_buf]))
            train_stats = agent.get_statistics()
            for stats in train_stats:
                logger.logkv(stats[0], stats[1])
            logger.dumpkvs()

            if num_ppo_updates % save_interval == 0:
                save_agent(agent, model_dir)

            tstart = time.perf_counter()

    # Save the final model.
    logger.info('Training done.')
    save_agent(agent, model_dir)


def save_agent(agent, saving_dir):
    if type(agent) == PPO:
        model_path = os.path.join(saving_dir, 'model.pt')
        agent.model.save_to_file(model_path)
        logger.info(f"Model saved to {model_path}")
    elif type(agent) == EnsembleAgent:
        agent.save(saving_dir)
        logger.info(f"Model saved to {saving_dir}/agent.pkl")
    else:
        raise RuntimeError 


def load_agent(agent, load_path, plot_dir=None):
    if type(agent) == PPO:
        agent.model.load_from_file(load_path)
        logger.info(f"Model loaded from {load_path}")
    elif type(agent) == EnsembleAgent:
        EnsembleAgent.load(load_path, plot_dir=plot_dir)
    else:
        raise RuntimeError


if __name__ == '__main__':
    trial = ProcgenTrial()
    trial.train()
