import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn, distributions
import pfrl
from pfrl.nn.lmbda import Lambda
from pfrl.utils import set_random_seed
from matplotlib import pyplot as plt

from skills import utils
from skills.agents import SAC
from skills.baseline import logger
from skills.baseline.train import load_agent, safe_mean, ProcgenTrial
from skills.envs import AntBoxEnv, AntBridgeEnv, AntGoalEnv, AntMixLongEnv


class AntTestTrial(ProcgenTrial):
    """
    execute the trained policy on ant, and look at the rendering
    """
    def parse_args(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            parents=[self.get_common_arg_parser()]
        )
        # defaults
        parser.set_defaults(hyperparams='procgen_ppo')

        # agent
        parser.add_argument('--load', '-l', type=str, default=None,
                            help='path to load agent')
        
        args = self.parse_common_args(parser)

        return args
    
    def make_ant_env(self, env_name, eval=True):
        assert 'ant' in env_name
        if env_name == "ant_box":
            env = AntBoxEnv(eval=eval)
        elif env_name == "ant_bridge":
            env = AntBridgeEnv(eval=eval)
        elif env_name == "ant_goal":
            env = AntGoalEnv(eval=eval)
        elif env_name == "ant_mixed":
            env = AntMixLongEnv(eval=eval)
        else:
            raise NotImplementedError(f"env_name {env_name} not found")
        return env
    
    def make_sac_agent(self, env):
        action_space = env.action_space
        obs_size = env.observation_space.shape[0]
        action_size = action_space.shape[0]

        def squashed_diagonal_gaussian_head(x):
            assert x.shape[-1] == action_size * 2
            mean, log_scale = torch.chunk(x, 2, dim=1)
            log_scale = torch.clamp(log_scale, -20.0, 2.0)
            var = torch.exp(log_scale * 2)
            base_distribution = distributions.Independent(
                distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
            )
            # cache_size=1 is required for numerical stability
            return distributions.transformed_distribution.TransformedDistribution(
                base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
            )

        policy = nn.Sequential(
            nn.Linear(obs_size, self.params['n_hidden_channels']),
            nn.ReLU(),
            nn.Linear(self.params['n_hidden_channels'], self.params['n_hidden_channels']),
            nn.ReLU(),
            nn.Linear(self.params['n_hidden_channels'], action_size * 2),
            Lambda(squashed_diagonal_gaussian_head),
        )
        torch.nn.init.xavier_uniform_(policy[0].weight)
        torch.nn.init.xavier_uniform_(policy[2].weight)
        torch.nn.init.xavier_uniform_(policy[4].weight)
        policy_optimizer = torch.optim.Adam(
            policy.parameters(), lr=self.params['learning_rate'], eps=self.params['adam_eps']
        )

        def make_q_func_with_optimizer():
            q_func = nn.Sequential(
                pfrl.nn.ConcatObsAndAction(),
                nn.Linear(obs_size + action_size, self.params['n_hidden_channels']),
                nn.ReLU(),
                nn.Linear(self.params['n_hidden_channels'], self.params['n_hidden_channels']),
                nn.ReLU(),
                nn.Linear(self.params['n_hidden_channels'], 1),
            )
            torch.nn.init.xavier_uniform_(q_func[1].weight)
            torch.nn.init.xavier_uniform_(q_func[3].weight)
            torch.nn.init.xavier_uniform_(q_func[5].weight)
            q_func_optimizer = torch.optim.Adam(
                q_func.parameters(), lr=self.params['learning_rate'], eps=self.params['adam_eps']
            )
            return q_func, q_func_optimizer

        q_func1, q_func1_optimizer = make_q_func_with_optimizer()
        q_func2, q_func2_optimizer = make_q_func_with_optimizer()

        rbuf = pfrl.replay_buffers.ReplayBuffer(10**6, num_steps=self.params['n_step_return'])

        def burnin_action_func():
            """Select random actions until model is updated one or more times."""
            return action_space.sample()

        # Hyperparameters in http://arxiv.org/abs/1802.09477
        agent = SAC(
            policy,
            q_func1,
            q_func2,
            policy_optimizer,
            q_func1_optimizer,
            q_func2_optimizer,
            rbuf,
            gamma=self.params['discount'],
            update_interval=self.params['update_interval'],
            replay_start_size=self.params['replay_start_size'],
            gpu=-1 if self.params['device']=='cpu' else 0,
            minibatch_size=self.params['batch_size'],
            burnin_action_func=burnin_action_func,
            entropy_target=-action_size,
            temperature_optimizer_lr=self.params['learning_rate'],
        )
        return agent

    def setup(self):
        self.check_params_validity()
        set_random_seed(self.params['seed'])

        utils.create_log_dir('results/visualize', remove_existing=True, log_git=True)
        
        # read hyperparameters 
        hyperparams_file = Path(self.params['load']) / 'hyperparams.csv'
        saved_params = utils.load_hyperparams(hyperparams_file)
        self.params = {**self.params, **saved_params}
        self.params['num_envs'] = 1

        self.eval_env = self.make_ant_env(self.params['env'], eval=True)

        # load the agent
        self.agent = self.make_sac_agent(self.eval_env)
        load_agent(self.agent, self.params['load'])
    
    def test(self):
        test_agent(
            self.agent,
            self.eval_env,
            eval_max_steps=500*4,
        )


def test_agent(
    agent,
    eval_env,
    eval_max_steps,
    log_interval=100,
):
    epinfo_buf = deque(maxlen=100)
    obs = eval_env.reset().astype(np.float32)

    for step_cnt in range(eval_max_steps):
        with agent.eval_mode():
            assert not agent.training
            
            # control loop
            action = agent.batch_act([obs])
            new_obs, reward, done, info = eval_env.step(action[0])
            new_obs = new_obs.astype(np.float32)
            img = info['obs_img']
            plt.imsave(f'./results/visualize/{step_cnt}.png', img)
            print(step_cnt, reward)

            obs = new_obs
            if done:
                obs = eval_env.reset().astype(np.float32)
    
        if (step_cnt + 1) % log_interval == 0:
            logger.logkv('steps', step_cnt + 1)
            logger.logkv('eval_ep_reward_mean',
                         safe_mean([info['r'] for info in epinfo_buf]))
            logger.logkv('eval_ep_len_mean',
                         safe_mean([info['l'] for info in epinfo_buf]))
            logger.dumpkvs()


if __name__ == '__main__':
    trial = AntTestTrial()
    trial.test()
