import os

import pandas
import seaborn as sns
import matplotlib.pyplot as plt


def plot_transfer_exp_training_curve_across_levels(exp_dir):
    """
    x-axis: steps in each level
    y-axis: reward, averaged across different levels
    """
    rewards = []
    for agent in os.listdir(exp_dir):
        agent_dir = os.path.join(exp_dir, agent)
        if not os.path.isdir(agent_dir):
            continue
        for seed in os.listdir(agent_dir):
            seed_dir = os.path.join(agent_dir, seed)
            csv_path = os.path.join(seed_dir, 'progress.csv')
            assert os.path.exists(csv_path)
            df = pandas.read_csv(csv_path, comment='#')
            df = df[['level_total_steps', 'level_index', 'ep_reward_mean']].copy()
            df['agent'] = agent
            df['seed'] = int(seed)
            rewards.append(df)
    rewards = pandas.concat(rewards, ignore_index=True)
    # average across different level_index
    rewards = rewards.groupby(['level_total_steps', 'agent', 'seed']).mean().reset_index()

    # plot
    sns.lineplot(
        data=rewards,
        x='level_total_steps',
        y='ep_reward_mean',
        hue='agent',
        style='agent',
    )
    plt.title('Training Curve Averaged Across Levels')
    plt.xlabel('Steps')
    plt.ylabel('Episodic Reward')
    save_path = os.path.dirname(exp_dir) + '/training_curve.png'
    plt.savefig(save_path)
    print(f'saved to {save_path}')
    plt.close()


def plot_transfer_exp_eval_curve(exp_dir):
    """
    x-axis: levels
    y-axis: eval reward at that level, averaged across the last few timesteps
    """
    rewards = []
    for agent in os.listdir(exp_dir):
        agent_dir = os.path.join(exp_dir, agent)
        if not os.path.isdir(agent_dir):
            continue
        for seed in os.listdir(agent_dir):
            seed_dir = os.path.join(agent_dir, seed)
            csv_path = os.path.join(seed_dir, 'progress.csv')
            assert os.path.exists(csv_path)
            df = pandas.read_csv(csv_path, comment='#')
            df = df[['level_total_steps', 'eval_ep_reward_mean', 'level_index']].copy()
            df = df.groupby('level_index').tail(20)  # only keep the last 20 timesteps
            df = df.groupby('level_index').mean().reset_index()  # and mean across those timesteps
            df['agent'] = agent
            df['seed'] = int(seed)
            rewards.append(df)
    rewards = pandas.concat(rewards, ignore_index=True)

    # plot
    sns.lineplot(
        data=rewards,
        x='level_index',
        y='eval_ep_reward_mean',
        hue='agent',
        style='agent',
    )
    plt.title('Eval Reward after Trained on Level 1 - k')
    plt.xlabel('Level')
    plt.ylabel('Eval Reward (averaged over last 20 steps at level k)')
    plt.xticks(range(len(rewards['level_index'].unique())))
    save_path = os.path.dirname(exp_dir) + '/eval_curve.png'
    plt.savefig(save_path)
    print(f'saved to {save_path}')
    plt.close()


def plot_reward_curve(csv_dir):
    """
    this is used to plot for a single agent
    read progress.csv and plot the reward curves, save in the save dir as csv
    """
    csv_path = os.path.join(csv_dir, 'progress.csv')
    df = pandas.read_csv(csv_path, comment='#')
    steps = df['total_steps']
    train_reward = df['ep_reward_mean']
    eval_reward = df['eval_ep_reward_mean']
    plt.plot(steps, train_reward, label='train')
    plt.plot(steps, eval_reward, label='eval')
    plt.legend()
    plt.title('Learning Curve')
    plt.xlabel('Steps')
    plt.ylabel('Episodic Reward')
    save_path = os.path.dirname(csv_path) + '/learning_curve.png'
    plt.savefig(save_path)
    plt.close()


def plot_train_eval_curve(exp_dir, kind='eval'):
    """
    plot the eval-curve of ensemble 1 and ensemble 3 and compare
    """
    assert kind in ['eval', 'train']
    keyword = 'eval_ep_reward_mean' if kind == 'eval' else 'ep_reward_mean'
    rewards = []
    for agent in os.listdir(exp_dir):
        if agent not in ['ensemble-1', 'ensemble-3']:
            continue
        agent_dir = os.path.join(exp_dir, agent)
        for seed in os.listdir(agent_dir):
            seed_dir = os.path.join(agent_dir, seed)
            csv_path = os.path.join(seed_dir, 'progress.csv')
            assert os.path.exists(csv_path)
            df = pandas.read_csv(csv_path, comment='#')
            df = df[['total_steps', keyword]].copy()
            sparsity = 5  # only plot every 4 points
            df = df[df.total_steps % (sparsity * 800) == 0]
            df[[keyword]] = df[[keyword]].rolling(20).mean()  # rolling mean to denoise
            df['agent'] = agent
            df['seed'] = int(seed)
            rewards.append(df)

    rewards = pandas.concat(rewards, ignore_index=True)

    # plot
    sns.lineplot(
        data=rewards,
        x='total_steps',
        y=keyword,
        hue='agent',
        style='agent'
    )
    plt.title(f'{kind} Curve')
    plt.xlabel('Steps')
    plt.ylabel('Episodic Reward')
    save_path = os.path.dirname(exp_dir) + f'/{kind}_curve.png'
    plt.savefig(save_path)
    print(f'saved to {save_path}')
    plt.close()


def plot_all_agents_reward_data(exp_dir):
    """
    given an experiments dir, find all the subdirs that represent different agents
    and gather their eval_ep_reward_mean data
    """
    rewards = []
    for agent in os.listdir(exp_dir):
        agent_dir = os.path.join(exp_dir, agent)
        if not os.path.isdir(agent_dir):
            continue
        for seed in os.listdir(agent_dir):
            seed_dir = os.path.join(agent_dir, seed)
            csv_path = os.path.join(seed_dir, 'progress.csv')
            assert os.path.exists(csv_path)
            df = pandas.read_csv(csv_path, comment='#')
            # df = df[df['total_steps'] % 32000 == 0]

            eval_df = df[['total_steps', 'eval_ep_reward_mean']].copy()
            eval_df['seed'] = int(seed)
            eval_df['agent'] = agent
            eval_df['kind'] = 'eval'
            eval_df.rename(columns={'eval_ep_reward_mean': 'reward'}, copy=False, inplace=True)

            train_df = df[['total_steps', 'ep_reward_mean']].copy()
            train_df['seed'] = int(seed)
            train_df['agent'] = agent
            train_df['kind'] = 'train'
            train_df.rename(columns={'ep_reward_mean': 'reward'}, copy=False, inplace=True)

            new_df = pandas.concat([eval_df, train_df], ignore_index=True)
            rewards.append(new_df)
    rewards = pandas.concat(rewards, ignore_index=True)

    # plot
    sns.lineplot(
        data=rewards,
        x='total_steps',
        y='reward',
        hue='agent',
        style='kind',
    )
    plt.title(f'Learning Curve: {exp_dir}')
    plt.xlabel('Steps')
    plt.ylabel('Episodic Reward')
    save_path = os.path.join(exp_dir, 'learning_curve.png')
    plt.savefig(save_path)
    plt.close()


def plot_all_agents_generalization_gap(exp_dir):
    """
    given an experiment dir, find all the subdirs that represent different agents
    and plot the difference between the training reward curve and the eval reward curve
    """
    rewards = []
    for agent in os.listdir(exp_dir):
        agent_dir = os.path.join(exp_dir, agent)
        if not os.path.isdir(agent_dir):
            continue
        for seed in os.listdir(agent_dir):
            seed_dir = os.path.join(agent_dir, seed)
            csv_path = os.path.join(seed_dir, 'progress.csv')
            assert os.path.exists(csv_path)
            df = pandas.read_csv(csv_path, comment='#')

            new_df = df[['total_steps']].copy()
            new_df['seed'] = int(seed)
            new_df['agent'] = agent
            new_df['reward_diff'] = df['ep_reward_mean'] - df['eval_ep_reward_mean']
            rewards.append(new_df)
        rewards = pandas.concat(rewards, ignore_index=True)

    # plot
    sns.lineplot(
        data=rewards,
        x='total_steps',
        y='reward_diff',
        hue='agent',
        style='agent',
    )
    plt.title(f'Generalization Gap: {exp_dir}')
    plt.xlabel('Steps')
    plt.ylabel('Episodic Training Reward - Episodic Eval Reward')
    save_path = os.path.join(exp_dir, 'generalization_gap.png')
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', '-l', required=True, help='path to the csv file')
    parser.add_argument('--compare', '-c', action='store_true', help='compare all agents in the same dir', default=False)
    parser.add_argument('--gap', '-g', action='store_true', help='plot the generalization gap', default=False)
    parser.add_argument('--evaluation', '-e', action='store_true', help='plot the evaluation curve', default=False)
    parser.add_argument('--train', '-t', action='store_true', help='plot the training curve', default=False)
    parser.add_argument('--transfer', '-f', action='store_true', help='plot the transfer curve', default=False)
    args = parser.parse_args()
    if args.compare:
        plot_all_agents_reward_data(args.load)
    elif args.gap:
        plot_all_agents_generalization_gap(args.load)
    elif args.evaluation:
        plot_train_eval_curve(args.load, kind='eval')
    elif args.train:
        plot_train_eval_curve(args.load, kind='train')
    elif args.transfer:
        plot_transfer_exp_eval_curve(args.load)
        plot_transfer_exp_training_curve_across_levels(args.load)
    else:
        plot_reward_curve(args.load)
