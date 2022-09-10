import os

import pandas
import seaborn as sns
import matplotlib.pyplot as plt


def plot_reward_curve(csv_dir):
    """
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


def plot_all_agents_reward_data(exp_dir):
    """
    given an experiments dir, find all the subdirs that represent different agents
    and gather their eval_ep_reward_mean data
    """
    rewards = []
    for agent in os.listdir(exp_dir):
        sub_dir = os.path.join(exp_dir, agent)
        if not os.path.isdir(sub_dir):
            continue
        csv_path = os.path.join(sub_dir, 'progress.csv')
        assert os.path.exists(csv_path)
        df = pandas.read_csv(csv_path, comment='#')
        # df = df[df['total_steps'] % 32000 == 0]

        eval_df = df[['total_steps', 'eval_ep_reward_mean']].copy()
        eval_df['agent'] = agent
        eval_df['kind'] = 'eval'
        eval_df.rename(columns={'eval_ep_reward_mean': 'reward'}, copy=False, inplace=True)

        train_df = df[['total_steps', 'ep_reward_mean']].copy()
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
        sub_dir = os.path.join(exp_dir, agent)
        if not os.path.isdir(sub_dir):
            continue
        csv_path = os.path.join(sub_dir, 'progress.csv')
        assert os.path.exists(csv_path)
        df = pandas.read_csv(csv_path, comment='#')

        new_df = df[['total_steps']].copy()
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
    args = parser.parse_args()
    if args.compare:
        plot_all_agents_reward_data(args.load)
        plot_all_agents_generalization_gap(args.load)
    else:
        plot_reward_curve(args.load)
