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


def grab_all_agents_reward_data(exp_dir):
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
        df = df[['total_steps', 'eval_ep_reward_mean']]
        df['agent'] = agent
        rewards.append(df)
    rewards = pandas.concat(rewards, ignore_index=True)

    # plot
    sns.lineplot(
        data=rewards,
        x='total_steps',
        y='eval_ep_reward_mean',
        hue='agent',
        style='agent',
    )
    plt.title(f'Learning Curve: {exp_dir}')
    plt.xlabel('Steps')
    plt.ylabel('Eval Episodic Reward')
    save_path = os.path.join(exp_dir, 'learning_curve.png')
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', '-l', required=True, help='path to the csv file')
    parser.add_argument('--compare', '-c', action='store_true', help='compare all agents in the same dir', default=False)
    args = parser.parse_args()
    if args.compare:
        grab_all_agents_reward_data(args.load)
    else:
        plot_reward_curve(args.load)
