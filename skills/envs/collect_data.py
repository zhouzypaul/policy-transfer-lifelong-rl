"""
collect screen to train initiation and termination datasets
"""
import numpy as np

from . import AntBoxEnv, AntBridgeEnv, AntGoalEnv


def collect_from_ant_box():
    env = AntBoxEnv()
    o = env.reset()
    # place ant in a starting position
    for x in np.linspace(0, 9, 10):
    # for x in np.linspace(-9, 9, 20):
        for y in np.linspace(-9, 25, 30):
            pos = (x, y)
            env.place_ant(pos)
            env.render_camera()


def collect_from_ant_bridge():
    env = AntBridgeEnv()
    o = env.reset()
    # place ant in a starting position
    for x in np.linspace(-9, 9, 20):
        for y in np.linspace(-9, 9, 20):
            pos = (x, y)
            print(pos)
            env.place_ant(pos)
            env.render_camera()


def collect_fron_ant_goal():
    env = AntGoalEnv()
    o = env.reset()
    # place ant in a starting position
    for x in np.linspace(-9, 9, 20):
        for y in np.linspace(-9, 9, 20):
            pos = (x, y)
            print(pos)
            env.place_ant(pos)
            env.render_camera()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='box')
    args = parser.parse_args()

    if args.env == 'box':
        collect_from_ant_box()
    elif args.env == 'bridge':
        collect_from_ant_bridge()
    elif args.env == 'goal':
        collect_fron_ant_goal()

