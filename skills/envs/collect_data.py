"""
collect screen to train initiation and termination datasets
"""
import numpy as np
import matplotlib.pyplot as plt

from . import AntBoxEnv, AntBridgeEnv, AntGoalEnv


def _place_ant_and_save_img(env, x, y, save_dir):
    pos = (x, y)
    print(pos)
    env.place_ant(pos)
    img = env.render_camera()
    plt.imsave(save_dir + '/{}_{}.png'.format(x, y), img)


def collect_from_ant_box(save_dir):
    """
    for initiation classifier:
        almost all the data collected in this function is a negative example, so just
        hand pick out the positive ones, and delete those with weird dynamics (when
        interacting with the box)
    for termination classifier:
        all pictures on the other side of the box are positive examples, while those
        on the starting side are negative. Hand pick them.
    """
    env = AntBoxEnv()
    o = env.reset()
    # place ant in a starting position
    for x in np.linspace(-9, 9, 18):
        for y in np.linspace(-9, 25, 30):
            _place_ant_and_save_img(env, x, y, save_dir)


def collect_from_ant_bridge(save_dir):
    """
    there is a large gap beteen the bridge and the ground, so 
    we teleport the ant to two different y-blocks
    """
    env = AntBridgeEnv()
    o = env.reset()
    # place ant in a starting position
    for x in np.linspace(-9, 9, 18):
        for y in np.linspace(-2, 4, 6):
            _place_ant_and_save_img(env, x, y, save_dir)
        for y in np.linspace(22, 27, 5):
            _place_ant_and_save_img(env, x, y, save_dir)


def collect_fron_ant_goal(save_dir):
    """
    teleport the ant to two differrnt y-blocks to avoid interacting with the wall
    """
    env = AntGoalEnv()
    o = env.reset()
    # place ant in a starting position
    for x in np.linspace(-9, 9, 18):
        for y in np.linspace(0, 8, 8):
            _place_ant_and_save_img(env, x, y, save_dir)
        for y in np.linspace(12, 28, 16):
            _place_ant_and_save_img(env, x, y, save_dir)


if __name__ == "__main__":
    import os
    import argparse
    from skills.utils import create_log_dir

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='box')
    parser.add_argument('--save_dir', type=str, default='results/classifier_data')
    parser.add_argument('--seed', type=int, default=4)
    args = parser.parse_args()

    # deterministic
    np.random.seed(args.seed)

    # create saving dir based on env
    save_dir = os.path.join(args.save_dir, args.env)
    create_log_dir(save_dir, remove_existing=True, log_git=False)

    if args.env == 'box':
        collect_from_ant_box(save_dir)
    elif args.env == 'bridge':
        collect_from_ant_bridge(save_dir)
    elif args.env == 'goal':
        collect_fron_ant_goal(save_dir)

