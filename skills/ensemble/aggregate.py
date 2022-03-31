"""
aggregate the policy output of the ensemble into a single action
"""
import numpy as np


def choose_most_popular(actions):
    """
    given a list of actions, choose the most popular one
    """
    counts = np.bincount(actions)
    return np.argmax(counts)


def uniform_stochastic_leader(actions):
    """
    choose a `leader` according to a uniform random distribution
    and all learners in the ensemble will listen to that leader for action selection

    the hope is eventually all learners will converge to the same action selection policy
    """
    return np.random.choice(actions)
