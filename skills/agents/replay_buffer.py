import random
import collections


Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer():
    def __init__(self, max_memory=10000):
        self._memory = collections.deque([], maxlen=max_memory)
        self._max_length = max_memory

    def add(self, transition):
        self._memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self._memory, batch_size)

    def __len__(self):
        return len(self._memory)

    def is_empty(self):
        return len(self._memory) == 0

    def is_full(self):
        return len(self._memory) == self._max_length
