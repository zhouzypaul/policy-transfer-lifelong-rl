"""
This file is adpated from pfrl/wrappers/atari_wrappers.py
https://github.com/pfnet/pfrl/blob/master/pfrl/wrappers/atari_wrappers.py
"""

from collections import deque

import gym
import numpy as np
from gym import spaces

import pfrl

try:
    import cv2

    cv2.ocl.setUseOpenCL(False)
    _is_cv2_available = True
except Exception:
    _is_cv2_available = False


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1
            )  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, info = self.env.step(self.noop_action)
            if done or info.get("needs_reset", False):
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for envs that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, info = self.env.step(1)
        if done or info.get("needs_reset", False):
            self.env.reset(**kwargs)
        obs, _, done, info = self.env.step(2)
        if done or info.get("needs_reset", False):
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done or info.get("needs_reset", False):
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """
        Make end-of-life == end-of-episode, but only reset on true game end.

        basically the same as pfrl, but instead of using self.needs_real_reset,
        use self.env.unwrapped.needs_real_reset
        this is so that other wrappers can change how `done` is handled, such as MonteNewGoalWrapper

        NOTE: any wrapper that changes how done is handled should set self.env.unwrapped.needs_real_reset
        this is so that those wrapper need not go before this wrapper
        """
        super().__init__(env)
        self.lives = 0
        self.env.unwrapped.needs_real_reset = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.env.unwrapped.needs_real_reset = done or info.get("needs_reset", False)
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few
            # frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.env.unwrapped.needs_real_reset:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs



class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, channel_order="hwc"):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        To use this wrapper, OpenCV-Python is required.

        this is almost the same as the pfrl version, except that in observation() it saves the
        original frame to the unwrapped env, so that FrameStack can access it
        """
        if not _is_cv2_available:
            raise RuntimeError(
                "Cannot import cv2 module. Please install OpenCV-Python to use"
                " WarpFrame."
            )
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        shape = {
            "hwc": (self.height, self.width, 1),
            "chw": (1, self.height, self.width),
        }
        self.observation_space = spaces.Box(
            low=0, high=255, shape=shape[channel_order], dtype=np.uint8
        )

    def observation(self, frame):
        self.env.unwrapped.original_frame = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return frame.reshape(self.observation_space.low.shape)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, channel_order="hwc"):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames

        the same as pfrl's, except adding code to keep track of self.env.unwrapped.original_stacked_frames
        this is so that we also have access to the original frame in the unwrapped env
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.env.unwrapped.original_stacked_frames = deque([], maxlen=k)  # (k, 210, 160, 3)
        self.stack_axis = {"hwc": 2, "chw": 0}[channel_order]
        orig_obs_space = env.observation_space
        low = np.repeat(orig_obs_space.low, k, axis=self.stack_axis)
        high = np.repeat(orig_obs_space.high, k, axis=self.stack_axis)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=orig_obs_space.dtype
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
            self.env.unwrapped.original_stacked_frames.append(self.env.unwrapped.original_frame)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        self.env.unwrapped.original_stacked_frames.append(self.env.unwrapped.original_frame)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames), stack_axis=self.stack_axis)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Divide frame values by 255.0 and return them as np.float32.
    Especially, when the original env.observation_space is np.uint8,
    this wrapper converts frame values into [0.0, 1.0] of dtype np.float32.
    """

    def __init__(self, env):
        assert isinstance(env.observation_space, spaces.Box)
        gym.ObservationWrapper.__init__(self, env)

        self.scale = 255.0

        orig_obs_space = env.observation_space
        self.observation_space = spaces.Box(
            low=self.observation(orig_obs_space.low),
            high=self.observation(orig_obs_space.high),
            dtype=np.float32,
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / self.scale


class LazyFrames(object):
    """Array-like object that lazily concat multiple frames.
    This object ensures that common frames between the observations are only
    stored once.  It exists purely to optimize memory usage which can be huge
    for DQN's 1M frames replay buffers.
    This object should only be converted to numpy array before being passed to
    the model.
    You'd not believe how complex the previous solution was.
    """

    def __init__(self, frames, stack_axis=2):
        self.stack_axis = stack_axis
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=self.stack_axis)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class FlickerFrame(gym.ObservationWrapper):
    """Stochastically flicker frames."""

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        if self.unwrapped.np_random.rand() < 0.5:
            return np.zeros_like(observation)
        else:
            return observation


def make_atari(env_id, max_frames=30 * 60 * 60):
    env = gym.make(env_id)
    assert "NoFrameskip" in env.spec.id
    assert isinstance(env, gym.wrappers.TimeLimit)
    # Unwrap TimeLimit wrapper because we use our own time limits
    env = env.env
    if max_frames:
        env = pfrl.wrappers.ContinuingTimeLimit(env, max_episode_steps=max_frames)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(
    env,
    warp_frames=True,
    episode_life=True,
    clip_rewards=True,
    frame_stack=True,
    scale=False,
    fire_reset=False,
    channel_order="chw",
    flicker=False,
):
    """Configure environment for DeepMind-style Atari."""
    if episode_life:
        env = EpisodicLifeEnv(env)
    if fire_reset and "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if warp_frames:
        env = WarpFrame(env, channel_order=channel_order)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if flicker:
        env = FlickerFrame(env)
    if frame_stack:
        env = FrameStack(env, 4, channel_order=channel_order)
    return env