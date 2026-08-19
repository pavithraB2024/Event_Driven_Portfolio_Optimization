"""
Parallel Environment Wrapper for Faster Training

Implements vectorized environments for parallel data collection:
- SubprocVecEnv: Run environments in separate processes
- DummyVecEnv: Run environments in same process (for debugging)
"""

import numpy as np
import multiprocessing as mp
from typing import List, Callable, Tuple, Optional
from abc import ABC, abstractmethod


class VecEnv(ABC):
    """
    Abstract base class for vectorized environments.
    """

    def __init__(self, num_envs: int, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.closed = False

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    @abstractmethod
    def close(self):
        pass

    def get_attr(self, attr_name: str, indices=None):
        """Get attribute from environments."""
        pass

    def set_attr(self, attr_name: str, value, indices=None):
        """Set attribute in environments."""
        pass


class DummyVecEnv(VecEnv):
    """
    Vectorized environment that runs multiple environments sequentially.
    Useful for debugging and when using non-picklable environments.
    """

    def __init__(self, env_fns: List[Callable]):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.actions = None

    def reset(self):
        obs = []
        for env in self.envs:
            o, _ = env.reset()
            obs.append(o)
        return np.array(obs)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs, rewards, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, terminated, truncated, info = env.step(self.actions[i])
            done = terminated or truncated

            if done:
                # Auto-reset
                o, _ = env.reset()
                info['terminal_observation'] = o

            obs.append(o)
            rewards.append(r)
            dones.append(done)
            infos.append(info)

        return np.array(obs), np.array(rewards), np.array(dones), infos

    def close(self):
        if self.closed:
            return
        for env in self.envs:
            env.close()
        self.closed = True

    def get_attr(self, attr_name: str, indices=None):
        target_envs = self._get_target_envs(indices)
        return [getattr(env, attr_name) for env in target_envs]

    def set_attr(self, attr_name: str, value, indices=None):
        target_envs = self._get_target_envs(indices)
        for env in target_envs:
            setattr(env, attr_name, value)

    def _get_target_envs(self, indices):
        if indices is None:
            return self.envs
        return [self.envs[i] for i in indices]


def _worker(remote, parent_remote, env_fn_wrapper):
    """
    Worker process for SubprocVecEnv.
    """
    parent_remote.close()
    env = env_fn_wrapper.x()

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == 'step':
                obs, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated

                if done:
                    # Auto-reset and store terminal observation
                    terminal_obs = obs
                    obs, _ = env.reset()
                    info['terminal_observation'] = terminal_obs

                remote.send((obs, reward, done, info))

            elif cmd == 'reset':
                obs, _ = env.reset()
                remote.send(obs)

            elif cmd == 'close':
                env.close()
                remote.close()
                break

            elif cmd == 'get_attr':
                remote.send(getattr(env, data))

            elif cmd == 'set_attr':
                attr_name, value = data
                setattr(env, attr_name, value)
                remote.send(None)

            else:
                raise NotImplementedError(f"Unknown command: {cmd}")

        except EOFError:
            break


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle).
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(VecEnv):
    """
    Vectorized environment that runs multiple environments in parallel subprocesses.

    Faster than DummyVecEnv for CPU-bound environments.
    """

    def __init__(self, env_fns: List[Callable], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Use 'spawn' for compatibility (especially on Windows)
            forkserver_available = 'forkserver' in mp.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'

        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []

        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_attr', 'observation_space'))
        observation_space = self.remotes[0].recv()
        self.remotes[0].send(('get_attr', 'action_space'))
        action_space = self.remotes[0].recv()

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.array(obs)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rewards, dones, infos = zip(*results)
        return np.array(obs), np.array(rewards), np.array(dones), list(infos)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_attr(self, attr_name: str, indices=None):
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value, indices=None):
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def _get_target_remotes(self, indices):
        if indices is None:
            return self.remotes
        return [self.remotes[i] for i in indices]


def make_vec_env(
    env_fn: Callable,
    n_envs: int = 4,
    vec_env_cls: type = SubprocVecEnv,
    **kwargs
) -> VecEnv:
    """
    Create a vectorized environment.

    Args:
        env_fn: Function that creates environment
        n_envs: Number of parallel environments
        vec_env_cls: VecEnv class to use (SubprocVecEnv or DummyVecEnv)
        **kwargs: Additional arguments for vec_env_cls

    Returns:
        Vectorized environment
    """
    env_fns = [env_fn for _ in range(n_envs)]
    return vec_env_cls(env_fns, **kwargs)


class VecNormalize:
    """
    Vectorized environment wrapper for normalizing observations and rewards.
    """

    def __init__(
        self,
        venv: VecEnv,
        obs_norm: bool = True,
        ret_norm: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8
    ):
        self.venv = venv
        self.obs_norm = obs_norm
        self.ret_norm = ret_norm
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon

        self.obs_rms = RunningMeanStd(shape=venv.observation_space.shape) if obs_norm else None
        self.ret_rms = RunningMeanStd(shape=()) if ret_norm else None
        self.returns = np.zeros(venv.num_envs)

        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.num_envs = venv.num_envs

    def reset(self):
        obs = self.venv.reset()
        self.returns = np.zeros(self.num_envs)
        if self.obs_norm:
            self.obs_rms.update(obs)
            return self._normalize_obs(obs)
        return obs

    def step(self, actions):
        obs, rewards, dones, infos = self.venv.step(actions)

        if self.obs_norm:
            self.obs_rms.update(obs)
            obs = self._normalize_obs(obs)

        if self.ret_norm:
            self.returns = self.returns * self.gamma + rewards
            self.ret_rms.update(self.returns)
            rewards = self._normalize_reward(rewards)
            self.returns[dones] = 0.0

        return obs, rewards, dones, infos

    def _normalize_obs(self, obs):
        return np.clip(
            (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
            -self.clip_obs, self.clip_obs
        )

    def _normalize_reward(self, reward):
        return np.clip(
            reward / np.sqrt(self.ret_rms.var + self.epsilon),
            -self.clip_reward, self.clip_reward
        )

    def close(self):
        self.venv.close()


class RunningMeanStd:
    """
    Running mean and std calculation using Welford's algorithm.
    """

    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count
