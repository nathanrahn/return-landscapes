import gym
import numpy as np
from gym import spaces
from collections import namedtuple
State = namedtuple('state', ['obs', 'reward'])


class SequentialMultiEnvWrapper(gym.Env):
    def __init__(self, env_fns):
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_envs = len(self.envs)
        self.action_space = spaces.Box(low=self.envs[0].action_space.low[None].repeat(len(self.envs), axis=0),
                                       high=self.envs[0].action_space.high[None].repeat(len(self.envs), axis=0),
                                       shape=(len(self.envs), self.envs[0].action_space.shape[0]),
                                       dtype=self.envs[0].action_space.dtype)
        self.observation_space = spaces.Box(low=self.envs[0].observation_space.low[None].repeat(len(self.envs), axis=0),
                                            high=self.envs[0].observation_space.high[None].repeat(len(self.envs), axis=0),
                                            shape=(len(self.envs), self.envs[0].observation_space.shape[0]),
                                            dtype=self.envs[0].observation_space.dtype)

    def _reset_idx(self, idx):
        return self.envs[idx].reset()

    def reset_where_done(self, observations, dones):
        for j, done in enumerate(dones):
            if done:
                observations[j], dones[j] = self._reset_idx(j), False
        return observations, dones

    def generate_masks(self, dones, infos):
        masks = []
        for done, info in zip(dones, infos):
            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0
            masks.append(mask)
        masks = np.array(masks)
        return masks

    def reset(self):
        obs = []
        for env in self.envs:
            obs.append(env.reset())
        obs = np.stack(obs)
        self.state = State(obs=obs, reward=np.zeros(self.num_envs))
        return obs

    def step(self, actions):
        obs, rews, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            ob, reward, done, info = env.step(action)
            obs.append(ob)
            rews.append(reward)
            dones.append(done)
            infos.append(info)

        obs, rews, dones, infos = np.stack(obs), np.stack(rews), np.stack(dones), infos

        self.state = State(obs=obs, reward=rews)
        return obs, rews, dones, infos

    def get_state(self):
        return self.state


class ParallelMultiEnvWrapper(gym.Env):
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self._env = gym.vector.AsyncVectorEnv(env_fns, shared_memory=False)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset_where_done(self, observations, dones):
        for j, done in enumerate(dones):
            if done:
                observations[j], dones[j] = self.reset_at(j), False
        return observations, dones

    def reset_at(self, index: int):
        self._env._assert_is_running()
        self._env.parent_pipes[index].send(("reset", {}))
        result, success = self._env.parent_pipes[index].recv()
        self._env._raise_if_errors([success])
        return result

    def generate_masks(self, dones, infos):
        masks = []
        for done, info in zip(dones, infos):
            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0
            masks.append(mask)
        masks = np.array(masks)
        return masks

    def reset(self):
        return self._env.reset()

    def step(self, actions):
        *rest, infos = self._env.step(actions)
        return *rest, [dict(zip(infos,t)) for t in zip(*infos.values())]
