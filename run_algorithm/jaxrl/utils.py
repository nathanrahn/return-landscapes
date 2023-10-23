from typing import Optional
import gym
from jaxrl.wrappers import BraxEvalWrapper, BraxGymWrapper, AutoResetWrapper
from brax import envs
import jax


def create_brax_env(env_name: str,
                    episode_length: int = 1000,
                    action_repeat: int = 1,
                    auto_reset: bool = True,
                    batch_size: Optional[int] = None,
                    eval_metrics: bool = False,
                    seed: int = 0,
                    **kwargs) -> envs.env.Env:
    """Creates an Env with a specified brax system."""
    env = envs._envs[env_name](**kwargs)
    if episode_length is not None:
        env = envs.wrappers.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        if eval_metrics:
            env = envs.wrappers.VectorWrapper(env, batch_size)
        else:
            env = envs.wrappers.VmapWrapper(env)
    if auto_reset:
        env = AutoResetWrapper(env)
    # ATTENTION: BraxEvalWrapper requires AutoResetWrapper to be the one right before it
    if eval_metrics:
        env = BraxEvalWrapper(env)
    else:
        env = BraxGymWrapper(env, seed=seed, num_seeds=batch_size)
    return env


def make_env(env_name: str,
             seed: int,
             eval_episodes: Optional[int] = None,
             eval_episode_length: int = 1000,
             num_envs: Optional[int] = None) -> gym.Env:
    if eval_episodes:
        env = create_brax_env(env_name=env_name, episode_length=eval_episode_length, eval_metrics=True,
                              batch_size=eval_episodes)
        env.step = jax.jit(env.step)
        env.reset = jax.jit(env.reset)
        return env
    else:
        env = create_brax_env(env_name=env_name, episode_length=eval_episode_length, batch_size=num_envs,
                              auto_reset=False, eval_metrics=False, seed=seed)
        return env
