from jaxrl.datasets.parallel_replay_buffer import ParallelReplayBuffer

from jaxrl.networks.common import Model
from jax.random import KeyArray
from typing import Callable, Dict, Optional
import copy

import numpy as np
import jax
import jax.numpy as jnp
from brax.envs.env import State
from brax_custom import envs
from brax_custom.envs import env as brax_env

import math

from util.mc_eval_brax import SimpleEvalMetrics, zero_metrics_like, SingleEpisodeEvalWrapper
from util.mc_eval_brax import get_evaluate_policy_fn

def purd_samples_initial_state(
        actor: Model,
        critic: Model,
        replay_buffer: ParallelReplayBuffer,
        update_fn: Callable,
        envname: str,
        eval_episodes: int,
        actor_batch_size: int,
        eval_batch_size: Optional[int] = None
):
    if eval_batch_size is None:
        eval_batch_size = eval_episodes

    rng = jax.random.PRNGKey(0)
    env = envs._envs[envname](init_state_noise_scale=0.0)
    env = SingleEpisodeEvalWrapper(env)
    env.reset = jax.jit(env.reset)
    env.step = jax.jit(env.step)

    vmapped_reset = jax.vmap(env.reset, in_axes=(0,))

    def update_step(batch, actor):
        breakpoint()
        new_actor, _ = update_fn(actor, critic, batch)
        return new_actor

    batch_update_step = jax.jit(jax.vmap(update_step, in_axes=(0, None)))

    dtype = jnp.zeros(1).dtype
    undis_returns = np.zeros(eval_episodes, dtype=dtype) 
    returns = np.zeros(eval_episodes, dtype=dtype) 

    eval_policy = get_evaluate_policy_fn(vmap_actor=True)

    base_actor = copy.deepcopy(actor)

    for batch in range(math.ceil(eval_episodes / eval_batch_size)):
        start = batch * eval_batch_size
        end = min((batch + 1) * eval_batch_size, eval_episodes)
        rng, *keys = jax.random.split(rng, end - start + 1)
        states = vmapped_reset(jnp.array(keys))
        batches = replay_buffer.sample_parallel_multibatch(actor_batch_size, end - start)
        batches = jax.tree_map(lambda x: x.squeeze(), batches)
        actor = batch_update_step(batches, base_actor)

        rng, *keys = jax.random.split(rng, end - start + 1)
        undis_rets, rets = eval_policy(
            states,
            actor,
            env.step,
            keys
        )

        # Save the returns
        returns[start:end] = rets
        undis_returns[start:end] = undis_rets

    return returns, undis_returns
