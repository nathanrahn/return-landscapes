from jaxrl.networks.common import Model
from jax.random import KeyArray
from typing import Callable, Dict
import copy

import numpy as np
import jax
import jax.numpy as jnp
from brax.envs.env import State
from brax import envs
from brax.envs import env as brax_env

import flax

from brax import jumpy as jp

from jaxrl.utils import make_env

from functools import partial

from jaxrl.networks.policies import sample_actions

@flax.struct.dataclass
class SimpleEvalMetrics:
    raw_metrics: Dict[str, jp.ndarray]
    discounted_metrics: Dict[str, jp.ndarray]
    current_discount: jp.ndarray
    is_done: jp.ndarray


def zero_metrics_like(metrics_tree):
    eval_metrics = SimpleEvalMetrics(
        raw_metrics=jax.tree_map(jp.zeros_like, metrics_tree),
        discounted_metrics=jax.tree_map(jp.zeros_like, metrics_tree),
        current_discount=jax.tree_map(jp.ones_like, metrics_tree),
        is_done=jax.tree_map(lambda x: jnp.zeros_like(x, dtype=bool), metrics_tree),
    )
    return eval_metrics


class SingleEpisodeEvalWrapper(brax_env.Wrapper):
    """Computes raw and discounted returns for a single episode, without flattening the batch dimension."""

    def reset(self, rng: jp.ndarray) -> brax_env.State:
        reset_state = self.env.reset(rng)
        reset_state.metrics["reward"] = reset_state.reward
        reset_state.info["eval_metrics"] = zero_metrics_like(reset_state.metrics)
        return reset_state

    def step(self, state: brax_env.State, action: jp.ndarray, discount=0.99) -> brax_env.State:
        state_metrics = state.info["eval_metrics"]

        nstate = self.env.step(state, action)
        nstate.metrics["reward"] = nstate.reward

        raw_metrics = jax.tree_map(
            # Only keep adding to the return when the episode is not done
            lambda a, b, d: a + b * ~d,
            state_metrics.raw_metrics,
            nstate.metrics,
            state_metrics.is_done,
        )
        current_discount = jax.tree_map(lambda a: a * discount, state_metrics.current_discount)
        discounted_metrics = jax.tree_map(
            # Only keep adding to the discounted return when the episode is not done
            lambda a, b, gamma, d: a + b * gamma * ~d,
            state_metrics.discounted_metrics,
            nstate.metrics,
            current_discount,
            state_metrics.is_done,
        )
        is_done = jax.tree_map(lambda s: s | nstate.done.astype(bool), state_metrics.is_done)

        eval_metrics = SimpleEvalMetrics(
            raw_metrics=raw_metrics,
            discounted_metrics=discounted_metrics,
            current_discount=current_discount,
            is_done=is_done,
        )
        nstate.info["eval_metrics"] = eval_metrics

        return nstate

def get_evaluate_policy_fn(
    action_noise = None,
    parameter_noise = None,
    episode_length: int = 1000,
    discount: float = 0.99,
    temperature: float = 1.0,
    vmap_actor: bool = False
):

    def eval_policy(
        state: State,
        actor: Model,
        eval_step_fn: Callable,
        rng: KeyArray,
    ):
        rng, key = jax.random.split(rng)

        def one_step(i, state_rng):
            state, rng = state_rng

            if parameter_noise is not None:
                rng, key = jax.random.split(rng)
                params = jax.tree_map(lambda x: x + jax.random.normal(key, x.shape) * parameter_noise, actor.params)
            else:
                params = actor.params

            action = actor.apply({"params": params}, state.obs, temperature)

            if action_noise is not None:
                rng, key = jax.random.split(rng)
                action = action + jax.random.normal(key, action.shape) * action_noise * temperature

            # TODO: Same as during learning?
            action = jnp.clip(action, -1, 1)

            next_state = eval_step_fn(state, action, discount=discount)
            return (next_state, rng)

        last_state, last_rng = jax.lax.fori_loop(0, episode_length, one_step, (state, key))
        eval_metrics = last_state.info["eval_metrics"]
        return eval_metrics.raw_metrics["reward"], eval_metrics.discounted_metrics["reward"]
    if vmap_actor:
        eval_policy = jax.vmap(eval_policy, in_axes=(0, 0, None, 0))
    else:
        eval_policy = jax.vmap(eval_policy, in_axes=(0, None, None, 0))
    eval_policy = jax.jit(eval_policy, static_argnums=(2,))
    return eval_policy
