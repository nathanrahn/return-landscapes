"""Implementations of algorithms for continuous control."""

import functools
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.ddpg.actor import update as update_actor
from jaxrl.agents.ddpg.critic import update as update_critic
from jaxrl.agents.sac.critic import target_update
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey


@functools.partial(jax.jit, static_argnames=("init_parameter_noise",))
@functools.partial(jax.vmap, in_axes=(0, 0, 0, None))
def update_perturbations(rng, done, parameter_perturbation, init_parameter_noise):
    rng, key = jax.random.split(rng)
    parameter_perturbation = jax.tree_map(
            lambda x: done * (jax.random.normal(key, x.shape) * init_parameter_noise) + (1 - done) * x,
            parameter_perturbation
    )
    return rng, parameter_perturbation


def _update_jit(
    rng: PRNGKey,
    actor: Model,
    target_actor: Model,
    critic: Model,
    target_critic: Model,
    batch: Batch,
    discount: float,
    policy_noise: float,
    noise_clip: float,
    tau: float,
    update_target: bool,
    actor_batch_size: int,
    critic_batch_size: int,
) -> Tuple[Model, Model, Model, Model, PRNGKey, InfoDict]:
    # Select only first critic_batch_size elements from batch
    critic_batch = Batch(
        batch.observations[:critic_batch_size],
        batch.actions[:critic_batch_size],
        batch.rewards[:critic_batch_size],
        batch.masks[:critic_batch_size],
        batch.next_observations[:critic_batch_size],
    )
    rng, key = jax.random.split(rng, 2)
    new_critic, critic_info = update_critic(
        key, target_actor, critic, target_critic, critic_batch, discount, policy_noise, noise_clip
    )
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    # Select only first critic_batch_size elements from batch
    actor_batch = Batch(
        batch.observations[:actor_batch_size],
        batch.actions[:actor_batch_size],
        batch.rewards[:actor_batch_size],
        batch.masks[:actor_batch_size],
        batch.next_observations[:actor_batch_size],
    )
    if update_target:
        new_actor, actor_info = update_actor(actor, new_critic, actor_batch)
    else:
        new_actor, actor_info = actor, {}

    if update_target:
        new_target_actor = target_update(new_actor, target_actor, tau)
    else:
        new_target_actor = target_actor

    return (
        new_actor,
        new_target_actor,
        new_critic,
        new_target_critic,
        rng,
        {
            **critic_info,
            **actor_info,
        },
    )


@functools.partial(jax.jit, static_argnames=("update_target", "actor_batch_size", "critic_batch_size"))
def _do_multiple_updates(
    rng: PRNGKey,
    actor: Model,
    target_actor: Model,
    critic: Model,
    target_critic: Model,
    batches: Batch,
    discount: float,
    policy_noise: float,
    noise_clip: float,
    tau: float,
    update_target: bool,
    actor_batch_size: int,
    critic_batch_size: int,
    step,
    num_updates: int,
) -> Tuple[int, PRNGKey, Model, Model, Model, Model, InfoDict]:
    def one_step(i, state):
        step, rng, actor, target_actor, critic, target_critic, info = state
        step = step + 1
        new_actor, new_target_actor, new_critic, new_target_critic, new_rng, info = _update_jit(
            rng,
            actor,
            target_actor,
            critic,
            target_critic,
            jax.tree_map(lambda x: jnp.take(x, i, axis=1), batches),
            discount,
            policy_noise,
            noise_clip,
            tau,
            update_target,
            actor_batch_size,
            critic_batch_size,
        )
        return step, new_rng, new_actor, new_target_actor, new_critic, new_target_critic, info

    step, rng, actor, target_actor, critic, target_critic, info = one_step(
        0, (step, rng, actor, target_actor, critic, target_critic, {})
    )
    return jax.lax.fori_loop(
        1, num_updates, one_step, (step, rng, actor, target_actor, critic, target_critic, info)
    )


class SimpleDDPGLearner(object):
    def __init__(
        self,
        seed: int,
        actor: Model,
        critic: Model,
        target_actor: Model,
        target_critic: Model,
        discount: float = 0.99,
        tau: float = 0.005,
        target_update_period: int = 1,
        exploration_noise: float = 0.1,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        update_behavior_freq: int = 1,
        parameter_noise: float = 0.0,
        init_parameter_noise: float = 0.0,
        **kwargs
    ):
        """TD3 implementation (https://arxiv.org/abs/1802.09477).
        """

        self.rng = jax.random.PRNGKey(seed)
        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.update_behavior_freq = update_behavior_freq
        self.parameter_noise = parameter_noise
        self.init_parameter_noise = init_parameter_noise
        self.parameter_perturbations = jax.tree_map(lambda x: jnp.zeros_like(x), actor.params)

        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic

        self.behavior_params = self.actor.params

        self.step = 1

    def sample_actions(self, observation: np.ndarray, temperature: float = 1.0, done: np.ndarray = None) -> jnp.ndarray:
        # Update the parameter_perturbations at the indeces in which is done with a sample from a normal distribution
        if done is not None and self.init_parameter_noise > 0:
            self.rng, self.parameter_perturbations = update_perturbations(self.rng, done,
                                                                          self.parameter_perturbations,
                                                                          self.init_parameter_noise)
        #rng, action = policies.sample_actions(
        #    self.rng,
        #    self.actor.apply_fn,
        #    self.behavior_params,
        #    observation,
        #    temperature,
        #    distribution="det",
        #    parameter_noise=self.parameter_noise,
        #    perturbations=self.parameter_perturbations
        #)
        action = self.actor.apply({'params': self.behavior_params}, observation, temperature)

        # TODO probably more efficient to do this directly in jax when using brax
        action = np.asarray(action)
        if self.exploration_noise > 0:
            action = (
                action + np.random.normal(size=action.shape) * self.exploration_noise * temperature
            )
        return np.clip(action, -1, 1)

    def agent_params(self):
        return {
            'actor': self.actor,
            'critic': self.critic,
            'target_actor': self.target_actor,
            'target_critic': self.target_critic,
            'behavior_params': self.behavior_params,
            'step': self.step
        }

    def set_agent_params(self, params):
        self.actor = params['actor']
        self.critic = params['critic']
        self.target_actor = params['target_actor']
        self.target_critic = params['target_critic']
        self.behavior_params = params['behavior_params']
        self.step = params['step']

    def update(
        self, batch: Batch, actor_batch_size: int, critic_batch_size: int, num_updates: int = 1
    ) -> InfoDict:
        (
            _,
            self.rng,
            self.actor,
            self.target_actor,
            self.critic,
            self.target_critic,
            info,
        ) = _do_multiple_updates(
            rng=self.rng,
            actor=self.actor,
            target_actor=self.target_actor,
            critic=self.critic,
            target_critic=self.target_critic,
            batches=batch,
            discount=self.discount,
            policy_noise=self.policy_noise,
            noise_clip=self.noise_clip,
            tau=self.tau,
            update_target=self.step % self.target_update_period == 0,
            actor_batch_size=actor_batch_size,
            critic_batch_size=critic_batch_size,
            step=self.step,
            num_updates=num_updates,
        )
        self.step += num_updates
        if self.step % self.update_behavior_freq == 0:
            self.behavior_params = self.actor.params
        return info

