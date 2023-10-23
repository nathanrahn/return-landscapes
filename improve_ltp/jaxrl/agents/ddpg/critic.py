from typing import Tuple

import jax.numpy as jnp
import jax

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey


def update(key: PRNGKey, actor: Model, critic: Model, target_critic: Model, batch: Batch,
           discount: float, policy_noise: float, noise_clip: float) -> Tuple[Model, InfoDict]:
    next_actions = actor(batch.next_observations)
    # Use target policy smoothing regularization
    noise = (
             jax.random.normal(key, shape=next_actions.shape) * policy_noise
    ).clip(-noise_clip, noise_clip)
    next_actions = (next_actions + noise).clip(-1, 1)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + discount * batch.masks * next_q

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply({'params': critic_params}, batch.observations, batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
