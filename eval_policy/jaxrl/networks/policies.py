import functools
from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from jaxrl.networks.common import MLP, Params, PRNGKey, default_init

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class MSEPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> jnp.ndarray:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        actions = nn.Dense(self.action_dim,
                           kernel_init=default_init())(outputs)
        return nn.tanh(actions)


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                    self.log_std_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        #log_stds = jnp.clip(log_stds, log_std_min, log_std_max)
        # suggested by Ilya for stability
        log_stds = log_std_min + (log_std_max - log_std_min) * 0.5 * (1 + nn.tanh(log_stds))

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        else:
            return base_dist


class NormalTanhMixturePolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    num_components: int = 5
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        logits = nn.Dense(self.action_dim * self.num_components,
                          kernel_init=default_init())(outputs)
        means = nn.Dense(self.action_dim * self.num_components,
                         kernel_init=default_init(),
                         bias_init=nn.initializers.normal(stddev=1.0))(outputs)
        log_stds = nn.Dense(self.action_dim * self.num_components,
                            kernel_init=default_init())(outputs)

        shape = list(observations.shape[:-1]) + [-1, self.num_components]
        logits = jnp.reshape(logits, shape)
        mu = jnp.reshape(means, shape)
        log_stds = jnp.reshape(log_stds, shape)

        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        components_distribution = tfd.Normal(loc=mu,
                                             scale=jnp.exp(log_stds) *
                                             temperature)

        base_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=components_distribution)

        dist = tfd.TransformedDistribution(distribution=base_dist,
                                           bijector=tfb.Tanh())

        return tfd.Independent(dist, 1)


@functools.partial(jax.jit, static_argnames=('actor_def', 'distribution', 'parameter_noise'))
@functools.partial(jax.vmap, in_axes=(0, None, 0, 0, None, None, None, 0))
def _sample_actions(
        rng: PRNGKey,
        actor_def: nn.Module,
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob',
        parameter_noise: float = 0.0,
        perturbations=None) -> Tuple[PRNGKey, jnp.ndarray]:
    if parameter_noise > 0.0:
        # Perturb the policy parameters with Gaussian noise
        actor_params = jax.tree_map(lambda x: x + jax.random.normal(rng, x.shape) * parameter_noise, actor_params)

    actor_params = jax.tree_map(lambda x, y: x + y, actor_params, perturbations)

    if distribution == 'det':
        return rng, actor_def.apply({'params': actor_params}, observations,
                                    temperature)
    else:
        dist = actor_def.apply({'params': actor_params}, observations,
                               temperature)
        rng, key = jax.random.split(rng)
        return rng, dist.sample(seed=key)


def sample_actions(
        rng: PRNGKey,
        actor_def: nn.Module,
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob',
        parameter_noise: float = 0.0,
        perturbations: np.ndarray = None) -> Tuple[PRNGKey, jnp.ndarray]:
    # If perturbations is none, set them to 0 with same shape as parameters
    if perturbations is None:
        perturbations = jax.tree_map(lambda x: jnp.zeros_like(x), actor_params)
    return _sample_actions(rng, actor_def, actor_params, observations,
                           temperature, distribution, parameter_noise,
                           perturbations)
