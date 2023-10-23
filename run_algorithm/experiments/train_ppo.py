# adapted from https://github.com/vwxyzjn/cleanrl/pull/313/commits/5d4c95d66285627ab8439302cb7c67eba7df2aed
import tqdm
from environment import get_output_dir
import zipfile
import os
import random
from typing import Sequence

from brax import envs
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import zeros, orthogonal
from flax.training.train_state import TrainState
from jaxrl.wrappers import wrap_for_training
from functools import partial
from jaxrl.networks.common import Model
from jaxrl.utils import make_env
from experiments.train_jaxrl import prefix_dict_keys
from brax.envs.env import State
from typing import Any
import functools
import wandb
from typing import Callable, Dict


@partial(jax.jit, static_argnums=(1, 2, 3, 5))
@partial(jax.vmap, in_axes=(0, None, None, None, 0, None))
def evaluate_brax(actor: Model, eval_step_fn: Callable, eval_reset_fn: Callable,
                  num_episodes: int, rng, episode_length: int = 1000) -> Dict[str, float]:
    rng, key = jax.random.split(rng)
    state = eval_reset_fn(rng=key)
    def one_step(i, state_ret):
        state, ret = state_ret
        action, _ = actor.apply({'params': actor.params}, state.obs)
        action = jnp.clip(action, -1, 1)
        next_state = eval_step_fn(state, action)
        ret = ret + next_state.reward
        return (next_state, ret)
    ret = jnp.zeros(num_episodes)
    last_state, ret = jax.lax.fori_loop(0, episode_length, one_step, (state, ret))
    eval_metrics = last_state.info['eval_metrics']
    avg_episode_length = (
          eval_metrics.completed_episodes_steps /
          eval_metrics.completed_episodes)
    metrics = dict(
          dict({
              f'eval/episode_{name}': value / eval_metrics.completed_episodes
              for name, value in eval_metrics.completed_episodes_metrics.items()
          }),
          **dict({
              'eval/completed_episodes': eval_metrics.completed_episodes,
              'eval/avg_episode_length': avg_episode_length
          }))
    metrics['return'] = metrics["eval/episode_reward"]
    return metrics


def evaluate(agent, env: gym.Env, num_episodes: int, episode_length: int) -> Dict[str, float]:
    if 'brax' in str(type(env)).lower():
        return evaluate_brax(agent.actor, env.step, env.reset, num_episodes, agent.rng, episode_length)
    returns = []
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        ret, length = 0, 0
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, reward, done, info = env.step(action)
            ret += reward
            length += 1
            if length >= episode_length:
                break
        returns.append(ret)
    return {'return': float(np.mean(returns))}


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=zeros)(x)
        critic = nn.tanh(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=zeros)(critic)
        critic = nn.tanh(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1), bias_init=zeros)(critic)
        return critic


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=zeros)(x)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=zeros)(actor_mean)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=zeros)(actor_mean)
        actor_logstd = self.param("actor_logstd", zeros, (1, self.action_dim))
        return actor_mean, actor_logstd


@flax.struct.dataclass
class AgentParams:
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict

@flax.struct.dataclass
class FakeAgent:
    actor: Model
    rng: jax.random.PRNGKey


@flax.struct.dataclass
class SaveStateNoOptState:
    params: flax.core.FrozenDict[str, Any]


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array
    next_states: State
    masks: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


@flax.struct.dataclass
class Stuff:
    agent_state: TrainState
    episode_stats: EpisodeStatistics
    next_obs: jnp.ndarray
    next_done: np.ndarray
    storage: Storage
    handle: jnp.ndarray
    global_step: int
    num_envs: int
    num_steps: int
    gamma: float
    gae_lambda: float
    norm_adv: bool
    clip_coef: float
    ent_coef: float
    vf_coef: float
    update_epochs: int
    batch_size: int
    minibatch_size: int


@functools.partial(jax.jit, static_argnames=("actor", "critic"))
def get_action_and_value(
    agent_state: TrainState,
    next_obs: np.ndarray,
    next_done: np.ndarray,
    storage: Storage,
    step: int,
    key: jax.random.PRNGKey,
    actor,
    critic
):
    action_mean, action_logstd = actor.apply(agent_state.params.actor_params, next_obs)
    action_std = jnp.exp(action_logstd)
    key, subkey = jax.random.split(key)
    action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
    logprob = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
    value = critic.apply(agent_state.params.critic_params, next_obs)
    storage = storage.replace(
        obs=storage.obs.at[step].set(next_obs),
        dones=storage.dones.at[step].set(next_done),
        actions=storage.actions.at[step].set(action),
        logprobs=storage.logprobs.at[step].set(logprob.sum(1)),
        values=storage.values.at[step].set(value.squeeze()),
    )
    return storage, action, key


@functools.partial(jax.jit, static_argnames=("actor", "critic"))
def get_action_and_value2(
    params: flax.core.FrozenDict,
    x: np.ndarray,
    action: np.ndarray,
    actor,
    critic
):
    action_mean, action_logstd = actor.apply(params.actor_params, x)
    action_std = jnp.exp(action_logstd)
    logprob = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
    entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
    value = critic.apply(params.critic_params, x).squeeze()
    return logprob.sum(1), entropy, value


@functools.partial(jax.jit, static_argnames=("critic", "num_steps", "gamma", "gae_lambda"))
def compute_gae(
    agent_state: TrainState,
    next_obs: np.ndarray,
    next_done: np.ndarray,
    storage: Storage,
    critic,
    num_steps,
    gamma,
    gae_lambda
):
    storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))
    next_value = critic.apply(agent_state.params.critic_params, next_obs).squeeze()
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - storage.dones[t + 1]
            nextvalues = storage.values[t + 1]
        delta = storage.rewards[t] + gamma * nextvalues * nextnonterminal - storage.values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        storage = storage.replace(advantages=storage.advantages.at[t].set(lastgaelam))
    storage = storage.replace(returns=storage.advantages + storage.values)
    return storage


@functools.partial(jax.jit, static_argnames=("norm_adv", "clip_coef",
                                             "ent_coef", "vf_coef",
                                             "update_epochs", "batch_size",
                                             "minibatch_size", "actor", "critic"))
def update_ppo(
    agent_state: TrainState,
    storage: Storage,
    key: jax.random.PRNGKey,
    norm_adv,
    clip_coef,
    ent_coef,
    vf_coef,
    update_epochs,
    batch_size,
    minibatch_size,
    actor,
    critic
):
    b_obs = storage.obs.reshape((-1,) + (storage.obs.shape[-1],))
    b_logprobs = storage.logprobs.reshape(-1)
    b_actions = storage.actions.reshape((-1,) + (storage.actions.shape[-1],))
    b_advantages = storage.advantages.reshape(-1)
    b_returns = storage.returns.reshape(-1)

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns):
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a, actor, critic)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()
        if norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
    for _ in range(update_epochs):
        key, subkey = jax.random.split(key)
        b_inds = jax.random.permutation(subkey, batch_size, independent=True)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]
            (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                agent_state.params,
                b_obs[mb_inds],
                b_actions[mb_inds],
                b_logprobs[mb_inds],
                b_advantages[mb_inds],
                b_returns[mb_inds],
            )
            agent_state = agent_state.apply_gradients(grads=grads)
    return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key


@functools.partial(jax.jit, static_argnames=("num_envs", "num_steps", "actor", "critic", "step_env_wrapped"))
def rollout(agent_state, episode_stats, next_obs, next_done, storage, key, handle, global_step,
            num_envs, num_steps, actor, critic, step_env_wrapped):
    def body_fun(i, carry):
        agent_state, episode_stats, next_obs, next_done, storage, key, handle, global_step = carry
        global_step += 1 * num_envs
        storage, action, key = get_action_and_value(agent_state, next_obs, next_done, storage, i, key, actor, critic)

        # TRY NOT TO MODIFY: execute the game and log data.
        action = jnp.clip(action, -1, 1)
        episode_stats, handle, (next_obs, reward, next_done, _) = step_env_wrapped(episode_stats, handle, action)
        storage = storage.replace(rewards=storage.rewards.at[i].set(reward))
        # Set the next state at the i-th position to the handle
        storage = storage.replace(next_states=jax.tree_map(lambda x, y: x.at[i].set(y[0]), storage.next_states, handle))
        # Set masks according to the following logic: return ~dones.astype("bool") | infos["truncation"].astype("bool")
        storage = storage.replace(masks=storage.masks.at[i].set(~next_done.astype("bool") | handle.info['truncation'][0].astype("bool")))
        return agent_state, episode_stats, next_obs, next_done, storage, key, handle, global_step

    carry = (agent_state, episode_stats, next_obs, next_done, storage, key, handle, global_step)
    carry = jax.lax.fori_loop(0, num_steps, body_fun, carry)
    return carry


class TrainPPO:
    def __init__(self, seed, env_id, total_timesteps, learning_rate, num_envs, num_steps, anneal_lr, gamma,
                 gae_lambda, num_minibatches, update_epochs, num_seeds, norm_adv, clip_coef, clip_vloss, ent_coef,
                 vf_coef, max_grad_norm, target_kl, agent_num_logs, eval_episodes):
        self.seed = seed
        self.num_seeds = num_seeds
        self.env_id = env_id
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.agent_num_logs = agent_num_logs
        self.eval_episodes = eval_episodes
        self.save_dir = get_output_dir()

        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_updates = self.total_timesteps // self.batch_size
        self.log_freq = int(self.num_updates // self.agent_num_logs)

    def make_seed_dirs(self):
        for seed in range(self.num_seeds):
            seed_dir = self.save_dir / f"seed{seed}"
            seed_dir.mkdir(exist_ok=True, parents=True)

    def run_seed(self, seed):
        # logs = []
        def linear_schedule(count):
            # anneal learning rate linearly after one training iteration which contains
            # (self.num_minibatches * self.update_epochs) gradient updates
            frac = 1.0 - (count // (self.num_minibatches * self.update_epochs)) / self.num_updates
            return self.learning_rate * frac
        random.seed(seed)
        np.random.seed(seed)

        key = jax.random.PRNGKey(seed)
        key, network_key, actor_key, critic_key = jax.random.split(key, 4)
        key_envs = jax.random.split(key, self.num_envs // 1)
        key_envs = jnp.reshape(key_envs, (1, -1) + key_envs.shape[1:])

        # env setup
        env = envs.get_environment(env_name=self.env_id)
        env = wrap_for_training(env, episode_length=1000, action_repeat=1)
        reset_fn = jax.jit(jax.vmap(env.reset))
        step_env_fn = jax.jit(jax.vmap(env.step))
        # eval_env = make_env(self.env_id, seed + 42, eval_episodes=self.eval_episodes)
        fake_env = make_env(self.env_id, seed, num_envs=1)

        @jax.jit
        def reset(key_envs):
            handle = reset_fn(key_envs)
            return handle, handle.obs.squeeze()
        @jax.jit
        def step_env(handle, actions):
            handle = step_env_fn(handle, actions.reshape(1, *actions.shape))
            return handle, (handle.obs.squeeze(), handle.reward.squeeze(), handle.done.squeeze(), handle.info)
        def step_env_wrapped(episode_stats, handle, action):
            handle, (next_obs, reward, next_done, info) = step_env(handle, action)
            new_episode_return = episode_stats.episode_returns + reward
            new_episode_length = episode_stats.episode_lengths + 1
            episode_stats = episode_stats.replace(
                episode_returns=(new_episode_return) * (1 - next_done),
                episode_lengths=(new_episode_length) * (1 - next_done),
                # only update the `returned_episode_returns` if the episode is done
                returned_episode_returns=jnp.where(
                    next_done, new_episode_return, episode_stats.returned_episode_returns
                ),
                returned_episode_lengths=jnp.where(
                    next_done, new_episode_length, episode_stats.returned_episode_lengths
                ),
            )
            return episode_stats, handle, (next_obs, reward, next_done, info)

        handle, next_obs = reset(key_envs)
        episode_stats = EpisodeStatistics(
            episode_returns=jnp.zeros(self.num_envs, dtype=jnp.float32),
            episode_lengths=jnp.zeros(self.num_envs, dtype=jnp.float32),
            returned_episode_returns=jnp.zeros(self.num_envs, dtype=jnp.float32),
            returned_episode_lengths=jnp.zeros(self.num_envs, dtype=jnp.float32),
        )

        actor = Actor(action_dim=env.action_size)
        critic = Critic()
        agent_state = TrainState.create(
            apply_fn=None,
            params=AgentParams(
                actor.init(actor_key, next_obs),
                critic.init(critic_key, next_obs),
            ),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(
                    learning_rate=linear_schedule if self.anneal_lr else self.learning_rate, eps=1e-5
                ),
            ),
        )
        key, actor_key = jax.random.split(key)
        observations = fake_env.observation_space.sample()[np.newaxis]
        eval_model = Model.create(actor, inputs=[actor_key, observations], tx=optax.adam(learning_rate=0.0))

        # ALGO Logic: Storage setup
        storage = Storage(
            obs=jnp.zeros((self.num_steps, self.num_envs) + (env.observation_size,)),
            actions=jnp.zeros((self.num_steps, self.num_envs) + (env.action_size,)),
            logprobs=jnp.zeros((self.num_steps, self.num_envs)),
            dones=jnp.zeros((self.num_steps, self.num_envs)),
            values=jnp.zeros((self.num_steps, self.num_envs)),
            advantages=jnp.zeros((self.num_steps, self.num_envs)),
            returns=jnp.zeros((self.num_steps, self.num_envs)),
            rewards=jnp.zeros((self.num_steps, self.num_envs)),
            next_states=jax.tree_map(
                lambda x: jnp.expand_dims(jnp.zeros_like(x),
                                          axis=0).repeat(self.num_steps, axis=0).squeeze(), handle),
            masks=jnp.ones((self.num_steps, self.num_envs))
        )


        # TRY NOT TO MODIFY: start the game
        global_step = 0
        next_done = np.zeros(self.num_envs)

        pbar = tqdm.tqdm(range(1, self.num_updates + 1))

        for update in pbar:
            if update % self.log_freq == 0:
                stuff = (agent_state, episode_stats, next_obs, next_done, storage,
                         handle, global_step, self.num_envs, self.num_steps,
                         self.gamma, self.gae_lambda, self.norm_adv, self.clip_coef, self.ent_coef, self.vf_coef,
                         self.update_epochs, self.batch_size, self.minibatch_size)
                # Construct a stuff dataclass with the above variables
                stuff = Stuff(*stuff)
                data = flax.serialization.to_bytes(stuff)
                # Save data to a file
                with zipfile.ZipFile(os.path.join(self.save_dir, f"seed{seed}.zip"), "a") as zipf:
                    with zipf.open(f"stuff_{global_step}", "w", force_zip64=True) as f:
                        f.write(data)

            agent_state, episode_stats, next_obs, next_done, storage, key, handle, global_step = rollout(
                agent_state, episode_stats, next_obs, next_done, storage,
                key, handle, global_step, self.num_envs, self.num_steps, actor, critic, step_env_wrapped
            )
            storage = compute_gae(agent_state, next_obs, next_done, storage,
                                  critic, self.num_steps, self.gamma, self.gae_lambda)
            # I want to insert evenly spaced during training
            agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
                agent_state, storage, key,
                self.norm_adv, self.clip_coef, self.ent_coef, self.vf_coef,
                self.update_epochs, self.batch_size, self.minibatch_size, actor, critic
            )

            avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
            pbar.set_description(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")
            # stat_dict = {"loss": loss, "pg_loss": pg_loss, "v_loss": v_loss, "entropy_loss": entropy_loss,
            #              "approx_kl": approx_kl, "avg_episodic_return": avg_episodic_return}
            # dict_to_log = {}
            # for info_key in stat_dict:
            #     dict_to_log[f"seed{seed}/{info_key}"] = stat_dict[info_key]

            if update % self.log_freq == 0:
                curr_params = agent_state.params.actor_params["params"]
                eval_model = eval_model.replace(params=curr_params)
                eval_model = jax.tree_map(lambda x: jnp.expand_dims(x, 0), eval_model)
                key, fake_rng = jax.random.split(key)
                fake_rng = jnp.expand_dims(fake_rng, 0)
                # fake_agent = FakeAgent(actor=eval_model, rng=fake_rng)
                # eval_stats = evaluate(fake_agent, eval_env, self.eval_episodes, 1000)
                # dict_to_log = {**dict_to_log, **prefix_dict_keys(eval_stats, f"seed{seed}/")}

                with zipfile.ZipFile(os.path.join(self.save_dir, f"seed{seed}.zip"), "a") as zipf:
                    save_state = SaveStateNoOptState(params=agent_state.params.actor_params["params"])
                    data = flax.serialization.to_bytes(save_state)
                    with zipf.open(f"actor_ckpt_{global_step}", "w", force_zip64=True) as f:
                        f.write(data)
                # logs.append((global_step, dict_to_log))
        return {}  # logs

    def run(self, output_dir, result_dir):
        self.make_seed_dirs()
        all_logs = []
        for i, seed in enumerate(range(self.seed, self.seed + self.num_seeds)):
            logs = self.run_seed(i)
            all_logs.append(logs)

        for i in range(len(all_logs[0])):
            for j in range(len(all_logs)):
                global_step, dict_to_log = all_logs[j][i]
                wandb.log(dict_to_log, step=int(global_step))
