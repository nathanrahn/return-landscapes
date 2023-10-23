import os
from tqdm import tqdm
from wandb.apis.public import Api
from jaxrl.datasets.parallel_replay_buffer import ParallelReplayBuffer

from jaxrl.networks import policies, critic_net
from jaxrl.agents.ddpg.actor import update as update_ddpg_actor
from jaxrl.agents.sac.actor import update as update_sac_actor
from jaxrl.networks.common import Model
import jaxrl.agents.sac.temperature as temperature
from jax.random import KeyArray
from typing import Callable, Dict
from states.states import STATES_10K as STATES
import copy

import numpy as np
import jax
import jax.numpy as jnp
from brax.envs.env import State
from brax import envs
from brax.envs import env as brax_env

import optax

import flax

from brax import jumpy as jp

from jaxrl.utils import make_env

import gym

from pathlib import Path

import matplotlib.pyplot as plt

from jaxrl.networks.policies import sample_actions

import logging

import matplotlib.pyplot as plt


import h5py
import math

import shutil

import glob

log = logging.getLogger(__name__)


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
    is_sac = False
):
    if action_noise is not None or parameter_noise is not None:
        raise NotImplementedError

    def eval_policy(actor, env: gym.Env, episode_length: int = 1000):
        def generate_action(actor, observation):
            action = actor.apply({'params': actor.params}, observation, 0.0)
            if is_sac:
                rng = jax.random.PRNGKey(0)
                rng, key = jax.random.split(rng)
                action = action.sample(seed=key)
            return action
        generate_action = jax.jit(jax.vmap(generate_action, in_axes=(0, 0)))
        observation, done = env.reset(), np.zeros(env.num_envs, dtype=bool)
        ret, length = np.zeros(env.num_envs), 0
        while not done.all():
            action = generate_action(actor, observation)
            observation, reward, env_done, info = env.step(action)

            ret += reward * ~done
            done = done | env_done

            length += 1
            if length >= episode_length:
                break
        return ret, ret
    return eval_policy


class PolicyVariance:
    def __init__(
        self,
        run_name,
        ckpt_indexes,
        eval_episodes,
        batch_size,
        episode_length,
        seed,
        noise_type,
        noise_amounts,
    ):
        self.run_name = run_name
        self.ckpt_indexes = ckpt_indexes
        self.eval_episodes = eval_episodes
        self.batch_size = batch_size
        self.episode_length = episode_length
        self.seed = seed
        self.noise_type = noise_type
        self.noise_amounts = noise_amounts # list

    def filter_log(self):
        logger = logging.getLogger("root")

        class CheckTypesFilter(logging.Filter):
            def filter(self, record):
                return "check_types" not in record.getMessage()

        logger.addFilter(CheckTypesFilter())

    def copy_h5s_from_directory(self, src, dest):
        src, dest = str(src), str(dest)
        for file in glob.glob(os.path.join(src, "*.h5")):
            shutil.copy(file, dest)

    def copy_file(self, src, dest):
        src, dest = str(src), str(dest)
        shutil.copy(src, dest)

    def construct_temp(
        self, path, key
    ):
        temp = Model.create(
            temperature.Temperature(0.0),
            inputs=[key],
            tx=optax.adam(learning_rate=0.0),
        )
        return temp.load(path)

    def construct_actor(self, experiment, path, key, hidden_dims, action_dim, observations, actor_lr):
        if experiment["algo"] == "ddpg":
            actor_def = policies.MSEPolicy(hidden_dims, action_dim)
        elif experiment["algo"] == "sac":
            actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim)
        else:
            raise ValueError(f"Unknown algo: {experiment['algo']}")
        actor = Model.create(
            actor_def,
            inputs=[key, observations],
            tx=optax.adam(learning_rate=actor_lr),
        )
        return actor.load(path)

    def construct_critic(
        self, experiment, path, key, hidden_dims, observations, actions, num_critics
    ):
        if experiment["algo"] == "ddpg" or experiment["algo"] == "sac":
            critic_def = critic_net.DoubleCritic(hidden_dims)
        else:
            raise ValueError(f"Unknown algo: {experiment['algo']}")
        critic = Model.create(
            critic_def,
            inputs=[key, observations, actions],
        )
        return critic.load(path)

    def construct_buffer(self, experiment, RESULT_DIR, dummy_env, ckpt_index):
        try:
            replay_buffer_size = int(experiment["config"]["replay_buffer_size"])
        except KeyError:
            replay_buffer_size = int(experiment["replay_buffer_size"])

        replay_buffer = ParallelReplayBuffer(
            observation_dim=dummy_env.observation_space.shape[-1],
            action_dim=dummy_env.action_space.shape[-1],
            dummy_state=dummy_env.get_state(),
            capacity=replay_buffer_size,
            num_seeds=1,
        )
        replay_buffer.load(f"{str(RESULT_DIR)}/buffer")
        replay_buffer.size = ckpt_index
        return replay_buffer

    def run(self, output_dir, this_job_result_dir):
        self.filter_log()

        api = Api()
        runs = api.runs(path="nonseq-exp/nonseq-exp", filters={"display_name": self.run_name})
        run = runs[0]

        for ckpt_index in tqdm(self.ckpt_indexes, desc="Checkpoints", position=0):
            for noise_amount in self.noise_amounts:
                self.run_once(run, noise_amount, ckpt_index)

    def run_once(self, run, noise_amount, ckpt_index):
        experiment = run.config["experiment"]

        dummy_env = make_env(experiment["env_name"], 0, num_envs=1)
        dummy_env.reset()

        RESULT_DIR = os.path.join(run.config["result_dir"], f"seed{self.seed}")

        # Load the actor
        rng = jax.random.PRNGKey(0)
        rng, key = jax.random.split(rng)
        actor_path = os.path.join(
            RESULT_DIR,
            f"actor_ckpt_{ckpt_index}",
        )
        # Very important that this be a tuple because otherwise the actor apply_fn can't be hashed
        # and therefore doesn't work with jit
        hidden_dims = tuple(experiment[f"{experiment['algo']}_config"]["hidden_dims"])
        actor_lr = experiment[f"{experiment['algo']}_config"]["actor_lr"]
        actions = dummy_env.action_space.sample()[np.newaxis]
        action_dim = actions.shape[-1]
        observations = dummy_env.observation_space.sample()[np.newaxis]
        actor = self.construct_actor(
            experiment,
            actor_path,
            key,
            hidden_dims,
            action_dim,
            observations,
            noise_amount if self.noise_type == "td3_optimization" or self.noise_type == "sac_optimization" else actor_lr,
        )

        if self.noise_type == "td3_optimization":
            rng, key = jax.random.split(rng)
            critic_path = os.path.join(RESULT_DIR, f"critic_ckpt_{ckpt_index}")
            replay_buffer = self.construct_buffer(experiment, RESULT_DIR, dummy_env, ckpt_index)
            self.critic = self.construct_critic(experiment, critic_path, key, hidden_dims, observations, actions, 2)
            def td3_optimization_step(batch, actor):
                new_actor, actor_info = update_ddpg_actor(actor, self.critic, batch)
                return new_actor
            td3_optimization_step = jax.jit(jax.vmap(td3_optimization_step, in_axes=(0, None)))
        elif self.noise_type == "sac_optimization":
            rng, key = jax.random.split(rng)
            critic_path = os.path.join(RESULT_DIR, f"critic_ckpt_{ckpt_index}")
            replay_buffer = self.construct_buffer(experiment, RESULT_DIR, dummy_env, ckpt_index)
            self.critic = self.construct_critic(experiment, critic_path, key, hidden_dims, observations, actions, 2)
            temp_path = os.path.join(RESULT_DIR, f"temp_ckpt_{ckpt_index}")
            rng, key = jax.random.split(rng)
            self.temp = self.construct_temp(temp_path, key)
            def sac_optimization_step(batch, actor, temp, key):
                new_actor, actor_info = update_sac_actor(key, actor, self.critic, temp, batch)
                return new_actor
            sac_optimization_step = jax.jit(jax.vmap(sac_optimization_step, in_axes=(0, None, None, None)))

        dtype = jnp.zeros(1).dtype
        undis_returns = np.zeros(self.eval_episodes, dtype=dtype)
        returns = np.zeros(self.eval_episodes, dtype=dtype)

        eval_policy = get_evaluate_policy_fn(is_sac=(self.noise_type == "sac_optimization"))
        # Iterate through all the eval episodes in batches of size batch_size
        base_actor = copy.deepcopy(actor)
        for batch in tqdm(list(range(math.ceil(self.eval_episodes / self.batch_size))), desc=str(noise_amount), position=1, leave=False):
            start = batch * self.batch_size
            end = min((batch + 1) * self.batch_size, self.eval_episodes)

            if self.noise_type == "td3_optimization":
                batches = replay_buffer.sample_parallel_multibatch(experiment["actor_batch_size"], end - start)
                # Remove seed dimension
                batches = jax.tree_map(lambda x: x.squeeze(), batches)
                actor = td3_optimization_step(batches, base_actor)
            elif self.noise_type == "sac_optimization":
                batches = replay_buffer.sample_parallel_multibatch(experiment["actor_batch_size"], end - start)
                # Remove seed dimension
                batches = jax.tree_map(lambda x: x.squeeze(), batches)
                rng, key = jax.random.split(rng)
                actor = sac_optimization_step(batches, base_actor, self.temp, key)

            env = make_env(experiment["env_name"], 0, num_envs=end-start, same_seeds=True)

            undis_rets, rets = eval_policy(
                actor,
                env
            )

            # Save the returns
            returns[start:end] = rets
            undis_returns[start:end] = undis_rets

        os.umask(0)

        with h5py.File(Path(RESULT_DIR) / f"s0_returns_{ckpt_index}.h5", "a") as f:
            dset_name = f"{self.noise_type}/{noise_amount}/{self.eval_episodes}"
            if dset_name in f:
                del f[dset_name]
            f.require_dataset(dset_name, (self.eval_episodes,), dtype=dtype)
            f[dset_name][:] = returns

            dset_name = f"und_returns/{self.noise_type}/{noise_amount}/{self.eval_episodes}"
            if dset_name in f:
                del f[dset_name]
            f.require_dataset(dset_name, (self.eval_episodes,), dtype=dtype)
            f[dset_name][:] = undis_returns