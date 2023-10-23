from experiments.train_ppo import Actor as PPOActor, Critic as PPOCritic, Stuff, rollout, compute_gae, update_ppo, AgentParams, EpisodeStatistics, Storage
from experiments import train_ppo
from jaxrl.agents.ddpg.actor import update as update_ddpg_actor
from jaxrl.agents.sac.actor import update as update_sac_actor
from jaxrl.datasets.parallel_replay_buffer import ParallelReplayBuffer
from jaxrl.networks import policies, critic_net
import jaxrl.agents.sac.temperature as temperature
import jax.numpy as jnp
from jaxrl.networks.common import Model
import optax
import jax
import flax
import zipfile
from brax import envs
from jaxrl.wrappers import wrap_for_training
from flax.training.train_state import TrainState
import functools
import os
import numpy as np
from wandb.apis.public import Api
from jaxrl.utils import make_env
from brax.envs.env import State
from brax.envs import env as brax_env
from brax import jumpy as jp
from typing import Callable, Dict
from jax.random import KeyArray
import psutil
import sys
import gc


def clear_caches():
    process = psutil.Process()
    if process.memory_info().vms > 4 * 2**30:  # >4GB memory usage
        for module_name, module in sys.modules.items():
            if module_name.startswith("jax"):
                for obj_name in dir(module):
                    obj = getattr(module, obj_name)
                    if hasattr(obj, "cache_clear"):
                        obj.cache_clear()
        gc.collect()


vmapped_rollout = jax.vmap(rollout, in_axes=(None, None, None, None, None,
                                             0, None, None, None, None,
                                             None, None, None))
copy_actor = jax.vmap(lambda _, ac: ac, in_axes=(0, None))

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
    episode_length: int = 1000,
    discount: float = 0.99,
    is_ppo: bool = False,
    is_sac: bool = False,
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
            params = actor.params

            if is_ppo:
                action, _ = actor.apply({"params": params}, state.obs)
            elif is_sac:
                dist = actor.apply({'params': params}, state.obs, temperature=0)
                rng, key = jax.random.split(rng)
                action = dist.sample(seed=key)
            else:
                action = actor.apply({"params": params}, state.obs, temperature)
            action = jnp.clip(action, -1, 1)

            next_state = eval_step_fn(state, action, discount=discount)
            return (next_state, rng)

        last_state, last_rng = jax.lax.fori_loop(0, episode_length, one_step, (state, key))
        eval_metrics = last_state.info["eval_metrics"]
        return eval_metrics.raw_metrics["reward"], eval_metrics.discounted_metrics["reward"]
    eval_policy = jax.jit(jax.vmap(eval_policy, in_axes=(0, 0, None, 0)), static_argnums=(2,))
    return eval_policy


def get_custom_env(env_name, **kwargs):
    # Get the raw brax env
    env = envs._envs[env_name](**kwargs)

    # VmapWrapper to parallelize rollouts
    # env = envs.wrappers.VmapWrapper(env)

    # Custom wrapper to keep track of single returns
    env = SingleEpisodeEvalWrapper(env)

    return env


def get_empty_stuff(dummy_state, actor, critic):
    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            actor.init(jax.random.PRNGKey(0), dummy_state.obs.squeeze()),
            critic.init(jax.random.PRNGKey(1), dummy_state.obs.squeeze())
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(0.1),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=0.1, eps=1e-5
            ),
        ),
    )
    episode_stats = EpisodeStatistics(None, None, None, None)
    storage = Storage(0, 0, 0, 0, 0, 0, 0, 0, dummy_state, 0)
    dummy_stuff = Stuff(agent_state, episode_stats, 0, 0, storage, dummy_state, *(12*[0]))
    return dummy_stuff


def construct_ppo_step_env(experiment):
    key = jax.random.PRNGKey(0)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    key_envs = jax.random.split(key, experiment['num_envs'] // 1)
    key_envs = jnp.reshape(key_envs, (1, -1) + key_envs.shape[1:])

    # env setup
    env = envs.get_environment(env_name=experiment['env_id'], init_state_noise_scale=1)
    env = wrap_for_training(env, episode_length=1000, action_repeat=1)
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_env_fn = jax.jit(jax.vmap(env.step))

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
    handle, _ = reset(key_envs)
    return step_env_wrapped, handle


def construct_buffer(experiment, RESULT_DIR, dummy_env, ckpt_index):
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


def construct_critic(
    experiment, path, key, hidden_dims, observations, actions, num_critics
):
    if "algo" in experiment:
        if experiment["algo"] == "ddpg":
            critic_def = critic_net.DoubleCritic(hidden_dims)
        elif experiment["algo"] == "sac":
            critic_def = critic_net.DoubleCritic(hidden_dims)
        else:
            raise ValueError(f"Unknown algo: {experiment['algo']}")
    else:
        critic_def = critic_net.DoubleCritic(hidden_dims)
    critic = Model.create(
        critic_def,
        inputs=[key, observations, actions],
    )
    return critic.load(path)


def construct_temp(
    path, key
):
    temp = Model.create(
        temperature.Temperature(0.0),
        inputs=[key],
        tx=optax.adam(learning_rate=0.0),
    )
    return temp.load(path)


def construct_actor(experiment, path, key, hidden_dims, action_dim, observations, actor_lr):
    is_ppo = False
    if "algo" in experiment:
        if experiment["algo"] == "ddpg":
            actor_def = policies.MSEPolicy(hidden_dims, action_dim)
        elif experiment["algo"] == "sac":
            actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim)
        else:
            raise ValueError(f"Unknown algo: {experiment['algo']}")
    elif "gae_lambda" in experiment:
        # It's ppo
        is_ppo = True
        actor_def = train_ppo.Actor(action_dim)
    else:
        raise ValueError(f"unknown algorithm type")
    actor = Model.create(
        actor_def,
        inputs=[key, observations],
        tx=optax.adam(learning_rate=actor_lr),
    )
    return actor.load(path, no_opt=is_ppo)


def construct_update_fn(noise_type, noise_amount, experiment, key, hidden_dims, action_dim,
                        observations, actions, ckpt_index, dummy_env, RESULT_DIR, rng):
    if noise_type == "init_parameter":
        def apply_noise(key, actor):
            return actor.replace(params=jax.tree_map(lambda x: x + jax.random.normal(key, x.shape) * noise_amount, actor.params))
        update_fn = jax.jit(jax.vmap(apply_noise, in_axes=(0, None)))
        update_stuff = None
    elif noise_type == "td3_optimization":
        rng, key = jax.random.split(rng)
        critic_path = os.path.join(RESULT_DIR, f"critic_ckpt_{ckpt_index}")
        replay_buffer = construct_buffer(experiment, RESULT_DIR, dummy_env, ckpt_index)
        critic = construct_critic(experiment, critic_path, key, hidden_dims, observations, actions, 2)
        def td3_optimization_step(batch, actor):
            new_actor, actor_info = update_ddpg_actor(actor, critic, batch)
            return new_actor
        update_fn = jax.jit(jax.vmap(td3_optimization_step, in_axes=(0, None)))
        update_stuff = replay_buffer
    elif noise_type == "sac_optimization":
        rng, key = jax.random.split(rng)
        critic_path = os.path.join(RESULT_DIR, f"critic_ckpt_{ckpt_index}")
        replay_buffer = construct_buffer(experiment, RESULT_DIR, dummy_env, ckpt_index)
        critic = construct_critic(experiment, critic_path, key, hidden_dims, observations, actions, 2)
        temp_path = os.path.join(RESULT_DIR, f"temp_ckpt_{ckpt_index}")
        rng, key = jax.random.split(rng)
        temp = construct_temp(temp_path, key)
        def sac_optimization_step(batch, actor, temp, key):
            new_actor, actor_info = update_sac_actor(key, actor, critic, temp, batch)
            return new_actor
        update_fn = jax.jit(jax.vmap(sac_optimization_step, in_axes=(0, None, None, None)))
        update_stuff = (replay_buffer, temp)
    elif noise_type == "ppo_optimization":
        critic_def = PPOCritic()
        actor_def = PPOActor(action_dim=env.action_size)
        dummy_critic_def = PPOCritic()
        dummy_actor_def = PPOActor(action_dim=env.action_size)
        step_env_wrapped, ppo_state = construct_ppo_step_env(experiment)
        # Load "stuff" with training state
        with zipfile.ZipFile(RESULT_DIR+'.zip', "r") as zipf:
            with zipf.open(f"stuff_{ckpt_index}") as f:
                stuff = flax.serialization.from_bytes(get_empty_stuff(ppo_state, dummy_actor_def, dummy_critic_def), f.read())
        update_stuff = (stuff.agent_state, stuff.episode_stats, stuff.next_obs, stuff.next_done,
                        stuff.storage, stuff.handle, stuff.global_step, stuff.num_envs, stuff.num_steps, stuff.gamma,
                        stuff.gae_lambda, stuff.norm_adv, stuff.clip_coef, stuff.ent_coef, stuff.vf_coef,
                        stuff.update_epochs, stuff.batch_size, stuff.minibatch_size,
                        actor_def, critic_def, step_env_wrapped)
        @functools.partial(jax.jit, static_argnums=(5,))
        @functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None))
        def ppo_optimization_step(agent_state, next_obs, next_done, storage, key, critic_def):
            storage = compute_gae(agent_state, next_obs, next_done, storage,
                                  critic_def, stuff.num_steps, stuff.gamma, stuff.gae_lambda)
            agent_state, *_ = update_ppo(
                agent_state, storage, key,
                stuff.norm_adv, stuff.clip_coef, stuff.ent_coef, stuff.vf_coef,
                1, stuff.minibatch_size, stuff.minibatch_size, actor_def, critic_def
            )
            return agent_state
        update_fn = ppo_optimization_step
    else:
        raise NotImplementedError
    return update_fn, update_stuff


def update_policy(noise_type, experiment, rng, base_actor, update_fn, update_stuff, start, end, actor_lr, observations):
    if noise_type == "init_parameter":
        rng, *keys = jax.random.split(rng, end - start + 1)
        actor = update_fn(jnp.array(keys), base_actor)
    elif noise_type == "td3_optimization":
        replay_buffer = update_stuff
        batches = replay_buffer.sample_parallel_multibatch(experiment["actor_batch_size"], end - start)
        # Remove seed dimension
        batches = jax.tree_map(lambda x: x.squeeze(), batches)
        actor = update_fn(batches, base_actor)
    elif noise_type == "sac_optimization":
        replay_buffer, temp = update_stuff
        batches = replay_buffer.sample_parallel_multibatch(experiment["actor_batch_size"], end - start)
        # Remove seed dimension
        batches = jax.tree_map(lambda x: x.squeeze(), batches)
        rng, key = jax.random.split(rng)
        actor = update_fn(batches, base_actor, temp, key)
    elif noise_type == "ppo_optimization":
        (agent_state, episode_stats, next_obs, next_done, storage, handle, global_step, num_envs, num_steps, gamma,
         gae_lambda, norm_adv, clip_coef, ent_coef, vf_coef, update_epochs, batch_size, minibatch_size,
         actor_def, critic_def, step_env_wrapped) = update_stuff
        rng, *keys = jax.random.split(rng, end - start + 1)
        (new_agent_state, new_episode_stats, new_next_obs, new_next_done, new_storage,
            new_key, new_handle, new_global_step) = vmapped_rollout(
                                                agent_state, episode_stats, next_obs, next_done, storage,
                                                jnp.array(keys), handle, global_step, num_envs, num_steps,
                                                actor_def, critic_def, step_env_wrapped)
        new_agent_state = update_fn(new_agent_state, new_next_obs, new_next_done, new_storage, new_key, critic_def)
        # Construct from the agent_state and the actor_def a model like the one required by eval_policy
        actor = Model.create(
            actor_def,
            inputs=[jax.random.PRNGKey(0), observations],
            tx=optax.adam(learning_rate=actor_lr),
        )
        actor = copy_actor(jnp.arange(end - start), actor)
        actor = actor.replace(params=new_agent_state.params.actor_params["params"])
    return actor


def get_actor_variables(run_name, seed, ckpt, side_info=False):
    api = Api()
    runs = api.runs(path="nonseq-exp/nonseq-exp", filters={"display_name": run_name})
    run = runs[0]
    experiment = run.config["experiment"]
    try:
        envname = experiment["env_name"]
    except KeyError:
        envname = experiment["env_id"]

    env = get_custom_env(envname, init_state_noise_scale=0.0)
    env.reset = jax.jit(env.reset)
    env.step = jax.jit(env.step)

    dummy_env = make_env(envname, 0, num_envs=1, init_state_noise_scale=0.0)
    dummy_env.reset()

    RESULT_DIR = os.path.join(run.config["result_dir"], f"seed{seed}")

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    try:
        hidden_dims = tuple(experiment[f"{experiment['algo']}_config"]["hidden_dims"])
    except KeyError:
        hidden_dims = None

    try:
        actor_lr = experiment[f"{experiment['algo']}_config"]["actor_lr"]
    except KeyError:
        actor_lr = 0.0

    actor_path = os.path.join(
        RESULT_DIR,
        f"actor_ckpt_{ckpt}",
    )

    actions = dummy_env.action_space.sample()[np.newaxis]
    action_dim = actions.shape[-1]
    observations = dummy_env.observation_space.sample()[np.newaxis]

    if side_info:
        return (experiment, actor_path, key, hidden_dims, action_dim, observations, 0.0), (experiment, RESULT_DIR, rng, env)
    else:
        return experiment, actor_path, key, hidden_dims, action_dim, observations, 0.0


def get_evaluate_policy_with_action_interpolation(
    episode_length: int = 1000,
    discount: float = 0.99,
    temperature: float = 1.0,
    vmap_alpha: bool = False,
    is_ppo: bool = False,
    is_sac: bool = False,
):

    def eval_policy(
        state: State,
        actor1: Model,
        actor2: Model,
        alpha: float,
        eval_step_fn: Callable,
    ):

        def one_step(i, state):
            params1 = actor1.params
            params2 = actor1.params

            if is_ppo:
                action1, _ = actor1.apply({"params": params1}, state.obs)
                action2, _ = actor1.apply({"params": params2}, state.obs)
            elif is_sac:
                dist = actor1.apply({'params': params1}, state.obs, temperature=0)
                action1 = dist.sample(seed=jax.random.PRNGKey(0))

                dist = actor2.apply({'params': params2}, state.obs, temperature=0)
                action2 = dist.sample(seed=jax.random.PRNGKey(0))
            else:
                action1 = actor1.apply({"params": params1}, state.obs, temperature)
                action2 = actor2.apply({"params": params2}, state.obs, temperature)
            action = alpha * action1 + (1 - alpha) * action2
            action = jnp.clip(action, -1, 1)
            next_state = eval_step_fn(state, action, discount=discount)
            return next_state

        last_state = jax.lax.fori_loop(0, episode_length, one_step, state)
        eval_metrics = last_state.info["eval_metrics"]
        return eval_metrics.raw_metrics["reward"], eval_metrics.discounted_metrics["reward"]
    if vmap_alpha:
        eval_policy = jax.vmap(eval_policy, in_axes=(None, None, None, 0, None))
    eval_policy = jax.jit(eval_policy, static_argnums=(4,))
    return eval_policy