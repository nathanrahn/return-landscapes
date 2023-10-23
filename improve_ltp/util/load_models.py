import jax
import numpy as np
import optax
import os

from jax.random import KeyArray
from wandb.apis.public import Api

from jaxrl.datasets.parallel_replay_buffer import ParallelReplayBuffer
from jaxrl.networks import policies, critic_net
from jaxrl.networks.common import Model
from jaxrl.networks.policies import sample_actions
from jaxrl.utils import make_env

def get_run_by_run_name(run_name):
        api = Api()
        runs = api.runs(path="nonseq-exp/nonseq-exp", filters={"display_name": run_name})
        return runs[0]

def load_actor_td3_simple(
        run,
        ckpt,
        seed,
        include_target=False
):
        experiment = run.config["experiment"]

        try:
            envname = experiment["env_name"]
        except KeyError:
            envname = experiment["env_id"]

        dummy_env = make_env(envname, 0, num_envs=1, init_state_noise_scale=0.0)
        dummy_env.reset()

        RESULT_DIR = os.path.join(run.config["result_dir"], f"seed{seed}")

        # Load the actor
        rng = jax.random.PRNGKey(0)
        rng, key = jax.random.split(rng)
        actor_path = os.path.join(
            RESULT_DIR,
            f"actor_ckpt_{ckpt}",
        )
        # Very important that this be a tuple because otherwise the actor apply_fn can't be hashed
        # and therefore doesn't work with jit
        try:
            hidden_dims = tuple(experiment[f"{experiment['algo']}_config"]["hidden_dims"])
        except KeyError:
            hidden_dims = None

        try:
            actor_lr = experiment[f"{experiment['algo']}_config"]["actor_lr"]
        except KeyError:
            actor_lr = 0.0

        actions = dummy_env.action_space.sample()[np.newaxis]
        action_dim = actions.shape[-1]
        observations = dummy_env.observation_space.sample()[np.newaxis]

        actor = construct_actor(
            experiment,
            actor_path,
            key,
            hidden_dims,
            action_dim,
            observations,
            actor_lr,
            trainable = True
        )

        if not include_target:
            return actor, RESULT_DIR

        target_actor_path = os.path.join(
            RESULT_DIR,
            f"target_actor_ckpt_{ckpt}",
        )

        target_actor = construct_actor(
            experiment,
            target_actor_path,
            key,
            hidden_dims,
            action_dim,
            observations,
            actor_lr,
            trainable = False
        )

        return actor, target_actor, RESULT_DIR

def load_critic_td3_simple(
        run,
        ckpt,
        seed,
        include_target=False
):
        experiment = run.config["experiment"]

        try:
            envname = experiment["env_name"]
        except KeyError:
            envname = experiment["env_id"]

        dummy_env = make_env(envname, 0, num_envs=1, init_state_noise_scale=0.0)
        dummy_env.reset()

        RESULT_DIR = os.path.join(run.config["result_dir"], f"seed{seed}")

        # Load the actor
        rng = jax.random.PRNGKey(0)
        rng, key = jax.random.split(rng)
        critic_path = os.path.join(
            RESULT_DIR,
            f"critic_ckpt_{ckpt}",
        )
        # Very important that this be a tuple because otherwise the actor apply_fn can't be hashed
        # and therefore doesn't work with jit
        try:
            hidden_dims = tuple(experiment[f"{experiment['algo']}_config"]["hidden_dims"])
        except KeyError:
            hidden_dims = None

        try:
            critic_lr = experiment[f"{experiment['algo']}_config"]["critic_lr"]
        except KeyError:
            critic_lr = 0.0

        actions = dummy_env.action_space.sample()[np.newaxis]
        action_dim = actions.shape[-1]
        observations = dummy_env.observation_space.sample()[np.newaxis]

        critic = construct_critic(
            experiment,
            critic_path,
            key,
            hidden_dims,
            observations,
            actions,
            critic_lr,
            2,
            trainable = True
        )

        if not include_target:
            return critic, RESULT_DIR

        target_critic_path = os.path.join(
            RESULT_DIR,
            f"target_critic_ckpt_{ckpt}",
        )

        target_critic = construct_critic(
            experiment,
            target_critic_path,
            key,
            hidden_dims,
            observations,
            actions,
            critic_lr,
            2,
            trainable = False
        )

        return critic, target_critic, RESULT_DIR

def load_buffer(
        run,
        ckpt,
        seed
):
        experiment = run.config["experiment"]

        try:
            envname = experiment["env_name"]
        except KeyError:
            envname = experiment["env_id"]

        dummy_env = make_env(envname, 0, num_envs=1, init_state_noise_scale=0.0)
        dummy_env.reset()

        RESULT_DIR = os.path.join(run.config["result_dir"], f"seed{seed}")

        buf = construct_buffer(
            experiment,
            RESULT_DIR,
            dummy_env,
            ckpt
        )

        return buf

def construct_actor(
        experiment,
        path,
        key,
        hidden_dims,
        action_dim,
        observations,
        actor_lr,
        trainable = True
):
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

    if trainable:
        actor = Model.create(
            actor_def,
            inputs=[key, observations],
            tx=optax.adam(learning_rate=actor_lr),
        )
    else:
        actor = Model.create(
            actor_def,
            inputs=[key, observations],
        )
    return actor.load(path, no_opt=is_ppo)

def construct_critic(
    experiment,
    path,
    key,
    hidden_dims,
    observations,
    actions,
    critic_lr,
    num_critics,
    trainable = True,
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

    if trainable:
        critic = Model.create(
            critic_def,
            inputs=[key, observations, actions],
            tx=optax.adam(learning_rate=critic_lr),
        )
    else:
        critic = Model.create(
            critic_def,
            inputs=[key, observations, actions],
        )
    return critic.load(path)

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
