import os
import random
import wandb

from environment import get_output_dir

import numpy as np
import tqdm

from jaxrl.agents import BCLearner
from jaxrl.datasets import ParallelReplayBuffer
from jaxrl.evaluation import evaluate_bc
from jaxrl.utils import make_env

import logging

import shutil

log = logging.getLogger(__name__)

from wandb.apis.public import Api

import flax
from jaxrl.networks.common import SaveState
from jaxrl.networks import policies
from jaxrl.networks.common import Model
import optax
import jax

from functools import partial


def prefix_dict_keys(d, prefix):
    return {prefix + str(k): v for k, v in d.items()}


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
    replay_buffer.load(f"{RESULT_DIR}/buffer")
    replay_buffer.size = ckpt_index
    return replay_buffer

def construct_actor(experiment, path, key, hidden_dims, action_dim, observations, actor_lr):
    if "algo" in experiment:
        if experiment["algo"] == "ddpg":
            actor_def = policies.MSEPolicy(hidden_dims, action_dim)
        elif experiment["algo"] == "sac":
            actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim)
        else:
            raise ValueError(f"Unknown algo: {experiment['algo']}")
    else:
        actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim)
    actor = Model.create(
        actor_def,
        inputs=[key, observations],
        tx=optax.adam(learning_rate=actor_lr),
    )
    return actor.load(path)

class TrainBC:
    def __init__(
        self,
        run_name, # Previous run name to point to
        seed, # Keep, Should be specified to point at a particular run
        eval_episodes, # Keep for eval
        eval_episode_length, # Keep for eval
        log_interval, # Keep for eval
        eval_interval, # Keep for eval
        max_steps, # Keep, number of gradient steps on BC
        tqdm, # Keep
        checkpoint, # Keep
        agent_log_starts, # Keep
        agent_log_step, # Keep
        agent_num_logs, # Keep
        replay_buffer_size, # Not sure, see about instantiating previous replay buffer
        batch_size,
        ckpt_index
    ):
        self.save_dir = get_output_dir()
        self.seed = seed
        self.eval_episodes = eval_episodes
        self.eval_episode_length = eval_episode_length
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.max_steps = int(max_steps)
        self.tqdm = tqdm
        self.checkpoint = checkpoint
        self.agent_log_starts = agent_log_starts
        self.agent_log_step = agent_log_step
        self.agent_num_logs = agent_num_logs
        self.agent_logging_points = self.compute_logging_points()
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.run_name = run_name
        self.ckpt_index = ckpt_index

    def compute_logging_points(self):
        logging_points = []
        for agent_log_start in self.agent_log_starts:
            for pt in range(
                agent_log_start,
                agent_log_start + self.agent_log_step * self.agent_num_logs,
                self.agent_log_step,
            ):
                logging_points.append(pt)
        return set(logging_points)

    def log_multiple_seeds_to_wandb(self, step, infos):
        if step % self.log_interval == 0:
            dict_to_log = {}
            for info_key in infos:
                value = infos[info_key]
                dict_to_log[info_key] = value
            wandb.log(dict_to_log, step=step)

    def copy_directory(self, src, dest):
        src, dest = str(src), str(dest)
        if src != dest:
            shutil.copytree(src, dest, dirs_exist_ok=True)

    def run(self, output_dir, result_dir):
        os.makedirs(self.save_dir, exist_ok=True)
        print("running in ", self.save_dir)

        api = Api()
        runs = api.runs(path="nonseq-exp/nonseq-exp", filters={"display_name": self.run_name})
        run = runs[0]

        experiment = run.config["experiment"]
        env_name = experiment["env_name"]
        RESULT_DIR = os.path.join(run.config["result_dir"], f"seed{self.seed}")

        env = make_env(env_name, self.seed, num_envs=1)
        eval_env = make_env(env_name, self.seed + 42, eval_episodes=self.eval_episodes)

        np.random.seed(self.seed)
        random.seed(self.seed)

        # very important
        env.reset()

        #######  Load the actor
        rng = jax.random.PRNGKey(0)
        rng, key = jax.random.split(rng)
        actor_path = os.path.join(
            RESULT_DIR,
            f"actor_ckpt_{self.ckpt_index}",
        )
        # Very important that this be a tuple because otherwise the actor apply_fn can't be hashed
        # and therefore doesn't work with jit
        hidden_dims = tuple(experiment[f"{experiment['algo']}_config"]["hidden_dims"])
        actor_lr = experiment[f"{experiment['algo']}_config"]["actor_lr"]
        actions = env.action_space.sample()[np.newaxis]
        action_dim = actions.shape[-1]
        observations = env.observation_space.sample()[np.newaxis]
        actor = construct_actor(
            experiment,
            actor_path,
            key,
            hidden_dims,
            action_dim,
            observations,
            actor_lr,
        )

        agent = BCLearner(
            seed=self.seed,
            observations=env.observation_space.sample()[np.newaxis],
            actions=env.action_space.sample()[np.newaxis],
            actor_lr=1e-3,
            num_steps=self.max_steps,
        )
        replay_buffer = construct_buffer(experiment, RESULT_DIR, env, self.ckpt_index)


        @jax.jit
        def get_actions(observations):
            return actor.apply({'params': actor.params}, observations, 0.0)

        for i in tqdm.tqdm(range(1, self.max_steps + 1), smoothing=0.1, disable=not self.tqdm):

            # TODO: make the batch size, it should be 256
            batch = replay_buffer.sample_parallel_multibatch(
                self.batch_size, 1 #updates_per_step
            )
            # relabel the actions in batch by applying the actor to the observations
            # make sure to just change the actions element of the batch

            new_act = get_actions(batch.observations)
            batch = batch._replace(actions=new_act)

            infos = agent.update(
                batch
            )
            self.log_multiple_seeds_to_wandb(i, infos)

            if i % self.eval_interval == 0:
                eval_stats = evaluate_bc(agent, eval_env, self.eval_episodes, episode_length=1000)
                self.log_multiple_seeds_to_wandb(i, eval_stats)

            if self.checkpoint and i in self.agent_logging_points:
                save_ckpt(agent.actor, self.save_dir, f"bc_ckpt_{i}.ckpt")



def save_ckpt(model, save_dir, fname) -> None:
    with open(os.path.join(save_dir, fname), "wb") as f:
        save_state = SaveState(params=model.params, opt_state=model.opt_state)
        data = flax.serialization.to_bytes(save_state)
        f.write(data)

def load_ckpt(save_dir, fname):
    # TODO: write this
    pass