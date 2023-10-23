import os
import random
import wandb

from environment import get_output_dir

import numpy as np
import tqdm

from jaxrl.agents import DDPGLearner
from jaxrl.agents.sac.sac_learner import SACLearner
from jaxrl.datasets import ParallelReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env

import logging

import shutil

log = logging.getLogger(__name__)


def prefix_dict_keys(d, prefix):
    return {prefix + str(k): v for k, v in d.items()}


class TrainJaxRL:
    def __init__(
        self,
        env_name,
        seed,
        num_seeds,
        eval_episodes,
        eval_episode_length,
        log_interval,
        eval_interval,
        critic_batch_size,
        actor_batch_size,
        max_steps,
        start_training,
        tqdm,
        reset_interval,
        checkpoint,
        sac_config,
        ddpg_config,
        agent_log_starts,
        agent_log_step,
        agent_num_logs,
        updates_per_step,
        replay_buffer_size,
        algo,
    ):
        self.env_name = env_name
        self.save_dir = get_output_dir()
        self.seed = seed
        self.num_seeds = num_seeds
        self.eval_episodes = eval_episodes
        self.eval_episode_length = eval_episode_length
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.actor_batch_size = actor_batch_size
        self.critic_batch_size = critic_batch_size
        self.max_steps = int(max_steps)
        self.start_training = int(start_training)
        self.tqdm = tqdm
        self.reset_interval = reset_interval
        self.checkpoint = checkpoint
        self.agent_log_starts = agent_log_starts
        self.agent_log_step = agent_log_step
        self.agent_num_logs = agent_num_logs
        self.agent_logging_points = self.compute_logging_points()
        self.updates_per_step = updates_per_step
        self.replay_buffer_size = replay_buffer_size
        self.algo = algo
        self.config = sac_config if algo == "sac" else ddpg_config

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
                for seed, value in enumerate(infos[info_key]):
                    dict_to_log[f"seed{seed}/{info_key}"] = value
            wandb.log(dict_to_log, step=step)

    def copy_directory(self, src, dest):
        src, dest = str(src), str(dest)
        if src != dest:
            shutil.copytree(src, dest, dirs_exist_ok=True)

    def make_seed_dirs(self):
        for seed in range(self.num_seeds):
            seed_dir = self.save_dir / f"seed{seed}"
            seed_dir.mkdir(exist_ok=True, parents=True)

    def run(self, output_dir, result_dir):
        os.makedirs(self.save_dir, exist_ok=True)

        self.make_seed_dirs()

        env = make_env(self.env_name, self.seed, num_envs=self.num_seeds)
        eval_env = make_env(self.env_name, self.seed + 42, eval_episodes=self.eval_episodes)

        np.random.seed(self.seed)
        random.seed(self.seed)

        kwargs = dict(self.config)
        if self.algo == "ddpg":
            Learner = DDPGLearner
        elif self.algo == "sac":
            Learner = SACLearner
        else:
            raise ValueError(f"Unknown algorithm: {self.algo}")

        agent = Learner(
            self.seed,
            env.observation_space.sample()[np.newaxis],
            env.action_space.sample()[np.newaxis],
            num_seeds=self.num_seeds,
            **kwargs,
        )
        observations, dones = env.reset(), False
        replay_buffer = ParallelReplayBuffer(
            observation_dim=env.observation_space.shape[-1],
            action_dim=env.action_space.shape[-1],
            dummy_state=env.get_state(),
            capacity=int(self.replay_buffer_size) or int(self.max_steps),
            num_seeds=self.num_seeds,
        )

        for i in tqdm.tqdm(range(1, self.max_steps + 1), smoothing=0.1, disable=not self.tqdm):
            if i <= self.start_training:
                actions = env.action_space.sample()
            else:
                actions = agent.sample_actions(observations)
            next_observations, rewards, dones, infos = env.step(actions)
            masks = env.generate_masks(dones, infos)

            replay_buffer.insert(
                observations,
                actions,
                masks,
                env.get_state(),
            )
            observations = next_observations

            observations, dones = env.reset_where_done(observations, dones)

            if i > self.start_training:
                batches = replay_buffer.sample_parallel_multibatch(
                    max(self.actor_batch_size, self.critic_batch_size), self.updates_per_step
                )
                infos = agent.update(
                    batches,
                    self.actor_batch_size,
                    self.critic_batch_size,
                    num_updates=self.updates_per_step,
                )
                self.log_multiple_seeds_to_wandb(i, infos)

            if i % self.eval_interval == 0:
                eval_stats = evaluate(agent, eval_env, self.eval_episodes, episode_length=1000)
                self.log_multiple_seeds_to_wandb(i, eval_stats)

            if self.checkpoint and i in self.agent_logging_points:
                agent.actor.save(self.save_dir, f"actor_ckpt_{i}")
                #agent.target_actor.save(self.save_dir, f"target_actor_ckpt_{i}")
                if self.algo == "sac":
                    agent.temp.save(self.save_dir, f"temp_ckpt_{i}")
                agent.critic.save(self.save_dir, f"critic_ckpt_{i}")
                agent.target_critic.save(self.save_dir, f"target_critic_ckpt_{i}")
        replay_buffer.save(self.save_dir, "buffer")
