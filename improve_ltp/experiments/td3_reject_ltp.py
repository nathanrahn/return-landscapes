import glob
import h5py
import jax
import jax.numpy as jnp
import logging
import math
import os
import shutil
import wandb

from functools import partial
from pathlib import Path
from typing import Tuple

from brax import envs
from jaxrl.agents.ddpg.simple_ddpg_learner import SimpleDDPGLearner
from jaxrl.agents.ddpg.actor import update as update_ddpg_actor
from jaxrl.agents.sac.critic import target_update
from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, PRNGKey
from jaxrl.utils import make_env
from util.stats_util import zero_normed_metric
from util.load_models import get_run_by_run_name, load_actor_td3_simple, load_critic_td3_simple, load_buffer
from util.mc_eval_brax import SingleEpisodeEvalWrapper
from util.mc_eval_brax import get_evaluate_policy_fn

log = logging.getLogger(__name__)

class TD3RejectLTP:
    def __init__(
            self,
            root_policy_conf,
            eval_episodes,
            episode_length,
            seed,
            lr,
            max_steps,
            check_interval,
            batch_size,
            heuristic,
            cvar_alpha,
    ):
        self.root_policy_conf = root_policy_conf
        self.eval_episodes = eval_episodes
        self.episode_length = episode_length
        self.seed = seed
        self.lr = lr
        self.rng = jax.random.PRNGKey(seed)
        self.max_steps = max_steps
        self.check_interval = check_interval
        self.batch_size = batch_size
        self.heuristic = heuristic
        self.cvar_alpha = cvar_alpha

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

    def get_heuristic_id(self):
        if self.heuristic == "ltp":
            return ltp
        if self.heuristic == "cvar":
            return f"{self.cvar_alpha}cvar"
        if self.heuristic == "none":
            return "noreject"
        raise ValueError(f"Unrecognized rejection heuristic: {self.heuristic}")

    def get_custom_env(self, run, **kwargs):
        logger = logging.getLogger("init")
        experiment = run.config["experiment"]
        try:
            envname = experiment["env_name"]
        except KeyError:
            envname = experiment["env_id"]

        # Get the raw brax env
        env = envs._envs[envname](**kwargs)
        # Custom wrapper to keep track of single returns
        env = SingleEpisodeEvalWrapper(env)
        env.reset = jax.jit(env.reset)
        env.step = jax.jit(env.step)
        logger.info(f"Loaded environment: {envname}")
        return env

    def run(self, output_dir, this_job_result_dir):
        logger = logging.getLogger("root")
        self.filter_log()

        run_name = self.root_policy_conf.run_name
        ckpt = self.root_policy_conf.ckpt
        actor_seed = self.root_policy_conf.seed

        run = get_run_by_run_name(run_name)
        actor, actor_dir = load_actor_td3_simple(run, ckpt, actor_seed)
        actor, target_actor, actor_dir = load_actor_td3_simple(run, ckpt, actor_seed, include_target=True)
        #target_actor, _ = load_actor_td3_simple(run, ckpt, actor_seed)
        actor_batch_size = run.config["experiment"]["actor_batch_size"]
        critic_batch_size = run.config["experiment"]["critic_batch_size"]
        logger.info("Loaded actor")

        critic, target_critic, _ = load_critic_td3_simple(run, ckpt, actor_seed, include_target=True)
        critic_batch_size = run.config["experiment"]["critic_batch_size"]
        logger.info("Loaded critic and target critic")
        replay_buffer = load_buffer(run, ckpt, actor_seed)
        logger.info("Loaded replay buffer")

        env = make_env(
            run.config["experiment"]["env_name"],
            self.seed,
            num_envs=1,
            init_state_noise_scale=0.1
        )

        purd_env = self.get_custom_env(
            run,
            init_state_noise_scale=0.0
        )

        agent = SimpleDDPGLearner(
                self.seed,
                actor,
                critic,
                target_actor,
                target_critic,
                **dict(run.config['experiment']['ddpg_config'])
        )
        logger.info("Initialized TD3 agent")

        dtype = jnp.zeros(self.max_steps).dtype

        def td3_policy_perturbation(batch, actor, critic):
            new_actor, actor_info = update_ddpg_actor(actor, critic, batch)
            return new_actor

        td3_policy_perturbation = jax.jit(jax.vmap(td3_policy_perturbation, in_axes=(0, None, None)))

        def td3_optimization_purd(
                rng,
                env,
                actor,
                critic,
                replay_buffer,
        ):
            dtype = jnp.zeros(1).dtype
            undis_returns = jnp.zeros(self.eval_episodes, dtype=dtype)

            vmapped_reset = jax.vmap(env.reset, in_axes=(0,))

            eval_policy = get_evaluate_policy_fn(vmap_actor=True)

            for batch in range(math.ceil(self.eval_episodes / self.batch_size)):
                start = batch * self.batch_size
                end = min((batch + 1) * self.batch_size, self.eval_episodes)

                rng, *keys = jax.random.split(rng, end - start + 1)
                keys = jnp.array(keys)

                batches = replay_buffer.sample_parallel_multibatch(actor_batch_size, end - start)
                batches = jax.tree_map(lambda x: x.squeeze(), batches)

                actors = td3_policy_perturbation(batches, actor, critic)
                states = vmapped_reset(keys)
                undis_rets, _ = eval_policy(states, actors, env.step, keys)

                undis_returns = undis_returns.at[start:end].set(undis_rets)
                del actors
            return undis_returns

        self.rng, key = jax.random.split(self.rng)

        logger.info("Evaluating root policy...")
        undis_rets = td3_optimization_purd(
                key,
                purd_env,
                actor,
                critic,
                replay_buffer,
        )
        logger.info("Done")

        last_ltp = ltp(undis_rets)
        last_mean = undis_rets.mean()
        last_cvar = purd_cvar(undis_rets, self.cvar_alpha)
        mean_trace = [last_mean]
        ltp_trace = [last_ltp]
        cvar_trace = [last_cvar]

        last_agent_params = agent.agent_params()
        last_replay_size = replay_buffer.size
        last_replay_index = replay_buffer.insert_index
        if wandb.run is not None:
            wandb.log({"PURD mean": last_mean, "PURD LTP": last_ltp, "PURD CVaR": last_cvar}, step=0)

        successes = 0

        observations, dones = env.reset(), False
        #observation, done = env.reset(self.rng), False
        for step in range(self.max_steps):
            self.rng, key = jax.random.split(self.rng)

            actions = agent.sample_actions(observations, done=dones)
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
            batches = replay_buffer.sample_parallel_multibatch(
                max(actor_batch_size, critic_batch_size), 1
            )
            infos = agent.update(
                batches,
                actor_batch_size,
                critic_batch_size,
                num_updates=1,
            )

            if ((step + 1) % self.check_interval) == 0:
                self.rng, key = jax.random.split(self.rng)
                undis_rets = td3_optimization_purd(
                        key,
                        purd_env,
                        agent.actor,
                        agent.critic,
                        replay_buffer,
                )
                cur_ltp = ltp(undis_rets)
                cur_mean = undis_rets.mean()
                cur_cvar = purd_cvar(undis_rets, self.cvar_alpha)
                if self.heuristic == "ltp":
                    reject = cur_ltp > last_ltp
                elif self.heuristic == "cvar":
                    reject = cur_cvar < last_cvar
                elif self.heuristic == "none":
                    reject = False
                else:
                    raise ValueError("Unrecognized rejection heuristic: {self.heuristic}")
                if reject:
                    logger.info("Fail")
                    agent.set_agent_params(last_agent_params)
                    replay_buffer.size = last_replay_size
                    replay_buffer.insert_index = last_replay_index
                else:
                    successes += 1
                    logger.info(f"Success: Mean {cur_mean:>.1f}, LTP {cur_ltp:>.4f}")
                    last_agent_params = agent.agent_params()
                    last_mean = cur_mean
                    last_ltp = cur_ltp
                    last_cvar = cur_cvar
                    mean_trace.append(cur_mean)
                    ltp_trace.append(cur_ltp)
                    cvar_trace.append(cur_cvar)
                    last_replay_size = replay_buffer.size
                    last_replay_index = replay_buffer.insert_index
            if wandb.run is not None:
                wandb.log({"PURD mean": last_mean, "PURD LTP": last_ltp, "PURD CVaR": last_cvar}, step=step)

        return

        logger.info(f"Successful steps: {successes}")

        outpath = Path(actor_dir) / f"td3_reject_ltp_{ckpt}.h5"
        logger.info(f"Writing results to {outpath}")
        breakpoint()
        with h5py.File(outpath, "a") as f:
            heuristic_id = self.get_heuristic_id()
            base_dset_name = f"{heuristic_id}/{self.check_interval}check/{self.max_steps}steps"
            dset_name = f"{base_dset_name}/means"
            logger.info(f"Dataset name: {dset_name}")
            if dset_name in f:
                del f[dset_name]
            f.require_dataset(dset_name, (self.max_steps + 1,), dtype=dtype)
            f[dset_name][:] = mean_trace

            dset_name = f"{base_dset_name}/ltps"
            logger.info(f"Dataset name: {dset_name}")
            if dset_name in f:
                del f[dset_name]
            f.require_dataset(dset_name, (self.max_steps + 1,), dtype=dtype)
            f[dset_name][:] = ltp_trace

def unstructured_purd(
        rng,
        env,
        actor,
        noise_scale,
        eval_episodes,
        episode_length,
        batch_size
):
    dtype = jnp.zeros(1).dtype
    undis_returns = jnp.zeros(eval_episodes, dtype=dtype)

    vmapped_reset = jax.vmap(env.reset, in_axes=(0,))

    eval_policy = get_evaluate_policy_fn(vmap_actor=True)

    for batch in range(math.ceil(eval_episodes / batch_size)):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, eval_episodes)

        rng, *keys = jax.random.split(rng, end - start + 1)
        keys = jnp.array(keys)

        actors = unstructured_policy_perturbation(keys, actor, noise_scale)
        states = vmapped_reset(keys)
        undis_rets, _ = eval_policy(states, actors, env.step, keys)

        undis_returns = undis_returns.at[start:end].set(undis_rets)
        del actors
#        undis_returns[start:end] = undis_rets
#        returns[start:end] = rets
    return undis_returns

@partial(jax.vmap, in_axes=(0, None, None))
def unstructured_policy_perturbation(rng, actor, noise_scale):
    leaves, treedef = jax.tree_flatten(actor.params)
    rngs = jax.random.split(rng, len(leaves))
    noisy_leaves = [
            leaf + noise_scale * jax.random.normal(key, leaf.shape)
            for (leaf, key) in zip(leaves, rngs)
    ]
    noisy_params = jax.tree_unflatten(treedef, noisy_leaves)
    return actor.replace(params=noisy_params)

def purd_cvar(samples, alpha):
    quantile = jnp.quantile(samples, alpha)
    return quantile + jnp.minimum(samples - quantile, 0).mean() / alpha

def ltp(returns, threshold=0.5, nbins=100, return_mode = False):
    mode = purd_mode(returns, threshold=threshold, nbins=nbins)
    _ltp = zero_normed_metric(returns, mode, threshold)
    if return_mode:
        return _ltp, mode
    return _ltp

def purd_mode(returns, threshold=0.5, nbins=100):
    hist, bin_edges = jnp.histogram(returns, bins=nbins)
    big_bin = jnp.argmax(hist)
    return (bin_edges[big_bin] + bin_edges[big_bin + 1]) // 2
