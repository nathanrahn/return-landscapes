from wandb.apis.public import Api

from experiments.policy_variance import get_evaluate_policy_fn, get_custom_env

from util.evaluation_util import get_actor_variables
from pathlib import Path
import os
from tqdm import tqdm
from jaxrl.networks.common import Model
from jaxrl.networks import policies
import numpy as np
import jax
import jax.numpy as jnp
import optax
from pathlib import Path
import h5py
import math
import copy
from util.evaluation_util import clear_caches, update_policy, construct_update_fn


class PolicyInterpolationSeed:
    def __init__(self, run_name, batch_size, num_points, num_purd_samples, noise_amount, same_seed, seed, first_half):
        self.run_name = run_name
        self.batch_size = batch_size
        self.num_points = num_points
        self.num_purd_samples = num_purd_samples
        self.noise_amount = noise_amount
        self.same_seed = same_seed
        self.seed = seed
        self.first_half = first_half

    def construct_actor(self, experiment, path, key, hidden_dims, action_dim, observations, actor_lr):
        is_ppo = False
        if "algo" in experiment:
            if experiment["algo"] == "ddpg":
                actor_def = policies.MSEPolicy(hidden_dims, action_dim)
            elif experiment["algo"] == "sac":
                actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim)
            else:
                raise ValueError(f"Unknown algo: {experiment['algo']}")
        elif "gae_lambda" in experiment:
            raise NotImplementedError("Actor loading for PPO")
        else:
            raise ValueError(f"unknown algorithm type")
        actor = Model.create(
            actor_def,
            inputs=[key, observations],
            tx=optax.adam(learning_rate=actor_lr),
        )
        return actor.load(path, no_opt=is_ppo)

    def run_one(self, run_name1, run_name2, seed1, ckpt1, seed2, ckpt2):
        alpha = jnp.linspace(0.0, 1.0, self.num_points, endpoint=True)
        actor1_variables, side_info = get_actor_variables(run_name1, seed1, ckpt1, side_info=True)
        experiment, RESULT_DIR, rng, env = side_info
        actor1 = self.construct_actor(*actor1_variables)
        actor2 = self.construct_actor(*get_actor_variables(run_name2, seed2, ckpt2))

        def convex_combination(alpha, actor1, actor2):
            # direction should be a pytree that looks like actor.params
            return actor1.replace(params=jax.tree_map(lambda x, y: alpha * x + (1 - alpha) * y,
                                                      actor1.params, actor2.params))

        # vmap over the first dimension of directions, but not over actor
        convex_combination = jax.vmap(convex_combination, in_axes=(0, None, None))
        actors = convex_combination(alpha, actor1, actor2)

        eval_policy = get_evaluate_policy_fn(
            vmap_actor=True,
            is_ppo="gae_lambda" in experiment,
            is_sac=experiment["algo"] == "sac",
        )

        actors = actors.replace(opt_state=None)
        if self.num_purd_samples is None:
            # evaluate the different actors directly
            undis_returns = np.zeros(self.num_points)
            for batch in tqdm(range(math.ceil(self.num_points / self.batch_size)), desc="batch"):
                if batch % 20 == 0:
                    # hacky way to get rid of the memory leak
                    clear_caches()
                start = batch * self.batch_size
                end = min((batch + 1) * self.batch_size, self.num_points)

                rng, *keys = jax.random.split(rng, end - start + 1)
                vmapped_reset = jax.vmap(env.reset, in_axes=(0,))
                states = vmapped_reset(jnp.array(keys))

                actors_batch = jax.tree_map(lambda x: x[start:end], actors)

                keys = jnp.array(keys)
                undis_rets, _ = eval_policy(
                    states,
                    actors_batch,
                    env.step,
                    keys
                )

                undis_returns[start:end] = undis_rets

            fpath = Path(RESULT_DIR) / "policy_interpolation.h5"

            with h5py.File(fpath, "a") as f:
                base = get_base_dset_string(run_name1, seed1, ckpt1,
                                            run_name2, seed2, ckpt2, self.num_points)

                path = os.path.join(base, "returns")
                f.require_dataset(path, undis_returns.shape, dtype=undis_returns.dtype)
                f[path][:] = undis_returns
        else:
            base_actors = actors
            update_fn, update_stuff = construct_update_fn("init_parameter", self.noise_amount, *([None]*10))
            # Compute (unstructured) post-update noisy distribution for each actor
            dtype = jnp.zeros(1).dtype
            undis_returns = np.zeros((self.num_purd_samples, self.num_points), dtype=dtype)
            for batch in tqdm(range(math.ceil(self.num_purd_samples / self.batch_size)), desc=str(self.noise_amount)):
                if batch % 20 == 0:
                    # hacky way to get rid of the memory leak
                    clear_caches()
                start = batch * self.batch_size
                end = min((batch + 1) * self.batch_size, self.num_points)

                rng, *keys = jax.random.split(rng, end - start + 1)
                vmapped_reset = jax.vmap(env.reset, in_axes=(0,))
                states = vmapped_reset(jnp.array(keys))

                actor = jax.vmap(update_policy, in_axes=(None, None, None, 0, None,
                                                         None, None, None, None, None))("init_parameter", None, rng,
                                                                                        base_actors, update_fn, update_stuff,
                                                                                        start, end, self.noise_amount, None)
                # Evaluate the policy
                rng, *keys = jax.random.split(rng, end - start + 1)
                keys = jnp.array(keys)

                undis_rets, rets = jax.vmap(eval_policy, in_axes=(None, 0, None, None))(
                    states,
                    actor,
                    env.step,
                    keys
                )

                # Save the returns
                undis_returns[start:end] = undis_rets.T
                os.umask(0)

            fpath = Path(RESULT_DIR) / "policy_interpolation_purd.h5"

            with h5py.File(fpath, "a") as f:
                base = get_base_dset_string(run_name1, seed1, ckpt1,
                                            run_name2, seed2, ckpt2, self.num_points)
                path = os.path.join(base, "returns/{}/{}".format(self.noise_amount, self.num_purd_samples))
                f.require_dataset(path, (self.num_purd_samples, self.num_points), dtype=undis_returns.dtype)
                f[path][:] = undis_returns

    def run(self, output_dir, this_job_result_dir):
        valid_checkpoints = [50000, 150000, 250000, 350000, 450000, 550000, 650000, 750000, 850000, 950000]
        ckpt_pairs = []
        for ckpt1 in valid_checkpoints:
            for ckpt2 in valid_checkpoints:
                # Don't include the same checkpoint twice
                if ckpt1 != ckpt2:
                    ckpt_pairs.append((ckpt1, ckpt2))
        ckpt_pairs = ckpt_pairs[:len(ckpt_pairs)//2] if self.first_half else ckpt_pairs[len(ckpt_pairs)//2:]
        if self.same_seed:
            # Select a particular seed
            seeds1 = [self.seed]*len(ckpt_pairs)
            seeds2 = [self.seed]*len(ckpt_pairs)
        else:
            seeds1 = [self.seed]*len(ckpt_pairs)
            # This only works if you don't exceed the number of seeds
            seeds2 = [self.seed+1]*len(ckpt_pairs)
        run_names = [self.run_name]*len(ckpt_pairs)
        ckpts1, ckpts2 = zip(*ckpt_pairs)
        for args in tqdm(zip(run_names, run_names, seeds1, ckpts1, seeds2, ckpts2), total=len(ckpt_pairs)):
            self.run_one(*args)


def save_pytree_to_h5(f, path, pytree):
    from silx.io.dictdump import dicttoh5
    dicttoh5(pytree, f, h5path=path, update_mode="modify")

def pytree_from_h5(f, path):
    from silx.io.dictdump import h5todict
    return h5todict(f, path)

def get_base_dset_string(run_name1, seed1, ckpt1, run_name2, seed2, ckpt2, num_points):
    return f"{run_name1}/{seed1}/{ckpt1}/{run_name2}/{run_name2}/{seed2}/{ckpt2}/{num_points}"
