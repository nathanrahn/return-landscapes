from typing import Tuple

import gym
import numpy as np

from jaxrl.datasets.dataset import Batch
import jax
from brax.envs.env import State
import jax.numpy as jnp

from functools import partial

import flax
from util.dict_util import flatten, unflatten
import os

import zipfile
import io
from pathlib import Path


class ParallelReplayBuffer:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        dummy_state: State,
        capacity: int,
        num_seeds: int,
    ):
        self.empty_states = jax.tree_map(
            lambda x: np.expand_dims(jnp.zeros_like(x), axis=0).repeat(capacity, axis=0),
            dummy_state,
        )
        self.observations = np.empty((num_seeds, capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty((num_seeds, capacity, action_dim), dtype=np.float32)
        self.masks = np.empty(
            (
                num_seeds,
                capacity,
            ),
            dtype=np.float32,
        )

        self.next_states = self.empty_states
        self.num_seeds = num_seeds

        self.size = 0
        self.insert_index = 0
        self.capacity = capacity

        self.n_parts = 4
        assert self.capacity % self.n_parts == 0

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        mask: float,
        next_state: State,
    ):
        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.masks[:, self.insert_index] = mask

        jax.tree_map(lambda x, y: x.__setitem__(self.insert_index, y), self.next_states, next_state)

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_parallel(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(
            observations=self.observations[:, indx],
            actions=self.actions[:, indx],
            rewards=self.next_states.reward[indx],
            masks=self.masks[:, indx],
            next_observations=self.next_states.obs[indx],
        )

    def sample_parallel_multibatch(self, batch_size: int, num_batches: int) -> Batch:
        indxs = np.random.randint(self.size, size=(num_batches, batch_size))
        return Batch(
            observations=self.observations[:, indxs],
            actions=self.actions[:, indxs],
            rewards=np.transpose(self.next_states.reward, (1, 0))[:, indxs],
            masks=self.masks[:, indxs],
            next_observations=np.transpose(self.next_states.obs, (1, 0, 2))[:, indxs],
        )

    def get_simulator_history(self) -> Tuple[State, jnp.ndarray, jnp.ndarray, int]:
        assert self.num_seeds == 1
        # Select the data corresponding to the first seed
        simulator_history = jax.tree_map(lambda x: x[:, 0, ...], self.next_states)
        return simulator_history, simulator_history.obs, simulator_history.done, self.size

    def insert_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        masks: np.ndarray,
        next_states: State,
    ) -> int:
        batch_len = observations.shape[1]
        if self.insert_index + batch_len < self.capacity:
            self.observations[:, self.insert_index:self.insert_index + batch_len] = observations
            self.actions[:, self.insert_index:self.insert_index + batch_len] = actions
            self.masks[:, self.insert_index:self.insert_index + batch_len] = masks

            jax.tree_map(
                lambda x, y: x.__setitem__(slice(self.insert_index, self.insert_index + batch_len), y),
                self.next_states,
                next_states,
            )

            this_insert_index = self.insert_index
            self.insert_index = (self.insert_index + batch_len) % self.capacity
            self.size = min(self.size + batch_len, self.capacity)
            return this_insert_index
        else:
            self.observations[:, self.insert_index:self.capacity] = observations[:,
                : self.capacity - self.insert_index
            ]
            self.actions[:, self.insert_index:self.capacity] = actions[:, :self.capacity - self.insert_index]
            self.masks[:, self.insert_index:self.capacity] = masks[:, :self.capacity - self.insert_index]

            jax.tree_map(
                lambda x, y: x.__setitem__(slice(self.insert_index, self.capacity), y),
                self.next_states,
                jax.tree_map(lambda x: x[:self.capacity - self.insert_index], next_states),
            )

            self.observations[:, 0:batch_len - (self.capacity - self.insert_index)] = observations[:,
                self.capacity - self.insert_index:
            ]
            self.actions[:, 0:batch_len - (self.capacity - self.insert_index)] = actions[:,
                self.capacity - self.insert_index:
            ]
            self.masks[:, 0:batch_len - (self.capacity - self.insert_index)] = masks[:,
                self.capacity - self.insert_index:
            ]

            jax.tree_map(
                lambda x, y: x.__setitem__(slice(0, batch_len - (self.capacity - self.insert_index)), y),
                self.next_states,
                jax.tree_map(lambda x: x[self.capacity - self.insert_index:], next_states)
            )

            this_insert_index = self.insert_index
            self.insert_index = (self.insert_index + batch_len) % self.capacity
            self.size = min(self.size + batch_len, self.capacity)
            return this_insert_index

    def to_save_dict(self, seed, chunk_size, i):
        observations = self.observations[seed, i * chunk_size : (i + 1) * chunk_size]
        actions = self.actions[seed, i * chunk_size : (i + 1) * chunk_size]

        partial_state = jax.tree_map(
            lambda x: x[i * chunk_size : (i + 1) * chunk_size, seed], self.next_states
        )
        partial_state_dict = flax.serialization.to_state_dict(partial_state)
        partial_state_dict_flat = flatten(partial_state_dict)

        return dict(observations=observations,
                    actions=actions,
                    size=self.size,
                    **partial_state_dict_flat)

    def save(self, save_dir, fname):
        chunk_size = self.capacity // self.n_parts
        # Save n_seeds replay buffers
        for seed in range(self.observations.shape[0]):
            with zipfile.ZipFile(os.path.join(save_dir, f"seed{seed}.zip"), "a") as zipf:
                for i in range(self.n_parts):
                    save_dict = self.to_save_dict(seed, chunk_size, i)
                    with zipf.open(f"{fname}_chunk_{i}.npz", "w", force_zip64=True) as f:
                        np.savez_compressed(
                            f,
                            **save_dict
                        )


    def process_part(self, npz_file, i):
        chunk_size = self.capacity // self.n_parts
        self.observations[:, i * chunk_size : (i + 1) * chunk_size] = npz_file["observations"][
            None, ...
        ]
        self.actions[:, i * chunk_size : (i + 1) * chunk_size] = npz_file["actions"][None, ...]

        self.size = int(npz_file["size"])

        state_keys = [
            k
            for k in npz_file.keys()
            if k
            not in [
                "observations",
                "actions",
                "size",
            ]
        ]
        state_dict_flat = {k: npz_file[k] for k in state_keys}
        state_dict = unflatten(state_dict_flat)
        empty_state_partial = jax.tree_map(
            lambda x: x[i * chunk_size : (i + 1) * chunk_size], self.empty_states
        )
        
        if "info" not in state_dict:
            state_dict["info"] = {}
            empty_state_partial = empty_state_partial.replace(info={})
            self.next_states = self.next_states.replace(info={})

        partial_state = flax.serialization.from_state_dict(empty_state_partial, state_dict)

        jax.tree_map(
            lambda x, y: x.__setitem__(
                slice(i * chunk_size, (i + 1) * chunk_size), y[:, None, ...]
            ),
            self.next_states,
            partial_state,
        )

    def load(self, data_path: str):
        data_path = Path(data_path)
        # Expecting a path like .../seed0/buffer
        zip_path = data_path.parents[0].with_suffix(".zip")
        # Load a single replay buffer from the n_parts separate chunks
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zipf:
                for i in range(self.n_parts):
                    with zipf.open(data_path.name + f"_chunk_{i}.npz", "r") as f:
                        self.process_part(np.load(f), i)
        else:
            for i in range(self.n_parts):
                with np.load(str(data_path) + f"_chunk_{i}.npz") as npz_file:
                    self.process_part(npz_file, i)
