import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
from jaxrl.datasets.dataset import Batch
import collections
from brax.envs.env import State
from jax.random import KeyArray
import gym
from typing import Union, Tuple
import flax

from util.dict_util import flatten, unflatten


BufferData = collections.namedtuple("BufferData", ["observations", "actions", "next_states"])


class JaxSimulatorReplayBuffer:
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: Union[gym.spaces.Discrete, gym.spaces.Box],
        dummy_state: State,
        capacity: int,
    ):
        empty_states = jax.tree_map(
            lambda x: jnp.expand_dims(jnp.zeros_like(x), axis=0).repeat(capacity, axis=0),
            dummy_state,
        )
        self.buffer_data = BufferData(
            observations=jnp.zeros(
                (capacity, *observation_space.shape), dtype=observation_space.dtype
            ),
            actions=jnp.zeros((capacity, *action_space.shape), dtype=action_space.dtype),
            next_states=empty_states,
        )

        self.size = 0
        self.rng = jax.random.PRNGKey(np.random.randint(0, 2 ** 32))

        self.insert_index = 0
        self.capacity = capacity

        # for saving the buffer
        self.n_parts = 4
        assert self.capacity % self.n_parts == 0

    @partial(jax.jit, static_argnums=(0,))
    def _insert(
        self, full_buffer: BufferData, new_items: BufferData, insert_index: int
    ) -> BufferData:
        new_buffer = jax.tree_map(
            lambda x, y: jax.lax.dynamic_update_index_in_dim(x, y, index=insert_index, axis=0),
            full_buffer,
            new_items,
        )
        return new_buffer

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        next_observation: np.ndarray,
        next_state: State,
    ):
        new_items = BufferData(observations=observation, actions=action, next_states=next_state)
        self.buffer_data = self._insert(self.buffer_data, new_items, self.insert_index)
        this_insert_index = self.insert_index
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return this_insert_index

    @partial(jax.jit, static_argnums=(0, 2))
    def _sample(self, full_buffer: BufferData, batch_size: int, key: KeyArray) -> Batch:
        indx = jax.random.randint(key=key, shape=(batch_size,), minval=0, maxval=self.size)
        selected_buffer = jax.tree_map(lambda x: jnp.take(x, indx, axis=0), full_buffer)
        return Batch(
            observations=selected_buffer.observations,
            actions=selected_buffer.actions,
            rewards=selected_buffer.next_states.reward,
            masks=1 - selected_buffer.next_states.done,
            next_observations=selected_buffer.next_states.obs,
        )

    def sample(self, batch_size: int) -> Batch:
        self.rng, key = jax.random.split(self.rng, 2)
        return self._sample(self.buffer_data, batch_size, key)

    def get_simulator_history(self) -> Tuple[State, jnp.ndarray, jnp.ndarray, int]:
        simulator_history = self.buffer_data.next_states
        return simulator_history, simulator_history.obs, simulator_history.done, self.size

    def save(self, data_path: str):
        # TODO implement this
        pass

    def load(self, data_path: str):
        raise NotImplementedError


class NumpySimulatorReplayBuffer:
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: Union[gym.spaces.Discrete, gym.spaces.Box],
        dummy_state: State,
        capacity: int,
    ):
        self.empty_states = jax.tree_map(
            lambda x: np.expand_dims(np.zeros_like(x), axis=0).repeat(capacity, axis=0), dummy_state
        )
        self.observations = np.zeros(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )
        self.actions = np.zeros((capacity, *action_space.shape), dtype=action_space.dtype)

        # Whether the observation was received as a result of resetting the simulator, due to
        # environment termination, timeout, or due to an NSE reset.
        self.is_reset = np.zeros((capacity,), dtype=bool)

        # Whether the previous state was terminal.
        self.previous_state_terminal = np.zeros((capacity,), dtype=bool)

        # Whether the reset was selected from the initial state distribution or the replay buffer.
        # If true, from initial state distribution.
        self.init_state_or_buffer = np.zeros((capacity,), dtype=bool)

        self.next_states = self.empty_states

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

        # for saving the buffer
        self.n_parts = 4
        assert self.capacity % self.n_parts == 0

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        mask: float,
        next_observation: np.ndarray,
        next_state: State,
        is_reset: bool = False,
        previous_state_terminal: bool = False,
        init_state_or_buffer: bool = False,
    ) -> int:
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action

        self.is_reset[self.insert_index] = is_reset
        self.previous_state_terminal[self.insert_index] = previous_state_terminal
        self.init_state_or_buffer[self.insert_index] = init_state_or_buffer

        jax.tree_map(lambda x, y: x.__setitem__(self.insert_index, y), self.next_states, next_state)

        this_insert_index = self.insert_index
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return this_insert_index

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(
            observations=self.observations[indx],
            actions=self.actions[indx],
            rewards=self.next_states.reward[indx],
            # TODO: is this correct?
            masks=1 - self.next_states.done[indx],
            next_observations=self.next_states.obs[indx],
        )

    def get_simulator_history(self) -> Tuple[State, jnp.ndarray, jnp.ndarray, int]:
        simulator_history = self.next_states
        return simulator_history, simulator_history.obs, simulator_history.done, self.size

    def sample_simulator_history(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indx = np.random.randint(self.size, size=batch_size)
        smaller_simulation_history = jax.tree_map(lambda x: x[indx], self.next_states)
        return (
            smaller_simulation_history,
            smaller_simulation_history.obs,
            smaller_simulation_history.done,
            indx,
        )

    def save(self, data_path: str):
        chunk_size = self.capacity // self.n_parts
        for i in range(self.n_parts):
            observations = self.observations[i * chunk_size : (i + 1) * chunk_size]
            actions = self.actions[i * chunk_size : (i + 1) * chunk_size]

            is_reset = self.is_reset[i * chunk_size : (i + 1) * chunk_size]
            previous_state_terminal = self.previous_state_terminal[
                i * chunk_size : (i + 1) * chunk_size
            ]
            init_state_or_buffer = self.init_state_or_buffer[i * chunk_size : (i + 1) * chunk_size]

            partial_state = jax.tree_map(
                lambda x: x[i * chunk_size : (i + 1) * chunk_size], self.next_states
            )
            partial_state_dict = flax.serialization.to_state_dict(partial_state)
            partial_state_dict_flat = flatten(partial_state_dict)

            np.savez_compressed(
                data_path + f"_chunk_{i}.npz",
                observations=observations,
                actions=actions,
                is_reset=is_reset,
                previous_state_terminal=previous_state_terminal,
                init_state_or_buffer=init_state_or_buffer,
                size=self.size,
                **partial_state_dict_flat,
            )

    def load(self, data_path: str):
        # Load the replay buffer from the n_parts separate chunks
        chunk_size = self.capacity // self.n_parts
        for i in range(self.n_parts):
            npz_file = np.load(data_path + f"_chunk_{i}.npz")
            self.observations[i * chunk_size : (i + 1) * chunk_size] = npz_file["observations"]
            self.actions[i * chunk_size : (i + 1) * chunk_size] = npz_file["actions"]

            self.is_reset[i * chunk_size : (i + 1) * chunk_size] = npz_file["is_reset"]
            self.previous_state_terminal[i * chunk_size : (i + 1) * chunk_size] = npz_file[
                "previous_state_terminal"
            ]
            self.init_state_or_buffer[i * chunk_size : (i + 1) * chunk_size] = npz_file[
                "init_state_or_buffer"
            ]

            self.size = int(npz_file["size"])

            state_keys = [
                k
                for k in npz_file.keys()
                if k
                not in [
                    "observations",
                    "actions",
                    "size",
                    "is_reset",
                    "previous_state_terminal",
                    "init_state_or_buffer",
                ]
            ]
            state_dict_flat = {k: npz_file[k] for k in state_keys}
            state_dict = unflatten(state_dict_flat)
            empty_state_partial = jax.tree_map(
                lambda x: x[i * chunk_size : (i + 1) * chunk_size], self.empty_states
            )
            partial_state = flax.serialization.from_state_dict(empty_state_partial, state_dict)

            jax.tree_map(
                lambda x, y: x.__setitem__(slice(i * chunk_size, (i + 1) * chunk_size), y),
                self.next_states,
                partial_state,
            )
            npz_file.close()


class DmcSimulatorReplayBuffer:
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: Union[gym.spaces.Discrete, gym.spaces.Box],
        dummy_state: np.ndarray,
        capacity: int,
    ):
        self.observations = np.zeros(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )
        self.actions = np.zeros((capacity, *action_space.shape), dtype=action_space.dtype)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, dummy_state.shape[-1]), dtype=np.float32)
        self.next_observations = np.zeros(
            (capacity, *observation_space.shape), dtype=observation_space.dtype
        )

        self.size = 0
        self.insert_index = 0
        self.capacity = capacity

        # for saving the buffer
        self.n_parts = 4
        assert self.capacity % self.n_parts == 0

    def insert(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: float,
        next_observation: np.ndarray,
        next_state: np.ndarray,
    ):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.dones[self.insert_index] = done
        self.next_observations[self.insert_index] = next_observation
        self.next_states[self.insert_index] = next_state

        this_insert_index = self.insert_index
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return this_insert_index

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(
            observations=self.observations[indx],
            actions=self.actions[indx],
            rewards=self.rewards[indx],
            masks=1 - self.dones[indx],
            next_observations=self.next_observations[indx],
        )

    def get_simulator_history(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        return self.next_states, self.next_observations, self.dones, self.size

    def sample_simulator_history(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indx = np.random.randint(self.size, size=batch_size)
        return self.next_states[indx], self.next_observations[indx], self.dones[indx], indx

    def save(self, data_path: str):
        # TODO implement this
        pass

    def load(self, data_path: str):
        raise NotImplementedError
