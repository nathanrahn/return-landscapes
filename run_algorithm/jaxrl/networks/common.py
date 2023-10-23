import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import zipfile
import io
from pathlib import Path


def tree_norm(tree):
    return jnp.sqrt(sum((x ** 2).sum() for x in jax.tree_leaves(tree)))


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        return x


@flax.struct.dataclass
class SaveState:
    params: Params
    opt_state: Optional[optax.OptState] = None


# TODO: Replace with TrainState when it's ready
# https://github.com/google/flax/blob/master/docs/flip/1009-optimizer-api.md#train-state
@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        inputs: Sequence[jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None,
    ) -> "Model":
        variables = model_def.init(*inputs)

        _, params = variables.pop("params")

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1, apply_fn=model_def, params=params, tx=tx, opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({"params": self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, "Model"]:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)
        grad_norm = tree_norm(grads)
        info["grad_norm"] = grad_norm

        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state), info

    def save(self, save_dir, fname) -> None:
        for i in range(jax.tree_util.tree_flatten(self.params)[0][0].shape[0]):
            lil_params = jax.tree_map(lambda x: x[i], self.params)
            lil_opt_state = jax.tree_map(lambda x: x[i], self.opt_state)
            with zipfile.ZipFile(os.path.join(save_dir, f"seed{i}.zip"), "a") as zipf:
                save_state = SaveState(params=lil_params, opt_state=lil_opt_state)
                data = flax.serialization.to_bytes(save_state)
                with zipf.open(fname, "w", force_zip64=True) as f:
                    f.write(data)

    def load(self, load_path: str) -> "Model":
        load_path = Path(load_path)
        # Load path is expected to point to a file like: .../seed0/actor_ckpt_550000
        zip_path = load_path.parents[0].with_suffix(".zip")
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zipf:
                with zipf.open(load_path.name, "r") as f:
                    contents = f.read()
                    saved_state = flax.serialization.from_bytes(
                        SaveState(params=self.params, opt_state=self.opt_state), contents
                    )
                    return self.replace(params=saved_state.params, opt_state=saved_state.opt_state)
        else:
            with open(load_path, "rb") as f:
                contents = f.read()
                try:
                    # Old method, just loads params
                    params = flax.serialization.from_bytes(self.params, contents)
                    return self.replace(params=params)
                except:
                    # New method, also loads optim state
                    saved_state = flax.serialization.from_bytes(
                        SaveState(params=self.params, opt_state=self.opt_state), contents
                    )
                    return self.replace(params=saved_state.params, opt_state=saved_state.opt_state)


def split_tree(tree, key):
    tree_head = tree.unfreeze()
    tree_enc = tree_head.pop(key)
    tree_head = flax.core.FrozenDict(tree_head)
    tree_enc = flax.core.FrozenDict(tree_enc)
    return tree_enc, tree_head


# to separate opt_state for encoder and other layers
@flax.struct.dataclass
class ModelDecoupleOpt:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    tx_enc: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state_enc: Optional[optax.OptState] = None
    opt_state_head: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        inputs: Sequence[jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None,
        tx_enc: Optional[optax.GradientTransformation] = None,
    ) -> "ModelDecoupleOpt":
        variables = model_def.init(*inputs)

        _, params = variables.pop("params")

        if tx is not None:
            if tx_enc is None:
                tx_enc = tx
            params_enc, params_head = split_tree(params, "SharedEncoder")
            opt_state_enc = tx_enc.init(params_enc)
            opt_state_head = tx.init(params_head)
        else:
            opt_state_enc, opt_state_head = None, None

        return cls(
            step=1,
            apply_fn=model_def,
            params=params,
            tx=tx,
            tx_enc=tx_enc,
            opt_state_enc=opt_state_enc,
            opt_state_head=opt_state_head,
        )

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({"params": self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, "Model"]:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)
        grad_norm = tree_norm(grads)
        info["grad_norm"] = grad_norm

        params_enc, params_head = split_tree(self.params, "SharedEncoder")
        grads_enc, grads_head = split_tree(grads, "SharedEncoder")

        updates_enc, new_opt_state_enc = self.tx_enc.update(
            grads_enc, self.opt_state_enc, params_enc
        )
        new_params_enc = optax.apply_updates(params_enc, updates_enc)

        updates_head, new_opt_state_head = self.tx.update(
            grads_head, self.opt_state_head, params_head
        )
        new_params_head = optax.apply_updates(params_head, updates_head)

        new_params = flax.core.FrozenDict({**new_params_head, "SharedEncoder": new_params_enc})
        return (
            self.replace(
                step=self.step + 1,
                params=new_params,
                opt_state_enc=new_opt_state_enc,
                opt_state_head=new_opt_state_head,
            ),
            info,
        )

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> "Model":
        with open(load_path, "rb") as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)
