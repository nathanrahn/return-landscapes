from brax.envs.wrappers import EpisodeWrapper, AutoResetWrapper, VmapWrapper
from brax.envs import env as brax_env
import jax


def wrap_for_training(env: brax_env.Env,
                      episode_length: int = 1000,
                      action_repeat: int = 1) -> brax_env.Wrapper:
    """Common wrapper pattern for all training agents.
    Args:
        env: environment to be wrapped
        episode_length: length of episode
        action_repeat: how many repeated actions to take per step
    Returns:
        An environment that is wrapped with Episode and AutoReset wrappers.  If the
        environment did not already have batch dimensions, it is additional Vmap
        wrapped.
    """
    env = EpisodeWrapper(env, episode_length, action_repeat)
    batched = False
    if hasattr(env, 'custom_tree_in_axes'):
        batch_indices, _ = jax.tree_util.tree_flatten(env.custom_tree_in_axes)
        if 0 in batch_indices:
            batched = True
    if not batched:
        env = VmapWrapper(env)
    env = AutoResetWrapper(env)
    return env
