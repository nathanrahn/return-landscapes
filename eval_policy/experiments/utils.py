"""
Utility functions used for classic control environments and pytorch.
"""
import copy
from typing import Optional, SupportsFloat, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


def verify_number_and_cast(x: SupportsFloat) -> float:
    """Verify parameter is a single number and cast to a float."""
    try:
        x = float(x)
    except (ValueError, TypeError):
        raise ValueError(f"An option ({x}) could not be converted to a float.")
    return x


def maybe_parse_reset_bounds(
    options: Optional[dict], default_low: float, default_high: float
) -> Tuple[float, float]:
    """
    This function can be called during a reset() to customize the sampling
    ranges for setting the initial state distributions.

    Args:
      options: Options passed in to reset().
      default_low: Default lower limit to use, if none specified in options.
      default_high: Default upper limit to use, if none specified in options.

    Returns:
      Tuple of the lower and upper limits.
    """
    if options is None:
        return default_low, default_high

    low = options.get("low") if "low" in options else default_low
    high = options.get("high") if "high" in options else default_high

    # We expect only numerical inputs.
    low = verify_number_and_cast(low)
    high = verify_number_and_cast(high)
    if low > high:
        raise ValueError(
            f"Lower bound ({low}) must be lower than higher bound ({high})."
        )

    return low, high


def copy_model(model, *args):
    # Create a copy of the model state dict
    model_copy_state_dict = copy.deepcopy(model.state_dict())

    # Create a new instance of the same model
    model_copy = type(model)(*args)

    # Load the copied state dict into the new model instance
    model_copy.load_state_dict(model_copy_state_dict)

    return model_copy


def copy_optimizer(optimizer, model_copy):
    # Create a new optimizer instance
    optimizer_copy = type(optimizer)(model_copy.parameters(), **optimizer.defaults)

    # Copy over the state
    for p_copy, p in zip(model_copy.parameters(), optimizer.param_groups[0]['params']):
        optimizer_copy.state[p_copy] = copy.deepcopy(optimizer.state[p])

    return optimizer_copy


def get_greedy_action(x, agent):
    logits, _ = agent(x)
    action = torch.argmax(logits, dim=2)
    return action


def evaluate_policy(vmapped_agent, envs, device, max_len=1000):
    # Reset the environments
    obs = envs.reset()
    episode_reward = np.zeros(envs.num_envs)
    done = envs.num_envs*[False]
    count = 0
    while not np.all(done):
        count += 1
        with torch.inference_mode():
            action = get_greedy_action(torch.Tensor(obs).unsqueeze(1).to(device), vmapped_agent)
        obs, reward, done, _ = envs.step(action.squeeze().cpu().detach().numpy())
        episode_reward = episode_reward + reward * (1 - done)
        if count > max_len:
            break
    return episode_reward


def single_update_agent(agent, optimizer, b_inds, b_obs, b_advantages, b_actions,
                        b_logprobs, b_values, b_returns, start, args):
    end = start + args.minibatch_size
    mb_inds = b_inds[start:end]

    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
    logratio = newlogprob - b_logprobs[mb_inds]
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()

    mb_advantages = b_advantages[mb_inds]
    if args.norm_adv:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
        v_clipped = b_values[mb_inds] + torch.clamp(
            newvalue - b_values[mb_inds],
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()
    return approx_kl, old_approx_kl, entropy_loss, pg_loss, v_loss


def single_update_vmapped_agent(agent, optimizer, b_inds, b_obs, b_advantages, b_actions,
                                b_logprobs, b_values, b_returns, start, args):
    end = start + args.minibatch_size
    mb_inds = b_inds[start:end]

    logits, newvalue = agent(b_obs[:, mb_inds])
    probs = Categorical(logits=logits)
    newlogprob = probs.log_prob(b_actions[:, mb_inds])
    entropy = probs.entropy()
    logratio = newlogprob - b_logprobs[:, mb_inds]
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()

    mb_advantages = b_advantages[:, mb_inds]
    if args.norm_adv:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean(axis=-1)

    # Value loss
    newvalue = newvalue.view(b_obs.shape[0], -1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - b_returns[:, mb_inds]) ** 2
        v_clipped = b_values[:, mb_inds] + torch.clamp(
            newvalue.squeeze() - b_values[:, mb_inds],
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - b_returns[:,mb_inds]) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - b_returns[:, mb_inds]) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    optimizer.zero_grad()
    loss.mean().backward()
    # nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()
    return approx_kl, old_approx_kl, entropy_loss, pg_loss, v_loss


def generate_data(agent, num_steps, obs, actions, logprobs, next_obs, rewards,
                  dones, done, next_done, values, envs, device, args):
    for step in range(0, num_steps):
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = envs.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
    # bootstrap value if not done
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values


def update_agent(agent, optimizer, b_obs, b_actions, b_logprobs, b_advantages,
                 b_values, b_returns, args):
    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            approx_kl, old_approx_kl, entropy_loss, pg_loss, v_loss = single_update_agent(
                agent, optimizer, b_inds, b_obs, b_advantages, b_actions,
                b_logprobs, b_values, b_returns, start, args
            )

        if args.target_kl is not None:
            if approx_kl > args.target_kl:
                break
    return approx_kl, old_approx_kl, entropy_loss, pg_loss, v_loss


def update_vmapped_agent(agent, optimizer, b_obs, b_actions, b_logprobs, b_advantages,
                         b_values, b_returns, args):
    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            approx_kl, old_approx_kl, entropy_loss, pg_loss, v_loss = single_update_vmapped_agent(
                agent, optimizer, b_inds, b_obs, b_advantages, b_actions,
                b_logprobs, b_values, b_returns, start, args
            )
    return approx_kl, old_approx_kl, entropy_loss, pg_loss, v_loss


def init_ensemble_optim(existing_optimizer, ensemble_optimizer, models):
    existing_optimizer_state = existing_optimizer.state_dict()

    # Create a new state dict for the ensemble optimizer
    ensemble_optimizer_state = ensemble_optimizer.state_dict()

    # Copy over global settings (like "param_groups")
    ensemble_optimizer_state['param_groups'] = existing_optimizer_state['param_groups']

    # For each model in the ensemble, replicate the state
    for model in models:
        for old_p_key, old_p_item in models[0].named_parameters():
            new_p_id = id(dict(model.named_parameters())[old_p_key])
            if old_p_key in existing_optimizer_state['state']:
                ensemble_optimizer_state['state'][new_p_id] = copy.deepcopy(existing_optimizer_state['state'][old_p_key])
    ensemble_optimizer.load_state_dict(ensemble_optimizer_state)
