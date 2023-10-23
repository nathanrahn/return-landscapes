# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool
from pathlib import Path

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import tqdm
from utils import copy_model, copy_optimizer
from utils import generate_data, update_vmapped_agent, evaluate_policy, init_ensemble_optim
from functools import partial
import copy
from torch.func import stack_module_state, functional_call

import h5py

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--ckpt-idxs', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument('--purd-samples', type=int, default=15)
    parser.add_argument('--outer-batch-size', type=int, default=5)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="nonseq-exp",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="nonseq-exp",
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Pong-v5",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--eval-len", type=int, default=1000,
        help="length of eval episodes")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--num-checkpoints", type=int, default=10,
        help="the number of checkpoints to log")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def forward(self, x):
        hidden = self.network(x / 255.0)
        return self.actor(hidden), self.critic(hidden)


class Actor(nn.Module):
    def __init__(self, network, actor):
        super().__init__()
        self.network = network
        self.actor = actor

    def forward(self, x):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        return logits


def save_checkpoint(agent, optimizer, ckpt_num, num_ckpts, run_name, base_path):
    save_dir = Path(os.path.join(base_path.format(run_name), "checkpoints"))
    # Ensure save_dir exists
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_dir / f"ckpt_{ckpt_num}-of-{num_ckpts}.pt",
    )


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    base_path = './runs/{}'  #"/home/nur/scratch/nonseq-exp/runs/{}"
    if args.track:
        import wandb

        if args.ckpt_path is not None:
            run_name = f"purd_{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(base_path.format(run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(
        envs.action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    assert args.purd_samples % args.outer_batch_size == 0
    fake_train_envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.outer_batch_size*args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )

    fake_train_envs.num_envs = args.outer_batch_size
    fake_train_envs.single_action_space = envs.action_space
    fake_train_envs.single_observation_space = envs.observation_space

    obs = torch.zeros((args.num_steps, args.num_envs*args.outer_batch_size) + envs.single_observation_space.shape).to(
        device
    )
    actions = torch.zeros((args.num_steps, args.num_envs*args.outer_batch_size) + envs.single_action_space.shape).to(
        device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs*args.outer_batch_size)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs*args.outer_batch_size)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs*args.outer_batch_size)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs*args.outer_batch_size)).to(device)
    avg_returns = deque(maxlen=20)
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(fake_train_envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs*args.outer_batch_size).to(device)
    num_updates = args.total_timesteps // args.batch_size

    eval_envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.outer_batch_size,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
        noop_max=1, # no random noops
        repeat_action_probability=0.0 # no sticky actions
    )

    eval_envs.num_envs = args.outer_batch_size
    eval_envs.single_action_space = envs.action_space
    eval_envs.single_observation_space = envs.observation_space

    purd = np.zeros(args.purd_samples)
    for ckpt_idx in tqdm.tqdm(args.ckpt_idxs):
        for idx in tqdm.tqdm(range(args.purd_samples//args.outer_batch_size)):
            path = os.path.join(args.ckpt_path, f'ckpt_{ckpt_idx}-of-10.pt')
            state_dicts = torch.load(path)
            agent.load_state_dict(state_dicts['model_state_dict'])
            optimizer.load_state_dict(state_dicts['optimizer_state_dict'])
            new_agent = copy_model(agent, envs).to(device)
            new_optimizer = copy_optimizer(optimizer, new_agent)
            params, buffers = stack_module_state([new_agent]*args.outer_batch_size)

            # Init ensembel optimizer
            ensemble_optimizer = torch.optim.Adam([p for p in params.values()], lr=args.learning_rate, eps=1e-5)
            init_ensemble_optim(optimizer, ensemble_optimizer, [new_agent]*args.outer_batch_size)
            # Create vmapped agent for updates and update it in parallel
            def fmodel(params, buffers, x):
                return functional_call(copy.deepcopy(new_agent), (params, buffers), (x,))
            vmapped_agent = partial(torch.vmap(fmodel), params, buffers)

            b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = generate_data(
                agent, args.num_steps, obs,
                actions,
                logprobs,
                next_obs,
                rewards,
                dones,
                False, next_done,
                values, fake_train_envs, device, args
            )

            b_obs = b_obs.reshape(args.outer_batch_size, b_obs.shape[0]//args.outer_batch_size, *b_obs.shape[1:])
            b_logprobs = b_logprobs.reshape(args.outer_batch_size, b_logprobs.shape[0]//args.outer_batch_size, *b_logprobs.shape[1:])
            b_actions = b_actions.reshape(args.outer_batch_size, b_actions.shape[0]//args.outer_batch_size, *b_actions.shape[1:])
            b_advantages = b_advantages.reshape(args.outer_batch_size, b_advantages.shape[0]//args.outer_batch_size, *b_advantages.shape[1:])
            b_returns = b_returns.reshape(args.outer_batch_size, b_returns.shape[0]//args.outer_batch_size, *b_returns.shape[1:])
            b_values = b_values.reshape(args.outer_batch_size, b_values.shape[0]//args.outer_batch_size, *b_values.shape[1:])

            approx_kl, old_approx_kl, entropy_loss, pg_loss, v_loss = update_vmapped_agent(
                vmapped_agent, ensemble_optimizer, b_obs, b_actions, b_logprobs,
                b_advantages, b_values, b_returns, args
            )

            # Derive policies from ensemble and evaluate them
            purd[idx*args.outer_batch_size:(idx+1)*args.outer_batch_size] = evaluate_policy(vmapped_agent,
                                                                                            eval_envs,
                                                                                            device,
                                                                                            max_len=args.eval_len)
        # TODO log purd (of shape (args.purd_samples,)) here
        # Put it in an h5 file called ckpt_{ckpt_idx}_purd.h5
        with h5py.File(os.path.join(args.ckpt_path, f'ckpt_{ckpt_idx}_purd.h5'), 'w') as f:
            f.create_dataset('purd', data=purd)
    envs.close()
    writer.close()
