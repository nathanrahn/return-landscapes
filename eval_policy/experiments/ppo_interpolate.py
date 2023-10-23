# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
from pdb import run
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
from gym.wrappers import TimeLimit, OrderEnforcing
import tqdm
import torch
from utils import copy_model
from utils import evaluate_policy
from functools import partial
import copy
from torch.func import stack_module_state, functional_call
import envpool
import h5py

from atari_policy_variance import Agent as AtariAgent


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    # Add an argument for the save directory
    parser.add_argument("--root-dir", type=str, default="/home/nur/scratch/nonseq-exp/runs/BeamRider-v5")
    parser.add_argument('--purd-samples', type=int, default=50)
    parser.add_argument('--num-interpolation-points', type=int, default=50)
    parser.add_argument('--eval-len', type=int, default=5000)
    parser.add_argument("--env-id", type=str, default="Breakout-v5", help="the id of the environment")
    # The run name directory string for run a
    parser.add_argument("--run-a", type=str, default=None)
    # The run name directory string for run b
    parser.add_argument("--run-b", type=str, default=None)
    # The checkpoint index for run a
    parser.add_argument("--ckpt-a", type=int, default=None)
    # The checkpoint index for run b
    parser.add_argument("--ckpt-b", type=int, default=None)

    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="nonseq-exp",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default='nonseq-exp',
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=300000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
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
    parser.add_argument("--clip-coef", type=float, default=0.2,
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
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def get_agent_params(root_dir, run_subdir, ckpt_idx):
    # filepath is root_dir / run_subdir / "checkpoints" / ckpt_x-of-10.pt where x is ckpt_idx
    fpath = os.path.join(root_dir, run_subdir, "checkpoints", f"ckpt_{ckpt_idx}-of-10.pt")
    return torch.load(fpath)["model_state_dict"]

def interpolate_params(params_a, params_b, num_interpolations):
    """ This function takes two sets of model parameters (as pytorch state_dicts) and interpolates between them"""
    frac_b = np.linspace(0, 1, num_interpolations, endpoint=True)
    all_params = []
    for frac in frac_b:
        interp_params = {}
        for k, v in params_a.items():
            interp_params[k] = (1-frac)*v + frac*params_b[k]
        all_params.append(interp_params)
    return all_params

def parallel_eval(new_agent, num_purd_samples, eval_envs, device, alpha=3e-4):
    purd = np.zeros(num_purd_samples)

    # Copies of the agent for parallel evaluation
    agents = [copy.deepcopy(new_agent) for _ in range(num_purd_samples)]
    for agent in agents:
        # Perturb the parameters by constructing a new state dict and loading it
        perturbed_params = {}
        for k, v in agent.state_dict().items():
            perturbed_params[k] = v + alpha * torch.randn_like(v)
        agent.load_state_dict(perturbed_params)

    params, buffers = stack_module_state(agents)

    # Create vmapped agent for updates and update it in parallel
    def fmodel(params, buffers, x):
        return functional_call(copy.deepcopy(new_agent), (params, buffers), (x,))
    vmapped_agent = partial(torch.vmap(fmodel), params, buffers)

    purd[:] = evaluate_policy(vmapped_agent,
                                eval_envs,
                                device,
                                max_len=args.eval_len)

    return purd


def get_string(run_a, run_b, ckpt_a, ckpt_b):
    return f"{run_a}_ckpt_{ckpt_a}_to_{run_b}_ckpt_{ckpt_b}"

def do_interpolation(
    root_dir, # The root directory containing all the run subdirectories
    run_a, # Name of the first run subdirectory
    run_b, # Name of the second run subdirectory
    ckpt_a, # Checkpoint index of the first run, from 1-10
    ckpt_b, # Checkpoint index of the second run, from 1-10
    args
):
    params_a = get_agent_params(root_dir, run_a, ckpt_a)
    params_b = get_agent_params(root_dir, run_b, ckpt_b)

    all_params = interpolate_params(params_a, params_b, args.num_interpolation_points)

    # Now we create an array to store the results
    results = np.zeros((len(all_params), args.purd_samples))

    # eval env setup
    eval_envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.purd_samples,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
        noop_max=1, # no random noops
        repeat_action_probability=0.0 # no sticky actions
    )
    eval_envs.num_envs = args.purd_samples
    eval_envs.single_action_space = eval_envs.action_space
    eval_envs.single_observation_space = eval_envs.observation_space
    BaseAgent = AtariAgent

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Now we loop through (using tqdm) the parameters and evaluate each one
    for i, params in tqdm.tqdm(list(enumerate(all_params))):
        agent = BaseAgent(eval_envs).to(device)
        agent.load_state_dict(params)
        new_agent = copy_model(agent, eval_envs).to(device)
        results[i, :] = parallel_eval(new_agent, args.purd_samples, eval_envs, device)

    # Now we save the results
    # Open the file pairwise_interpolation.h5 in append mode
    with h5py.File(os.path.join(args.root_dir, "pairwise_interpolation.h5"), "a") as f:
        # Create a new group for this interpolation
        grp = f.create_group(get_string(run_a, run_b, ckpt_a, ckpt_b))
        # Save the results
        grp.create_dataset("results", data=results)
    return results


if __name__ == "__main__":
    args = parse_args()

    do_interpolation(args.root_dir, args.run_a, args.run_b, args.ckpt_a, args.ckpt_b, args)