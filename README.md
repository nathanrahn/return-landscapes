# Policy Optimization in a Noisy Neighborhood: On Return Landscapes in Continuous Control

This repository contains code for reproducing the experiments in our NeurIPS 2023 paper, ["Policy Optimization in a Noisy Neighborhood: On Return Landscapes in Continuous Control"](https://arxiv.org/abs/2309.14597) by Nate Rahn*, Pierluca D'Oro*, Harley Wiltzer, Pierre-Luc Bacon, and Marc G. Bellemare.

**Abstract:** Deep reinforcement learning agents for continuous control are known to exhibit significant instability in their performance over time. 
In this work, we provide a fresh perspective on these behaviors by studying the return landscape: the mapping between a policy and a return.
We find that popular algorithms traverse *noisy neighborhoods* of this landscape, in which a single update to the policy parameters leads to a wide range of returns.
By taking a distributional view of these returns, we map the landscape, characterizing failure-prone regions of policy space and revealing a hidden dimension of policy quality. 
We show that the landscape exhibits surprising structure by finding simple paths in parameter space which improve the stability of a policy.
To conclude, we develop a distribution-aware procedure which finds such paths, navigating away from noisy neighborhoods in order to improve the robustness of a policy.
Taken together, our results provide new insight into the optimization, evaluation, and design of agents.

![return_landscapes](https://github.com/nathanrahn/return-landscapes/assets/77698360/b4926fee-b71f-495f-bf84-2b6f22e5bd27)


## Hydra

Our code is built on the [Hydra](https://hydra.cc/) configuration framework. The commands listed below automatically perform sweeps over the cross product of parameters specified as lists (however, note that `param=1,2,3` will perform a sweep, while `param="[1,2,3]"` reflects setting a parameter to a particular list value). See `run_algorithm/conf/job_config` for an example of a configuration which can be used to run such a sweep on a SLURM cluster by specifying `job_config=cc_gpu` -- it will generate an array job element for each member of the sweep.

## Weights and Biases

Our code relies on Weights and Biases for logging. To run the code, you will need to set up a free `wandb` project, and update the `wandb.project_name` and `wandb.entity_name` entries in run `run_algorithm/conf/config.yaml`, `eval_policy/conf/config.yaml`, and `improve_ltp/conf/config.yaml` to point to your `wandb` project and username. Additionally, set the `WANDB_API_KEY` environment variable for authentication.

## Setting up the environment

Create a virtual environment and install the dependencies using `pip install -r requirements.txt`.

## Running the baseline algorithms

This command runs TD3 for the default value of 1M steps on the 4 Brax environments `halfcheetah,walker2d,hopper` and `ant`. The code runs 20 seeds in parallel inside of a single job using JAX `vmap`. The parameter `experiment.agent_log_starts` sets the points at which the policy will be logged. The policy will be periodically evaulated at intervals of `experiment.eval_interval=10000` and logged to `wandb`.

```bash
cd run_algorithm

python main.py --multirun experiment.algo=ddpg experiment.env_name=halfcheetah,walker2d,hopper,ant experiment.reset_interval=1000 experiment.ddpg_config.exploration_noise=0.1 experiment.num_seeds=20 experiment.agent_log_starts="[50000, 150000, 250000, 350000, 450000, 550000, 650000, 750000, 850000, 950000]" experiment.eval_interval=10000 experiment.agent_num_logs=1
```
To run Soft Actor-Critic, set `experiment.algo=sac`.

To run PPO, use the following commmand:
```bash
cd run_algorithm

python main.py --multirun experiment=train_ppo experiment.env_id=ant,halfcheetah,hopper,walker2d experiment.agent_num_logs=10 random_seed=0 experiment.num_seeds=20
```


## Running the policy evaluation stage

The following command will perform massively parallel evaluation of the policies logged by the baseline runs under a single TD3 update. In particular, the command sweeps over multiple wandb run names (replace these with those that were created by the previous command) corresponding to runs of the baseline algorithm for different environments, and the seeds within those runs. `experiment.ckpt_indexes` is the list of previously logged checkpoints, `eval_episodes` refers to the number of independent updates to estimate the post-update return distribution, and `experiment.batch_size` refers to the number of rollouts that will be done in parallel to evaluate the post-update policies. Finally, `experiment.noise_amounts="[3e-4]"` sets the learning rate used for the update.
```bash
cd eval_policy

python main.py --multirun experiment=policy_variance experiment.run_name="'<run_name_1>','<run_name_2>','<run_name_3>','<run_name_4>'" experiment.seed="range(20)" experiment.ckpt_indexes=[50000,150000,250000,350000,450000,550000,650000,750000,850000,950000] experiment.eval_episodes=1000 experiment.batch_size=1000 experiment.noise_type="td3_optimization" experiment.noise_amounts="[3e-4]" &
```

Same command, but using the SAC update:
```bash
cd eval_policy

python main.py --multirun experiment=policy_variance experiment.run_name="''<run_name_1>','<run_name_2>','<run_name_3>','<run_name_4>'" experiment.seed="range(20)" experiment.ckpt_indexes=[50000,150000,250000,350000,450000,550000,650000,750000,850000,950000] experiment.eval_episodes=1000 experiment.batch_size=1000 experiment.noise_type="sac_optimization" experiment.noise_amounts="[3e-4]" &
```

Same thing for PPO: The `ckpt_indexes` that are indicated here are the steps at which checkpoints will be logged using the PPO baseline as above.
```bash
cd eval_policy

python main.py --multirun experiment.ckpt_indexes=[5963776,11960320,17956864,23953408,29949952,35946496,41943040,47939584,53936128,59932672] experiment=policy_variance experiment.run_name="'<run_name_1>','<run_name_2>','<run_name_3>','<run_name_4>'" experiment.seed="range(20)" experiment.eval_episodes=1000 experiment.batch_size=50 experiment.noise_type="'ppo_optimization'" experiment.noise_amounts="[0]"
```

**Outputs of the code:** These functions will log the results of the evaluations to a particular `result_dir` (controlled by Hydra), which is also logged to `wandb` to the `result_dir` config field. In this directory, within the subdirectory corresponding to the seed of interest (e.g. `seed0`), you will find a file called `s0_returns_{ckpt_index}.h5` which contains the post-update returns produced. The data is stored at a particular path in this file [described by this line of code](https://github.com/nathanrahn/return-landscapes/blob/main/eval_policy/experiments/policy_variance.py#L603) that is based on the parameters passed to the evaluation job above. The structure of the data is a flat array of length `eval_episodes` containing returns obtained by the policy under independent updates.

## Running the experiments for results on interpolation

The following command is used to run the experiments which study the return landscape at interpolation points between logged policies at the default checkpoints for TD3. The code considers all possible pairs of checkpoints `(c1, c2)` where the checkpoints come from the same seed, when `experiment.same_seed=true`, and from two different seeds, when `experiment.same_seed=false`. For `experiment.num_points` interpolation points, the code will generate an estimate of the post-update return distribution at these points and log the results to a file.
```bash
cd eval_policy

python main.py --multirun experiment=policy_interpolation_seed experiment.num_purd_samples=100 experiment.num_points=100 experiment.batch_size=10 experiment.seed=0,1,2,3,4,5,6,7,8,9 experiment.same_seed=false,true experiment.first_half=false,true experiment.run_name="'<run_name_1>','<run_name_2>','<run_name_3>','<run_name_4>'" 
```

**Outputs of the code:** The results of the experiment are logged to the same per-seed result directory as the policy evaluation stage experiments. The log file is called `policy_interpolation_purd.h5` and the logging structure is described [here](https://github.com/nathanrahn/return-landscapes/blob/main/eval_policy/experiments/policy_interpolation_seed.py#L149). In particular, for each interpolation pair, we store an array of size `num_purd_samples x num_points`.

## Running the experiments for LTP reduction
To evaluate the TD3 Post-Update-CVaR Rejection algorithm (Algorithm 1 in the paper), first generate starting policies following the instructions [above](#running-the-baseline-algorithms). Note down their respective run names from wandb, as well as the checkpoint indices at which the learned policies are saved. For example, if you saved a checkpoint at step 650000 in the wandb run `'16904957'`, we will refer to `ROOT_POLICY_RUN="'16904957'"` and `ROOT_POLICY_CKPT=650000`. Experiments can then be run for example as follows,

```bash
cd improve_ltp

# The wandb run name for policy training/checkpointing
export ROOT_POLICY_RUN="'16904957'"

# The seeds and checkpoints identifying the root policies
export ROOT_POLICY_CKPT="50000,150000,250000,350000,450000,550000,650000,750000,850000,950000"
export ROOT_POLICY_SEEDS="range(10)"

# The total number of gradient steps for the experiment.
# Updates can be rejected in this algorithm, so this is not the number of applied updates,
# but rather the number of proposed updates.
export MAX_GRADIENT_STEPS=40

# The CVaR level used to accept/reject updates on the basis of the post-update return distribution
export CVAR=0.1

# The seeds to be used for independent runs of this experiment
export REJECT_LTP_SEEDS=range(10)

python main.py --multirun \
	experiment=td3_reject_ltp \
	experiment.root_policy_conf.run_name=$ROOT_POLICY_RUN \
	experiment.root_policy_conf.ckpt=$ROOT_POLICY_CKPT \
	experiment.root_policy_conf.seed=$ROOT_POLICY_SEEDS \
	experiment.eval_episodes=1000 \
	experiment.batch_size=1000 \
	experiment.max_steps=$MAX_GRADIENT_STEPS \
	experiment.heuristic=cvar \
	experiment.check_interval=1 \
	experiment.cvar_alpha=$CVAR \
    experiment.seed=$REJECT_LTP_SEEDS &

```

This script logs statistics of the post-update return distributions over the course of the update steps (in this case, 40 update steps) in wandb. The metrics labeled `PURD mean` and `PURD LTP` are respectively the mean and the LTP of the post-update return distributions of the policy parameters over the course of the experiment.

## Atari experiments

In the appendix, we also replicate our findings for discrete-control PPO on Atari environments. The code is based on `cleanrl` and can be run as follows.

First, create a new virtual environment, since the Atari experiments require a different set of dependencies. Then, install them using `pip install -r atari_requirements.txt`.

Run the baseline:
```bash
cd run_algorithm

python experiments/ppo_atari_envpool.py --num-checkpoints=10 --track=true --seed=5 --env-id=Breakout-v5
```
This will create a run directory in the `runs/` folder within the project corresponding to the run, and the run directory will have a subfolder called `checkpoints/`. 

```bash
cd eval_policy

python experiments/atari_policy_variance.py --track true --ckpt-path <PATH_TO_CHECKPOINTS_FOLDER> --purd-samples 1000 --outer-batch-size 100 --env-id Breakout-v5 --eval-len 10000 --ckpt-idxs 1 2 3 4 5 6 7 8 9 10
```
This saves a file called `ckpt_{ckpt_idx}_purd.h5` containing the samples from the post-update return distribution. The precise log format is [here](https://github.com/nathanrahn/return-landscapes/blob/main/eval_policy/experiments/atari_policy_variance.py#L345).

Finally, to run the interpolation experiment on Atari, copy the run directories corresponding to a given environment but different seeds to a new environment specific folder. For example, if we have run directories `runs/Breakout-v5__ppo_atari_envpool__1__1691184179/` and `runs/Breakout-v5__ppo_atari_envpool__2__1691184179/` to start, make the directory structure like `runs/Breakout-v5/Breakout-v5__ppo_atari_envpool__1__1691184179/` and `runs/Breakout-v5/Breakout-v5__ppo_atari_envpool__2__1691184179/`.

Then, you can run the command as follows, which computes the estimated post-update return distribution at interpolation points between two specific policies:
```bash
cd eval_policy

python experiments/ppo_interpolate.py --root-dir <e.g. runs/Breakout-v5> --run-a <e.g. Breakout-v5__ppo_atari_envpool__1__1691184179> --run-b <e.g. Breakout-v5__ppo_atari_envpool__2__1691184179> --ckpt-a 10 --ckpt-b 7 --env-id Breakout-v5 --eval-len 1000
```
The results of the experiment are simiarly logged to a file called `pairwise_interpolation.h5` in the `root-dir`, with format described [here](https://github.com/nathanrahn/return-landscapes/blob/main/eval_policy/experiments/ppo_interpolate.py#L198).

