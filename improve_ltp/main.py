import os
os.environ["MUJOCO_GL"] = "osmesa"

import contextlib
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate, get_original_cwd
import wandb
import git
from pip._internal.operations import freeze

import shutil
import logging
from hydra.core.hydra_config import HydraConfig
import uuid

from environment import get_output_dir, get_run_dir, scratch

log = logging.getLogger(__name__)

UNIQUE_ID = str(uuid.uuid4())
CONF_DIR = scratch() / "configs" / UNIQUE_ID


def require_clean_git_status(repo_path):
    repo = git.Repo(repo_path)
    if repo.is_dirty(untracked_files=True):
        raise RuntimeError("Non-local experiment run requires clean git status.")


def get_requirements_dict():
    requirement_strings = freeze.freeze()
    requirements = {}
    for string in requirement_strings:
        try:
            package, version = string.split("==")
        except ValueError:
            package, version = string.split(" @ ")
        requirements[package] = version
    return {"requirements": requirements}


class CopyToResultDir(object):
    def __init__(self, src, dest):
        self.src = src
        self.dest = dest

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        if self.src != self.dest:
            shutil.copytree(self.src, self.dest, dirs_exist_ok=True)


@hydra.main(config_path=CONF_DIR, config_name="config")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    multirun_dir = os.path.join(get_original_cwd(), hydra_cfg.sweep.dir)
    output_dir = get_output_dir()
    result_dir = os.getcwd()

    other_configs_dict = {
        "multirun_dir": multirun_dir,
        "output_dir": output_dir,
        "result_dir": result_dir,
    }
    requirements_dict = get_requirements_dict()
    config_dict = {**config_dict, **other_configs_dict, **requirements_dict}

    os.environ["WANDB_START_METHOD"] = "thread"
    if cfg.use_wandb:
        run = wandb.init(
            project=cfg.wandb.project_name,
            entity=cfg.wandb.entity_name,
            config=config_dict,
            mode=cfg.wandb.mode,
            dir=str(output_dir),
            name=(
                str(os.environ["SLURM_ARRAY_JOB_ID"]) + "_" + str(os.environ["SLURM_ARRAY_TASK_ID"])
                if "SLURM_ARRAY_TASK_ID" in os.environ
                else None
            ),
        )
    else:
        run = contextlib.nullcontext()

    log.info(f"Running in working directory: {os.getcwd()}")

    with run:
        with CopyToResultDir(str(output_dir), str(result_dir)):

            log.info(f"Run config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

            log.info(f"Creating experiment: {cfg.experiment._target_}")
            experiment = instantiate(cfg.experiment)
            log.info(f"Running experiment: {cfg.experiment._target_}")
            experiment.run(output_dir, result_dir)
            log.info(f"Finished running experiment.")


if __name__ == "__main__":
    shutil.copytree("conf", CONF_DIR, dirs_exist_ok=True)
    OmegaConf.register_new_resolver("custom_run_dir", get_run_dir)
    main()
