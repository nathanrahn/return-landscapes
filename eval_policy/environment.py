from multiprocessing.sharedctypes import Value
import os
from pathlib import Path
import socket


def on_cc():
    return "CC_CLUSTER" in os.environ


def on_mila():
    if "HOME" in os.environ:
        if "mila" in os.environ["HOME"]:
            return True
    return False


def which_cluster(hostname: str) -> str:
    if "int.ets1" in hostname:
        return "beluga"
    if "narval" in hostname:
        return "narval"
    raise ValueError("Unknown cluster")


def my_cluster():
    return which_cluster(socket.gethostname())


def on_cluster():
    return on_cc() or on_mila()


def scratch():
    if on_cluster():
        return Path(os.environ["SCRATCH"])
    return Path.home() / "scratch"


def get_run_dir():
    if on_cluster():
        return scratch() / "nonseq-exp"
    return Path(os.getcwd())


def get_output_dir():
    if "SLURM_TMPDIR" in os.environ:
        return Path(os.environ["SLURM_TMPDIR"])
    return Path(os.getcwd())
