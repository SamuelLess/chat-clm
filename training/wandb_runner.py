import os
import subprocess
import json
from typing import Dict, Any
import typer
from dotenv import load_dotenv
import wandb

from options import DEFAULT_TRAINING_OPTIONS, SWEEP_CONFIG

app = typer.Typer()


def load_env_vars() -> tuple:
    load_dotenv(dotenv_path="../.env")
    train_bin = os.getenv("TRAIN_BIN")
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_entity = os.getenv("WANDB_ENTITY")
    if not train_bin:
        raise RuntimeError("TRAIN_BIN environment variable isn't set.")
    if not wandb_project:
        raise RuntimeError("WANDB_PROJECT environment variable not set.")
    if not wandb_entity:
        raise RuntimeError("WANDB_ENTITY environment variable not set.")
    return train_bin, wandb_project, wandb_entity


def release_compile():
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    parent_of_parent_dir = os.path.dirname(parent_dir)
    proc = subprocess.Popen(
        ["cargo", "build", "--release"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=parent_of_parent_dir,
    )
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Training with binary failed: {stderr}")
    print("Release compile finished.")


def run_train_bin(train_bin: str, options: Dict[str, Any]) -> Dict[str, Any]:
    release_compile()
    # Serialize options as JSON string for stdin
    options_str = json.dumps(options)
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    parent_of_parent_dir = os.path.dirname(parent_dir)
    print(f"Running training binary: {train_bin}")
    proc = subprocess.Popen(
        [train_bin, "train"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        cwd=parent_of_parent_dir,
    )
    stdout, stderr = proc.communicate(options_str)
    if proc.returncode != 0:
        raise RuntimeError(f"Training with binary failed: {stderr}")
    # Parse only the last line of stdout as JSON
    lines = stdout.strip().split("\n")
    if not lines:
        raise RuntimeError("No output from training binary.")
    last_line = lines[-1]
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        raise RuntimeError(f"Last line of output is not valid JSON: {last_line}")


@app.command()
def run():
    """
    Run the training binary with the given options and log results to wandb.
    """
    train_bin, wandb_project, wandb_entity = load_env_vars()
    options_dict = DEFAULT_TRAINING_OPTIONS.copy()
    wandb.init(project=wandb_project, entity=wandb_entity, config=options_dict)
    print(f"wandb config: {dict(wandb.config)}")
    name = wandb.run.name
    result = run_train_bin(train_bin, wandb.config._as_dict() | {'model_id': name})
    # print(f"Training result: {result}")
    wandb.log(json.loads(result))
    wandb.finish()
    typer.echo("Run complete and logged to wandb.")


@app.command()
def sweep(count: int = 15, sweep_id: str = None):
    """
    Run a sweep with the given options and log results to wandb.
    """
    train_bin, wandb_project, wandb_entity = load_env_vars()
    # options_dict = DEFAULT_TRAINING_OPTIONS.copy()
    # wandb.init(project=wandb_project, entity=wandb_entity, config=options_dict)
    if sweep_id is None:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=wandb_project)
    wandb.agent(sweep_id, function=run, count=count)
    wandb.finish()
    typer.echo("Sweep complete and logged to wandb.")


if __name__ == "__main__":
    app()
