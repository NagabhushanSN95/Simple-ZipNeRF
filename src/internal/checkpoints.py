import os
import shutil
from pathlib import Path

import accelerate
import torch
import glob


def restore_checkpoint(
        checkpoint_dir: Path,
        accelerator: accelerate.Accelerator,
        logger=None
):
    dirs = sorted(checkpoint_dir.glob("*"))
    path = dirs[-1] if len(dirs) > 0 else None
    if path is None:
        if logger is not None:
            logger.info("Checkpoint does not exist. Starting a new training run.")
        init_step = 0
    else:
        if logger is not None:
            logger.info(f"Resuming from checkpoint {path.as_posix()}")
        accelerator.load_state(path.as_posix())
        init_step = int(path.stem)
    return init_step


def save_checkpoint(save_dir,
                    accelerator: accelerate.Accelerator,
                    step=0,
                    total_limit=3):
    if total_limit > 0:
        folders = glob.glob(os.path.join(save_dir, "*"))
        folders.sort()
        for folder in folders[: len(folders) + 1 - total_limit]:
            shutil.rmtree(folder)
    accelerator.save_state(os.path.join(save_dir, f"{step:06d}"))
