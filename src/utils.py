from pathlib import Path
import random

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from typing import Union
def load_yaml(path: Union[str, Path]) -> dict:
    """Load a YAML configuration file."""
    path = Path(path)
    
    try:
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    path = Path(path)
    if path.exists() and path.is_file():
        raise ValueError(f"Path exists as file, not directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path