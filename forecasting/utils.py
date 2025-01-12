import numpy as np
from numpy import sin, cos, pi
import pandas as pd
import yaml
import pickle
from pathlib import Path
import os


def load_yaml(file_path: str):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def adjust_path_to_project_root(config, project_root):
    return {key: (os.path.join(project_root, value) if key.endswith('path') else value)
            for key, value in config.items()}


features_config = load_yaml('../configs/features.yaml')
models_config = load_yaml('../configs/models.yaml')
random_search_config = load_yaml('../configs/random_search.yaml')

train_config = load_yaml('../configs/train.yaml')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
train_config = adjust_path_to_project_root(train_config, PROJECT_ROOT)


def sin_cycle(x: np.ndarray, period: int):
    if isinstance(x, pd.Series):
        x = x.values
    result = cos(2 * pi * x / period)
    return result.reshape(-1, 1)


def save_model(model, output_name: str):
    with open(output_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_path: str):
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model
