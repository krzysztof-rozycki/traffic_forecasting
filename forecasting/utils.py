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


def get_current_file_directory():
    current_file_path = os.path.abspath(__file__)
    # Get the directory in which the current file is located
    module_directory = os.path.dirname(current_file_path)
    # Get the parent directory of module_directory
    parent_directory = os.path.dirname(module_directory)
    return parent_directory

PROJECT_ROOT = get_current_file_directory()
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
features_config = load_yaml(os.path.join(PROJECT_ROOT, 'configs/features.yaml'))
models_config = load_yaml(os.path.join(PROJECT_ROOT, 'configs/models.yaml'))
random_search_config = load_yaml(os.path.join(PROJECT_ROOT, 'configs/random_search.yaml'))

train_config = load_yaml(os.path.join(PROJECT_ROOT, 'configs/train.yaml'))
train_config = adjust_path_to_project_root(train_config, PROJECT_ROOT)


def cos_cycle(x: np.ndarray, period: int):
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
