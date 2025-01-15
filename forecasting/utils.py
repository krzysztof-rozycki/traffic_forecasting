import numpy as np
from numpy import sin, cos, pi
import pandas as pd
import yaml
import pickle
import os


def load_yaml(file_path: str):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def adjust_path_to_project_root(config, project_root):
    return {key: (os.path.join(project_root, value) if key.endswith('path') else value)
            for key, value in config.items()}


def get_current_file_directory():
    config_path = os.getenv('FORECASTING_CONFIG_PATH')
    return config_path

config_path = get_current_file_directory()
features_config = load_yaml(os.path.join(config_path, 'configs/features.yaml'))
models_config = load_yaml(os.path.join(config_path, 'configs/models.yaml'))
random_search_config = load_yaml(os.path.join(config_path, 'configs/random_search.yaml'))

train_config = load_yaml(os.path.join(config_path, 'configs/train.yaml'))
train_config = adjust_path_to_project_root(train_config, config_path)


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
