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


<<<<<<< Updated upstream
PROJECT_ROOT = Path(__file__).resolve().parent.parent
features_config = load_yaml(os.path.join(PROJECT_ROOT, 'configs/features.yaml'))
models_config = load_yaml(os.path.join(PROJECT_ROOT, 'configs/models.yaml'))
random_search_config = load_yaml(os.path.join(PROJECT_ROOT, 'configs/random_search.yaml'))
=======
def get_current_file_directory():
    config_path = os.getenv('FORECASTING_CONFIG_PATH')
    return config_path

config_path = get_current_file_directory()
print(f"PROJECT_ROOT: {config_path}")
features_config = load_yaml(os.path.join(config_path, 'configs/features.yaml'))
models_config = load_yaml(os.path.join(config_path, 'configs/models.yaml'))
random_search_config = load_yaml(os.path.join(config_path, 'configs/random_search.yaml'))
>>>>>>> Stashed changes

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
