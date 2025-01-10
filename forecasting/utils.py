from numpy import sin, cos, pi
import pandas as pd
import yaml


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

features_config = load_yaml('../configs/features.yaml')
models_config = load_yaml('../configs/models.yaml')
random_search_config = load_yaml('../configs/random_search.yaml')
train_config = load_yaml('../configs/train.yaml')


def sin_cycle(x, period):
    if isinstance(x, pd.Series):
        x = x.values
    result = cos(2 * pi * x / period)
    return result.reshape(-1, 1)