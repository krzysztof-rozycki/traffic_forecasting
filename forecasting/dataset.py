import pandas as pd


def import_data(file_path: str):
    return pd.read_csv(file_path)
