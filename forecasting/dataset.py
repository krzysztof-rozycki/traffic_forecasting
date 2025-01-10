import pandas as pd
from sklearn.model_selection import train_test_split
from forecasting.utils import train_config


def import_data(file_path: str):
    return pd.read_csv(file_path)


def make_train_test_split(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_config['train_size'],
        shuffle=False,
        # random_state=train_config['random_state']
    )
    return X_train, X_test, y_train, y_test
