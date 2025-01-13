import pandas as pd
from sklearn.model_selection import train_test_split
from forecasting.utils import train_config, features_config


def import_data(file_path: str):
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame):
    df_modified = pd.get_dummies(df, columns=['weather_main', 'weather_description'])

    # change format of dummy variables from bool to numeric and combine into one data frame
    weather_dummies_columns = df_modified.filter(regex='^weather').columns
    df_dummies = df_modified[weather_dummies_columns].astype(int)
    df_no_dummies = df_modified.drop(columns=weather_dummies_columns)
    df_full = pd.concat([df_no_dummies, df_dummies], axis=1)

    return df_full


def split_into_x_y(df):
    target_name = features_config['target']
    X = df.drop(target_name, axis=1)
    y = df[target_name].values.ravel()  # ravel() is required to avoid warnings from the sklearn package
    return X, y


def make_train_test_split(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_config['train_size'],
        shuffle=False
    )
    return X_train, X_test, y_train, y_test
