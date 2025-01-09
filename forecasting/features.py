import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def reformat_columns(df: pd.DataFrame):
    # changes format from object to date time
    df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M')
    return df


def make_features(df: pd.DataFrame):
    df = df.copy()
    df['hour'] = df['date_time'].dt.hour.astype(str)
    df['month'] = df['date_time'].dt.month.astype(str)
    df['day_of_week'] = (df['date_time'].dt.day_of_week+1).astype(str)
    return df


def make_categorical_pipeline():
    categorical_features = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
            ('ohe', OneHotEncoder(drop='first', min_frequency=0.1, handle_unknown='infrequent_if_exist'))
        ]
    )

    return categorical_features


def make_numeric_pipeline():
    numeric_features = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )

    return numeric_features


def make_col_transformer(categorical_features, numerical_features):
    col_transformer = ColumnTransformer(
        [
            ('categorical', make_categorical_pipeline(), categorical_features),
            ('numeric', make_numeric_pipeline(), numerical_features)
        ]
    )

    return col_transformer
